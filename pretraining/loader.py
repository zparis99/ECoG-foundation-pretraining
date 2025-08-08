# TODO check filtering, envelope and resampling with Arnab, implement code such that we can flexibly load data from different patients

import numpy as np
import pandas as pd
import time as t
import os
import mne
from mne_bids import BIDSPath
from pyedflib import highlevel
import torch
from torch.utils.data import Dataset, Sampler, IterableDataset
import logging
import json
from typing import Sequence

from ecog_foundation_model.config import ECoGDataConfig, VideoMAEExperimentConfig

logger = logging.getLogger(__name__)


# TODO: change this to use metdata for setting values of fields
class ECoGFileDataset(Dataset):
    CACHE_FILE = "loader_cache.json"

    def __init__(self, path: str, config: ECoGDataConfig, metadata: dict):
        super().__init__()
        self.config = config
        self.path = path
        self.bands = config.bands
        self.fs = config.original_fs
        self.new_fs = config.new_fs
        self.sample_length = config.sample_length
        self.signal = None
        self.num_reads = 0

        assert self.bands == metadata["bands"]
        assert self.new_fs == metadata["new_fs"]

        self.max_samples = int(
            metadata["total_timepoints_at_new_fs"] / self.sample_length / self.new_fs
        )

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        if self.signal is None:
            raise RuntimeError("Signal not preloaded.")
        elif idx < 0 or idx >= len(self):
            raise ValueError(f"index {idx} not in range [0, {len(self) - 1}]")
        self.num_reads += 1

        # Exclude examples where the sample goes past the end of the signal.
        start_timepoint = int(idx * self.sample_length * self.new_fs)
        end_timepoint = int((idx + 1) * self.sample_length * self.new_fs)

        # The signal is already in the final [c, t, h, w] format
        return self.signal[:, start_timepoint:end_timepoint, :, :]

    def preload_data(self):
        logger.debug(
            "-----------------------------------------------------------------------------"
        )
        logger.debug("Reading new file: %s", self.path)
        # Call the new loading function
        self.signal = self._load_preprocessed_data()
        return self

    def free_data(self):
        logger.debug("Freeing file: %s", self.path)
        logger.debug(
            "Freeing %s, num reads: %d, length: %d",
            self.path,
            self.num_reads,
            self.max_samples,
        )
        self.num_reads = 0
        del self.signal
        self.signal = None

    # This function is no longer needed in its original form if data is preprocessed
    def sample_data(self, start_sample, end_sample) -> np.array:
        # This method will be simplified as preprocessing is done already
        # The __getitem__ now directly extracts the slice from self.signal
        raise NotImplementedError(
            "sample_data should not be called directly. Use __getitem__."
        )

    def _load_preprocessed_data(self):
        """Loads preprocessed and normalized data from a .npy file."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Preprocessed file not found: {self.path}")
        sig = np.load(self.path)
        # Ensure the loaded data is of the expected type if not already
        return np.float32(sig)


class SequentialMultiFileECoGDataset(IterableDataset):
    def __init__(
        self,
        paths_and_metadata,
        config: ECoGDataConfig,
        file_dataset=ECoGFileDataset,
        load_on_init: bool = False,
    ):
        super().__init__()
        self.paths_and_metadata = paths_and_metadata
        self.config = config

        # Create file datasets for each filepath
        self.file_datasets = []
        for path, metadata in self.paths_and_metadata:
            dataset = file_dataset(path, config, metadata)
            self.file_datasets.append(dataset)

        # Calculate total number of samples across all files
        self.total_samples = sum(len(dataset) for dataset in self.file_datasets)

        # Track current file and sample index
        self.current_file_idx = 0
        self.current_dataset = None

        if load_on_init:
            self._preload_current_file()

    def _preload_current_file(self):
        """Preload the current file dataset."""
        if self.current_dataset is not None:
            self.current_dataset.free_data()

        if self.current_file_idx < len(self.file_datasets):
            self.current_dataset = self.file_datasets[
                self.current_file_idx
            ].preload_data()
            logger.debug(
                f"Preloaded file {self.current_file_idx}: {self.paths_and_metadata[self.current_file_idx][0]}"
            )

    def __iter__(self):
        # Start with the first file
        self.current_file_idx = 0
        self.current_sample_idx = 0

        # Preload the first file
        self._preload_current_file()

        return self

    def __next__(self):
        # If we've gone through all files, raise StopIteration
        if self.current_file_idx >= len(self.file_datasets):
            raise StopIteration

        # Get the current dataset
        dataset = self.current_dataset

        # If we've gone through all samples in the current file
        if self.current_sample_idx >= len(dataset):
            # Free the current dataset
            dataset.free_data()

            # Move to the next file
            self.current_file_idx += 1
            self.current_sample_idx = 0

            # If we've gone through all files, raise StopIteration
            if self.current_file_idx >= len(self.file_datasets):
                raise StopIteration

            # Preload the next file
            self._preload_current_file()
            dataset = self.current_dataset

        # Get the sample from the current dataset
        sample = dataset[self.current_sample_idx]

        # Increment the sample index
        self.current_sample_idx += 1

        return sample

    def __len__(self):
        return self.total_samples


class MultiFileECoGDataset(Dataset):

    def __init__(
        self,
        paths_and_metadata,
        config: ECoGDataConfig,
        file_dataset=ECoGFileDataset,
        generator: torch.Generator = None,
        load_on_init: bool = False,
    ):
        super().__init__()
        self.datasets = [
            file_dataset(filepath, config, metadata)
            for filepath, metadata in paths_and_metadata
        ]
        self.max_open_files = config.max_open_files

        self.generator = generator
        self.dataset_samples = [len(dataset) for dataset in self.datasets]

        # Initialize datasets.
        self.active_datasets = []
        if load_on_init:
            self.refresh_dataset_ordering()

    def refresh_dataset_ordering(self):
        """Free any open datasets and prepare for a new epoch by changing the dataset ordering."""
        for i in self.active_datasets:
            self.datasets[i].free_data()

        # Generate ordering of datasets to be loaded from.
        self.dataset_order = torch.randperm(
            len(self.datasets), generator=self.generator
        ).tolist()

        # Initialize first batch of files
        self.active_datasets = self.dataset_order[: self.max_open_files]
        self.remaining_datasets = self.dataset_order[self.max_open_files :]
        for i in self.active_datasets:
            self.datasets[i].preload_data()

        # Track number of samples to be read from each file.
        self.num_samples_to_read = [0 for _ in range(len(self.datasets))]
        # Track number of samples actually read from file so far.
        self.num_samples_read = [0 for _ in range(len(self.datasets))]

    def get_active_datasets(self):
        return self.active_datasets

    def use_dataset(self, dataset_idx):
        """Track that the dataset at dataset_idx will be read from."""
        if self.num_samples_to_read[dataset_idx] >= self.dataset_samples[dataset_idx]:
            raise RuntimeError(
                f"Requesting too many samples from dataset {dataset_idx}"
            )
        self.num_samples_to_read[dataset_idx] += 1
        if self.num_samples_to_read[dataset_idx] == self.dataset_samples[dataset_idx]:
            self.active_datasets.remove(dataset_idx)

            # Setup new dataset.
            if self.remaining_datasets:
                self.datasets[self.remaining_datasets[0]].preload_data()
                self.active_datasets = self.active_datasets + [
                    self.remaining_datasets[0]
                ]
                self.remaining_datasets = self.remaining_datasets[1:]

    def __getitem__(self, idx: tuple[int, int]):
        """Returns the sample found by idx = (idx of dataset, idx of sample).

        idx: (dataset to fetch from, index of sample)
        """
        # Increment read counter and check if read == samples == max samples
        dataset_idx, sample_idx = idx
        value = self.datasets[dataset_idx][sample_idx]
        self.num_samples_read[dataset_idx] += 1

        # Free data when we can.
        if (
            self.num_samples_read[dataset_idx] == self.num_samples_to_read[dataset_idx]
            and self.num_samples_read[dataset_idx] == self.dataset_samples[dataset_idx]
        ):
            self.datasets[dataset_idx].free_data()

        return value

    def __len__(self):
        total = 0
        for d in self.datasets:
            total += len(d)
        return total


class BufferedFileRandomSampler(Sampler):
    def __init__(
        self,
        dataset: MultiFileECoGDataset,
        generator: torch.Generator = None,
    ):
        """Randomly samples from max_open_files in dataset at a time to include some randomness in data loading.

        dataset: dataset to read from
        generator: torch generator to use to make results reproducible
        """
        self.dataset = dataset
        self.generator = generator

    def __iter__(self):
        self.dataset.refresh_dataset_ordering()
        active_datasets = self.dataset.get_active_datasets()
        while active_datasets:
            active_datasets = self.dataset.get_active_datasets()
            dataset_idx = active_datasets[
                torch.randint(len(active_datasets), (), generator=self.generator)
            ]
            sample_idx = torch.randint(
                len(self.dataset.datasets[dataset_idx]), (), generator=self.generator
            )
            self.dataset.use_dataset(dataset_idx)

            yield (dataset_idx, sample_idx)

    def __len__(self):
        return len(self.dataset)


def split_dataframe(shuffle: bool, df: pd.DataFrame, ratio: float):
    """
    Shuffles a pandas dataframe and splits it into two dataframes with the specified ratio

    Args:
        shuffle: If true then shuffle the dataframe before splitting.
        df: The dataframe to split
        ratio: The proportion of data for the first dataframe (default: 0.9)

    Returns:
        df1: train split dataframe containing a proportion of ratio of full dataframe
        df2: test split dataframe containing a proportion of 1-ratio of the full dataframe
    """

    # # Shuffle the dataframe
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # Calculate the split index based on the ratios
    split_index = int(ratio * len(df))

    # Create the two dataframes
    df1 = df.iloc[:split_index, :]
    df2 = df.iloc[split_index:, :]

    return df1, df2


def read_raw(filename):
    """
    Reads and loads an edf file into a mne raw object: https://mne.tools/stable/auto_tutorials/raw/10_raw_overview.html

    Args:
        filename: Path to edf file

    Returns:
        raw: a mne raw instance
    """

    raw = mne.io.read_raw(filename, verbose=False)

    return raw


def get_dataset_path_info(
    data_split: pd.DataFrame,
    preprocessed_root: str,
    config: ECoGDataConfig,
) -> tuple[list[str], int, pd.DataFrame]:
    """Generates information about the data referenced in data_split.

    Args:
        data_split (pd.DataFrame): Dataframe storing references to the files to be used in this data split.
        preprocessed_root (str): Root directory where the preprocessed .npy and .json files are stored.

    Returns: (List of filepaths to be used for data_split, Number of samples for the data split, Dataframe with columns {'name': <filepath>, 'num_samples': <number of samples in file>})
    """
    split_paths_and_metadata = []
    num_samples = 0
    sample_desc = []

    for i, row in data_split.iterrows():
        # Construct paths for both .npy data and .json metadata
        data_bids_path = BIDSPath(
            root=preprocessed_root,
            datatype="ieeg",
            subject=f"{row.subject:02d}",
            task=f"part{row.task:03d}chunk{row.chunk:02d}",
            suffix="desc-preproc_norm_ieeg",
            extension=".npy",
            check=False,
        )
        data_path = str(data_bids_path.fpath)

        meta_bids_path = BIDSPath(
            root=preprocessed_root,
            datatype="ieeg",
            subject=f"{row.subject:02d}",
            task=f"part{row.task:03d}chunk{row.chunk:02d}",
            suffix="desc-preproc_norm_ieeg_meta",  # Match the new suffix
            extension=".json",
            check=False,
        )
        meta_path = str(meta_bids_path.fpath)

        if os.path.exists(data_path) and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            num_samples_in_file = (
                metadata["total_timepoints_at_new_fs"]
                / config.sample_length
                / metadata["new_fs"]
            )

            sample_desc.append(
                {
                    "name": data_path,
                    "metadata": meta_path,
                    "num_samples": num_samples_in_file,
                }
            )
            num_samples += num_samples_in_file
            split_paths_and_metadata.append((data_path, metadata))

    return split_paths_and_metadata, num_samples, pd.DataFrame(sample_desc)


def _create_train_dataloader_from_df(
    root: str,  # This `root` should now be the `preprocessed_normalized_root`
    data_files_df: pd.DataFrame,
    ecog_data_config: ECoGDataConfig,
) -> tuple[torch.utils.data.DataLoader, int, pd.DataFrame]:
    # load and concatenate data for train split
    # Pass the 'root' (which is now preprocessed_normalized_root) to get_dataset_path_info
    paths_and_metadata, num_samples, sample_desc = get_dataset_path_info(
        data_files_df,
        root,  # Pass root as preprocessed_root
        ecog_data_config,
    )
    paths_and_metadata = paths_and_metadata[
        : int(ecog_data_config.data_size * len(paths_and_metadata))
    ]
    dataset = MultiFileECoGDataset(paths_and_metadata, ecog_data_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ecog_data_config.batch_size,
        sampler=BufferedFileRandomSampler(dataset),
    )
    return dataloader, num_samples, sample_desc


def _create_test_dataloader_from_df(
    root: str, data_files_df: pd.DataFrame, ecog_data_config: ECoGDataConfig
) -> tuple[torch.utils.data.DataLoader, int, pd.DataFrame]:
    paths_and_metadata, num_samples, sample_desc = get_dataset_path_info(
        data_files_df,
        root,  # Pass root as preprocessed_root
        ecog_data_config,
    )
    dataset = SequentialMultiFileECoGDataset(paths_and_metadata, ecog_data_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ecog_data_config.batch_size,
    )
    return dataloader, num_samples, sample_desc


def dl_setup(
    config: VideoMAEExperimentConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Sets up dataloaders for train and test split using preprocessed data.
    """
    dataset_path = os.path.join(os.getcwd(), config.ecog_data_config.dataset_path)
    preprocessed_normalized_root = os.path.join(
        dataset_path, "derivatives", "preprocessed_normalized"
    )
    data = pd.read_csv(os.path.join(dataset_path, "dataset.csv"))

    # only look at subset of data
    # data = data.iloc[: int(len(data) * config.ecog_data_config.data_size), :]
    train_data, test_data = split_dataframe(
        config.ecog_data_config.shuffle,
        data,
        config.ecog_data_config.train_data_proportion,
    )

    # Pass the new preprocessed_normalized_root to _create_train_dataloader_from_df
    train_dl, num_train_samples, train_samples_desc = _create_train_dataloader_from_df(
        preprocessed_normalized_root, train_data, config.ecog_data_config
    )
    # Pass the new preprocessed_normalized_root to create_test_dataloader_from_df
    test_dl, _, test_samples_desc = _create_test_dataloader_from_df(
        preprocessed_normalized_root, test_data, config.ecog_data_config
    )

    dir = os.getcwd() + f"/results/samples/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    train_samples_desc.to_csv(
        dir + f"{config.job_name}_train_samples.csv",
        index=False,
    )

    test_samples_desc.to_csv(
        dir + f"{config.job_name}_test_samples.csv",
        index=False,
    )

    return train_dl, test_dl, num_train_samples
