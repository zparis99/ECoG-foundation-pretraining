import numpy as np
import torch

from ecog_foundation_model.config import ECoGDataConfig
from loader import BufferedFileRandomSampler, MultiFileECoGDataset


NUM_CHANNELS = 64
FILE_SAMPLING_FREQUENCY = 512
MAX_DATASET_LENGTH = 100


# Mock class to track how the dataset is utilized.
class MockECoGDataset:
    def __init__(self, _filepath, _config, use_cache=False):
        self.preload_calls = 0
        self.free_calls = 0
        self.dataset_len = int(np.random.randint(10, MAX_DATASET_LENGTH))

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, i):
        return 0

    def preload_data(self):
        self.preload_calls += 1

    def free_data(self):
        self.free_calls += 1


def create_dataloader(filepaths, data_config, use_cache=False):
    dataset = MultiFileECoGDataset(filepaths, data_config, use_cache=use_cache)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        sampler=BufferedFileRandomSampler(dataset),
    )
    return dataloader


def create_fake_sin_data():
    # Create fake data to be used in tests. Sin wave scaled by index of channel
    # Add sine wave scaled by electrode index to all channels.
    times = np.linspace(0, 100, 100 * FILE_SAMPLING_FREQUENCY)
    data = np.array([(i + 1) * np.sin(np.pi * times) for i in range(NUM_CHANNELS + 1)])
    return data


def test_file_dataset_can_handle_configurable_bands(dataset_creation_fn):
    config = ECoGDataConfig(
        batch_size=32,
        bands=[[4, 8], [8, 13], [13, 30], [30, 55]],
        new_fs=32,
    )
    file_dataset = dataset_creation_fn(
        config,
        data=create_fake_sin_data(),
        file_sampling_frequency=FILE_SAMPLING_FREQUENCY,
    )
    file_dataset.preload_data()

    for i in range(len(file_dataset)):
        assert file_dataset[i].shape == (
            len(config.bands),
            config.sample_length * config.new_fs,
            8,
            8,
        )


def test_file_dataset_grid_creation_returns_input_data(dataset_creation_fn):
    config = ECoGDataConfig(
        batch_size=32,
        bands=[[4, 8], [8, 13], [13, 30], [30, 55]],
        new_fs=20,
    )
    fake_data = create_fake_sin_data()
    file_dataset = dataset_creation_fn(
        config, data=fake_data, file_sampling_frequency=FILE_SAMPLING_FREQUENCY
    )

    assert np.allclose(
        file_dataset._load_grid_data(),
        fake_data[:64, :],
    )


def test_file_dataset_can_handle_missing_channel(dataset_creation_fn):
    config = ECoGDataConfig(
        batch_size=32,
        bands=[[4, 8], [8, 13], [13, 30], [30, 55]],
        new_fs=20,
    )
    # Omit "G1" from list of channels so loader will fill that channels with zeros.
    ch_names = ["G" + str(i + 1) for i in range(1, 65)]
    # To make checking easier just make data all ones.
    fake_data = np.ones((len(ch_names), 100 * FILE_SAMPLING_FREQUENCY))
    file_dataset = dataset_creation_fn(
        config,
        ch_names=ch_names,
        data=fake_data,
        file_sampling_frequency=FILE_SAMPLING_FREQUENCY,
    )

    actual_data = file_dataset._load_grid_data()
    assert np.all(np.isnan(actual_data[0]))
    assert np.allclose(
        actual_data[1:],
        np.ones((63, 100 * FILE_SAMPLING_FREQUENCY)),
    )
    assert actual_data.dtype == np.float32


def test_file_dataset_drops_short_signals(dataset_creation_fn):
    config = ECoGDataConfig(
        batch_size=32,
        bands=[[4, 8], [8, 13], [13, 30], [30, 55]],
        new_fs=20,
        sample_length=1,
    )
    # Create data for 10.5 seconds.
    fake_data = np.ones((65, int(10.5 * FILE_SAMPLING_FREQUENCY)))
    file_dataset = dataset_creation_fn(
        config, data=fake_data, file_sampling_frequency=FILE_SAMPLING_FREQUENCY
    )

    assert len(file_dataset) == 10


def test_sampler_correctly_samples_from_every_dataset():
    # Create a dataset with 10 files
    num_datasets = 10

    dataset = MultiFileECoGDataset(
        [None for _ in range(num_datasets)],
        ECoGDataConfig(),
        file_dataset=MockECoGDataset,
    )
    sampler = BufferedFileRandomSampler(dataset)

    samples_per_dataset_tracker = [0 for _ in range(num_datasets)]
    for idx in sampler:
        dataset_idx, sample_idx = idx

        # Ensure sample_idx is in the right range.
        assert sample_idx >= 0 and sample_idx < len(dataset.datasets[dataset_idx])

        # Make sure no maxed out dataset is used.
        assert samples_per_dataset_tracker[dataset_idx] < len(
            dataset.datasets[dataset_idx]
        )

        # Make sure dataset has been preloaded.
        assert dataset.datasets[dataset_idx].preload_calls == 1

        samples_per_dataset_tracker[dataset_idx] += 1
        # Make sure dataset is properly removed from list of active datasets, but not freed until
        # data has been fetched.
        if samples_per_dataset_tracker[dataset_idx] >= len(
            dataset.datasets[dataset_idx]
        ):
            assert dataset_idx not in dataset.get_active_datasets()
            assert dataset.datasets[dataset_idx].free_calls == 0

        # Ensure data is only freed after it has been accessed by the dataset.
        _val = dataset[idx]
        if samples_per_dataset_tracker[dataset_idx] >= len(
            dataset.datasets[dataset_idx]
        ):
            assert dataset.datasets[dataset_idx].free_calls == 1

    # All datasets have been called the right number of times.
    assert (
        np.array(samples_per_dataset_tracker)
        == [len(dataset_i) for dataset_i in dataset.datasets]
    ).all()

    # Every dataset has been preloaded and freed.
    for dataset in dataset.datasets:
        assert dataset.preload_calls == 1
        assert dataset.free_calls == 1


def test_dataloader_frees_data_correctly(create_fake_mne_file_fn):
    data_config = ECoGDataConfig(
        batch_size=16,
        bands=[[4, 8], [8, 13], [13, 30], [30, 55]],
        new_fs=32,
        sample_length=2,
    )
    # Two batches of data per file. Create 20 files.
    ch_names = ["G" + str(i + 1) for i in range(64 + 1)]
    fake_data = np.ones((65, int(32 * 2 * 512)))
    filepaths = []
    for i in range(20):
        filepaths.append(
            create_fake_mne_file_fn(ch_names, fake_data, 512, str(i) + "_raw.fif")
        )
    dataloader = create_dataloader(filepaths, data_config, use_cache=False)
    num_batches = 0
    for data in dataloader:
        assert data.shape == (
            data_config.batch_size,
            len(data_config.bands),
            data_config.sample_length * data_config.new_fs,
            8,
            8,
        )
        num_batches += 1

    assert (
        num_batches
        == sum([len(dataset) for dataset in dataloader.dataset.datasets]) // 16
    )

    # Ensure all data is freed by the end.
    for dataset in dataloader.dataset.datasets:
        assert dataset.signal is None
