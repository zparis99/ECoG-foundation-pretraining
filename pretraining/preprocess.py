import numpy as np
import pandas as pd
import os
import json
import mne
from mne_bids import BIDSPath
from tqdm.auto import tqdm

import logging

from ecog_foundation_model.config import ECoGDataConfig
from ecog_foundation_model import constants
from ecog_foundation_model.ecog_utils import (
    preprocess_and_normalize_neural_data,
)

logger = logging.getLogger(__name__)


def preprocess_and_normalize_one_chunk(
    input_filepath: str,
    output_filepath_npy: str,  # Rename for clarity
    output_filepath_meta: str,  # New argument for metadata file
    config: ECoGDataConfig,
    num_electrodes: int = 64,
    dtype=np.float32,
):
    """
    Reads one EDF file, extracts grid data, and then uses the generic function
    to filter, resample, and z-score normalize per-electrode per-band.
    Saves the result as a .npy file and associated metadata as a .json file.

    Args:
        input_filepath (str): Path to the raw EDF file.
        output_filepath_npy (str): Path where the preprocessed and normalized .npy file will be saved.
        output_filepath_meta (str): Path where the metadata .json file will be saved.
        config (ECoGDataConfig): Configuration object.
        num_electrodes (int): Expected number of electrodes.
        dtype: Data type for the output array.
    """
    logger.info(f"Processing and normalizing: {input_filepath}")

    raw = mne.io.read_raw(input_filepath, verbose=False)

    grid_ch_names = []
    for i in range(num_electrodes):
        channel = "G" + str(i + 1)
        if np.isin(channel, raw.info.ch_names):
            grid_ch_names.append(channel)

    sig = raw.get_data(picks=grid_ch_names)
    n_samples_original_fs = sig.shape[1]

    for i in range(num_electrodes):
        channel = "G" + str(i + 1)
        if not np.isin(channel, raw.info.ch_names):
            sig = np.insert(
                sig, i, np.full((n_samples_original_fs,), np.nan, dtype=dtype), axis=0
            )

    sig = sig[:num_electrodes, :]
    sig = np.float32(sig)

    final_preprocessed_signal = preprocess_and_normalize_neural_data(
        signal=sig,
        original_fs=config.original_fs,
        target_fs=config.new_fs,
        bands=config.bands,
        apply_envelope=config.env,
        num_electrodes=num_electrodes,
        dtype=dtype,
    )

    # Save the preprocessed and normalized data
    os.makedirs(os.path.dirname(output_filepath_npy), exist_ok=True)
    np.save(output_filepath_npy, final_preprocessed_signal)
    logger.info(f"Saved preprocessed data to: {output_filepath_npy}")

    # --- Save metadata to a JSON file ---
    total_timepoints_at_new_fs = final_preprocessed_signal.shape[1]

    metadata = {
        "total_timepoints_at_new_fs": total_timepoints_at_new_fs,
        "original_fs": config.original_fs,
        "new_fs": config.new_fs,
        "sample_length_secs": config.sample_length,
        "bands": config.bands,
        "num_bands": final_preprocessed_signal.shape[0],
        "grid_height": final_preprocessed_signal.shape[2],
        "grid_width": final_preprocessed_signal.shape[3],
        # Add any other relevant metadata
    }

    os.makedirs(os.path.dirname(output_filepath_meta), exist_ok=True)
    with open(output_filepath_meta, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved metadata to: {output_filepath_meta}")


def preprocess_all_ecog_data(
    config: ECoGDataConfig, dataset_root: str, output_root: str
):
    logger.info("Starting batch preprocessing of ECoG data...")
    data_csv_path = os.path.join(dataset_root, "dataset.csv")
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"dataset.csv not found at {data_csv_path}")

    all_data_df = pd.read_csv(data_csv_path)
    processed_derivatives_root = os.path.join(
        output_root, "derivatives", "preprocessed_normalized"
    )
    os.makedirs(processed_derivatives_root, exist_ok=True)

    for i, row in tqdm(
        all_data_df.iterrows(), total=len(all_data_df), desc="Processing data"
    ):
        subject = f"{row.subject:02d}"
        task = f"part{row.task:03d}chunk{row.chunk:02d}"

        input_bids_path = BIDSPath(
            root=dataset_root,
            datatype="ieeg",
            subject=subject,
            task=task,
            suffix="ieeg",
            extension=".edf",
            check=False,
        )
        input_filepath = str(input_bids_path.fpath)

        # Construct paths for both .npy and .json
        output_bids_path_npy = BIDSPath(
            root=output_root,
            datatype="ieeg",
            subject=subject,
            task=task,
            suffix="desc-preproc_norm_ieeg",
            extension=".npy",
            check=False,
        )
        output_filepath_npy = str(output_bids_path_npy.fpath)

        output_bids_path_meta = BIDSPath(
            root=output_root,
            datatype="ieeg",
            subject=subject,
            task=task,
            suffix="desc-preproc_norm_ieeg_meta",
            extension=".json",
            check=False,  # New suffix/extension for metadata
        )
        output_filepath_meta = str(output_bids_path_meta.fpath)

        os.makedirs(os.path.dirname(output_filepath_npy), exist_ok=True)
        os.makedirs(os.path.dirname(output_filepath_meta), exist_ok=True)

        # Check if both files exist
        if os.path.exists(output_filepath_npy) and os.path.exists(output_filepath_meta):
            logger.info(
                f"Skipping already processed file (npy and meta): {output_filepath_npy}"
            )
            continue

        try:
            preprocess_and_normalize_one_chunk(
                input_filepath=input_filepath,
                output_filepath_npy=output_filepath_npy,
                output_filepath_meta=output_filepath_meta,
                config=config,
                num_electrodes=constants.GRID_SIZE * constants.GRID_SIZE,
            )
        except Exception as e:
            logger.error(f"Failed to process {input_filepath}: {e}")
            continue

    logger.info("Batch preprocessing complete.")


if __name__ == "__main__":
    # Define your ECoGDataConfig instance
    ecog_config = ECoGDataConfig(
        bands=[[70, 200]],
        env=False,
        original_fs=512,
        new_fs=64,
        sample_length=1,
    )

    raw_data_root = "/scratch/gpfs/ln1144/ECoG-foundation-model/dataset_full"
    preprocessed_output_root = "/scratch/gpfs/zparis/ECoG-foundation-pretraining/dataset_full/derivatives/preprocessed_normalized"

    # Run the preprocessing
    preprocess_all_ecog_data(ecog_config, raw_data_root, preprocessed_output_root)
