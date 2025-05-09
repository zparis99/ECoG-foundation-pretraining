# Coded with help from Claude :)

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import math
import os
from torch.utils.tensorboard import SummaryWriter
from matplotlib.figure import Figure
from utils import apply_mask_to_batch


def scale_signal_minmax(signal, reference):
    """
    Scale a signal to match the min/max range of a reference signal.
    """

    signal_min, signal_max = np.nanmin(signal), np.nanmax(signal)
    ref_min, ref_max = np.nanmin(reference), np.nanmax(reference)

    if signal_max == signal_min:
        return signal

    scaled = (signal - signal_min) / (signal_max - signal_min)
    scaled = scaled * (ref_max - ref_min) + ref_min
    return scaled


def interpolate_signal(signal, target_length):
    """
    Interpolate a signal to a target length.

    Parameters:
    -----------
    signal : np.ndarray
        Signal to interpolate
    target_length : int
        Desired length after interpolation

    Returns:
    --------
    np.ndarray
        Interpolated signal of length target_length
    """
    original_steps = np.arange(len(signal))
    target_steps = np.linspace(0, len(signal) - 1, target_length)
    interpolator = interp1d(original_steps, signal)
    return interpolator(target_steps)


def plot_multi_band_reconstruction(
    original_signal,
    reconstructed_signal,
    pred_t_dim,
    batch_idx=0,
    height_idx=0,
    width_idx=0,
    epoch=0,
    scale_output=False,
    seen_signal=None,
):
    """
    Plot original and reconstructed signals for all bands in a subplot grid.
    Returns the figure object instead of saving/showing.

    Parameters:
    -----------
    original_signal : np.ndarray
        Original signal of shape [batch_size, num_bands, time_steps, height, width]
    reconstructed_signal : np.ndarray
        Reconstructed signal of shape [batch_size, num_bands, time_steps * (pred_t_dim / frame_patch_size), height, width]
    pred_t_dim : int
        Number of temporal outputs in prediction
    batch_idx : int
        Index of the batch to plot
    height_idx : int
        Height position to plot
    width_idx : int
        Width position to plot
    epoch : int
        Current training epoch (for title)
    scale_output : bool
        Whether to scale the reconstructed signal to match the original
    seen_signal: Optional[np.ndarray]
        Optional signal which has nan values filled for the masked out portion of the signal.

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    num_bands = original_signal.shape[1]

    # Calculate subplot grid dimensions
    num_cols = min(3, num_bands)
    num_rows = math.ceil(num_bands / num_cols)

    # Create figure with subplots
    fig = Figure(figsize=(6 * num_cols, 4 * num_rows))
    fig.suptitle(
        f"Multi-band Signal Reconstruction (Epoch {epoch})\n"
        f"Electrode ({height_idx}, {width_idx})",
        fontsize=16,
        y=1.02,
    )
    # We generate pred_t_dim predictions per patch so setup times of predictions here.
    reconstruction_prediction_times = np.linspace(
        0, original_signal.shape[2] - 1, pred_t_dim
    ).astype(np.int64)

    # Create subplots for each band
    for band_idx in range(num_bands):
        # Extract signals for this band
        original = original_signal[batch_idx, band_idx, :, height_idx, width_idx]
        reconstructed_downsampled = reconstructed_signal[
            batch_idx, band_idx, :, height_idx, width_idx
        ]

        # Scale reconstructed signal if requested
        if scale_output:
            reconstructed_downsampled = scale_signal_minmax(
                reconstructed_downsampled, original
            )

        # Create subplot
        ax = fig.add_subplot(num_rows, num_cols, band_idx + 1)

        # Plot signals
        time_steps = np.arange(len(original))
        ax.plot(time_steps, original, label="Original", color="blue", alpha=0.7)
        if seen_signal is not None:
            seen_electrode = seen_signal[batch_idx, band_idx, :, height_idx, width_idx]
            ax.plot(
                time_steps,
                seen_electrode,
                label="Seen Signal",
                color="yellow",
                alpha=0.8,
            )
        ax.plot(
            reconstruction_prediction_times,
            reconstructed_downsampled,
            label="Reconstructed",
            color="red",
            alpha=0.7,
            linestyle="--",
        )

        # Calculate metrics
        original_no_nan = original[~np.isnan(original)]
        # Interpolate reconstructed signal
        reconstructed_no_nan = reconstructed_downsampled[
            ~np.isnan(reconstructed_downsampled)
        ]
        reconstructed_no_nan = interpolate_signal(
            reconstructed_no_nan, len(original_no_nan)
        )
        mse = np.mean((original_no_nan - reconstructed_no_nan) ** 2)
        mae = np.mean(np.abs(original_no_nan - reconstructed_no_nan))
        corr = np.corrcoef(original_no_nan, reconstructed_no_nan)[0, 1]

        # Add metrics text
        metrics_text = f"MSE: {mse:.4f}\n" f"MAE: {mae:.4f}\n" f"Corr: {corr:.4f}"
        ax.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
        )

        # Customize subplot
        ax.set_title(f"Band {band_idx}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        # Only show legend for first subplot
        if band_idx == 0:
            ax.legend()

    # Adjust layout
    fig.tight_layout()
    return fig


def save_reconstruction_plot(
    original_signal,
    reconstructed_signal,
    epoch,
    output_dir,
    log_writer=None,
    pred_t_dim=4,
    batch_idx=0,
    height_idx=0,
    width_idx=0,
    tag="signal_reconstruction",
    scale_output=False,
):
    """
    Generate, save, and optionally log to TensorBoard a multi-band signal reconstruction plot.

    Parameters:
    -----------
    original_signal : np.ndarray
        Original signal of shape [batch_size, num_bands, time_steps, height, width]
    reconstructed_signal : np.ndarray
        Reconstructed signal of shape [batch_size, num_bands, time_steps/t_patch_size, height, width]
    epoch : int
        Current epoch number
    output_dir : str
        Directory to save plot files
    writer : torch.utils.tensorboard.SummaryWriter, optional
        TensorBoard writer for logging plots
    pred_t_dim : int
        Number of predictions for each sample.
    batch_idx : int
        Index of the batch to visualize
    height_idx : int
        Height position to visualize
    width_idx : int
        Width position to visualize
    tag : str
        Tag for TensorBoard logging
    scale_output : bool
        Whether to scale the reconstructed signal to match the original
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate the figure
    fig = plot_multi_band_reconstruction(
        original_signal,
        reconstructed_signal,
        pred_t_dim=pred_t_dim,
        batch_idx=batch_idx,
        height_idx=height_idx,
        width_idx=width_idx,
        epoch=epoch,
        scale_output=scale_output,
    )

    # Save to file
    file_name = f"reconstruction_epoch_{epoch:04d}_scaled={scale_output}.png"
    save_path = os.path.join(output_dir, file_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Log to TensorBoard if writer is provided
    if log_writer is not None:
        log_writer.add_figure(tag, fig, global_step=epoch)

    plt.close(fig)
