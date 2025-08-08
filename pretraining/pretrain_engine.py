import os

import torch
from torch.utils.data import DataLoader
from mask import *
from utils import *
from plot import save_reconstruction_plot, get_seen_signal
from ecog_foundation_model.mae_st_util.models_mae import MaskedAutoencoderViT
from ecog_foundation_model.config import VideoMAEExperimentConfig

import ecog_foundation_model.mae_st_util.misc as misc
from ecog_foundation_model.mae_st_util.logging import master_print as print


def model_forward(model, signal, mask_ratio, alpha):
    """Pass signal through model after converting nan's to 0."""
    signal = torch.nan_to_num(signal)
    return model(signal, mask_ratio=mask_ratio, alpha=alpha)


def train_single_epoch(
    train_dl: DataLoader,
    epoch: int,
    accelerator,
    optimizer,
    lr_scheduler,
    device: str,
    model: MaskedAutoencoderViT,
    config: VideoMAEExperimentConfig,
    logger,
    log_writer=None,
):
    model.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "loss", misc.SmoothedValue(window_size=1, fmt="{value: 6f}")
    )
    metric_logger.add_meter("mse", misc.SmoothedValue(window_size=1, fmt="{value: 6f}"))
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "correlation", misc.SmoothedValue(window_size=1, fmt="{value: 6f}")
    )
    header = "Epoch: [{}]".format(epoch)

    gradient_accumulation_steps = config.trainer_config.gradient_accumulation_steps

    # Initialize a counter for accumulated steps
    accumulated_steps = 0

    for train_i, batch in enumerate(
        metric_logger.log_every(train_dl, config.logging_config.print_freq, header)
    ):
        # We don't call optimizer.zero_grad() here directly.
        # It will be called inside the accumulation logic or before the first step.

        signal = batch.to(device)

        padding_mask = get_padding_mask(signal, device)
        model.initialize_mask(padding_mask)

        loss, mse, _, _, _, correlation = model_forward(
            model,
            signal,
            config.video_mae_task_config.encoder_mask_ratio,
            config.video_mae_task_config.alpha,
        )

        # Scale the loss by the number of accumulation steps
        # This is important when loss is averaged over a batch to ensure correct gradient magnitude
        loss = loss / gradient_accumulation_steps

        if torch.isnan(mse):
            logger.error(
                f"Got nan loss for index {train_i}. Ignoring and continuing..."
            )
            # If we skip a batch, we should not count it towards accumulation
            continue

        accelerator.backward(loss)

        accumulated_steps += 1

        # Perform optimizer step and scheduler step only after accumulating gradients
        if (train_i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        loss_value = loss.item() * gradient_accumulation_steps
        mse_value = mse.item()
        correlation_value = correlation.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(mse=mse_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        metric_logger.update(correlation=correlation_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((train_i / len(train_dl) + epoch) * 1000)
            log_writer.add_scalar("loss/train", loss_value, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
            log_writer.add_scalar("mse/train", mse_value, epoch_1000x)
            log_writer.add_scalar("correlation/train", correlation_value, epoch_1000x)

    # Handle any remaining accumulated gradients if the total number of batches is not
    # a perfect multiple of gradient_accumulation_steps
    if accumulated_steps > 0 and accumulated_steps % gradient_accumulation_steps != 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_single_epoch(
    test_dl: DataLoader,
    epoch: int,
    device: str,
    model: MaskedAutoencoderViT,
    config: VideoMAEExperimentConfig,
    logger,
    log_writer=None,
):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        running_mse = 0.0
        running_correlation = 0.0
        for test_i, batch in enumerate(test_dl):
            signal = batch.to(device)

            padding_mask = get_padding_mask(signal, device)
            # TODO: We don't necessarily need to call this so often but for now this is easier.
            # We could be more clever with this though.
            model.initialize_mask(padding_mask)

            # TODO: Add more metrics using the other outputs.
            loss, mse, pred, mask, _, correlation = model_forward(
                model,
                signal,
                config.video_mae_task_config.encoder_mask_ratio,
                config.video_mae_task_config.alpha,
            )
            if torch.isnan(mse):
                logger.error(
                    f"Got nan loss for index {test_i}. Ignoring and continuing..."
                )
                continue

            # Draw a plot of the first electrode in the first batch so we can see the reconstruction.
            if test_i == 0:
                signal_np = signal.detach().cpu().numpy()

                pred_signal = model.unpatchify(pred)
                pred_signal_np = pred_signal.detach().cpu().numpy()

                plot_path = os.path.join(
                    config.logging_config.plot_dir, config.job_name
                )
                seen_signal = get_seen_signal(model, signal, mask)

                # Save a reconstruction plot for scaled signal as well.
                save_reconstruction_plot(
                    signal_np,
                    pred_signal_np,
                    epoch,
                    plot_path,
                    log_writer=log_writer,
                    pred_t_dim=model.pred_t_dim,
                    tag="signal_reconstruction",
                    scale_output=False,
                    seen_signal=seen_signal,
                )

            running_loss += loss.item()
            running_mse += mse.item()
            running_correlation += correlation.item()

        loss_mean = running_loss / (test_i + 1)
        mse_mean = running_mse / (test_i + 1)
        correlation_mean = running_correlation / (test_i + 1)

        # Write averages for test data.
        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This aligns with the training loop.
            """
            epoch_1000x = int((test_i / len(test_dl) + epoch) * 1000)
            log_writer.add_scalar("loss/test", loss_mean, epoch_1000x)
            log_writer.add_scalar("mse/test", mse_mean, epoch_1000x)
            # Write overall correlation, individual channels, and bands.
            log_writer.add_scalar("correlation/test", correlation_mean, epoch_1000x)

    return loss_mean, mse_mean, correlation_mean
