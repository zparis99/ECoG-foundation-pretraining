import multiprocessing as mp
import time as t
import logging
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataclasses import is_dataclass
from typing import Any, Union

from ecog_foundation_model.config import (
    VideoMAEExperimentConfig,
    write_config_file_to_yaml,
)
from pretrain_engine import train_single_epoch, test_single_epoch


logger = logging.getLogger(__name__)


def get_nested_value(obj: Union[dict, Any], path: str) -> Any:
    fields = path.split(".")
    current = obj
    for field in fields:
        if isinstance(current, dict):
            current = current[field]
        elif is_dataclass(current):
            current = getattr(current, field)
        else:
            raise TypeError(
                f"Cannot access field '{field}' on non-dict, non-dataclass object: {current}"
            )
    return current


def train_model(
    config: VideoMAEExperimentConfig,
    device: str,
    model,
    train_dl: torch.utils.data.DataLoader,
    test_dl: torch.utils.data.DataLoader,
    optimizer,
    lr_scheduler,
    accelerator,
    data_type,
    local_rank,
):
    """
    Runs model training

    Args:
        config: experiment config for this run
        device: the gpu to be used for model training
        model: an untrained model instance with randomly initialized parameters
        train_dl: dataloader instance for train split
        test_dl: dataloader instance for test split
        num_patches: number of patches in which the input data is segmented
        optimizer: Adam optimizer instance - https://www.analyticsvidhya.com/blog/2023/12/adam-optimizer/
        lr_scheduler: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        accelerator: an accelerator instance - https://huggingface.co/docs/accelerate/en/index
        data_type: the data type to be used, we use "fp16" mixed precision - https://towardsdatascience.com/understanding-mixed-precision-training-4b246679c7c4
        local_rank: the local rank environment variable (only needed for multi-gpu training)

    Returns:
        model: model instance with updated parameters after training
    """
    best_loss = float("inf")

    torch.cuda.empty_cache()
    model, optimizer, train_dl, test_dl = accelerator.prepare(
        model, optimizer, train_dl, test_dl
    )

    os.makedirs(config.logging_config.event_log_dir, exist_ok=True)
    # Append time element to job name to differentiate between different runs.
    if config.format_fields:
        format_values = [get_nested_value(config, s) for s in config.format_fields]
        config.job_name = config.job_name.format(*format_values)
    config.job_name = config.job_name + "_" + str(int(t.time() // 60))
    log_writer = SummaryWriter(
        log_dir=os.path.join(config.logging_config.event_log_dir, config.job_name)
    )
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", config.job_name)
    best_checkpoint_dir = os.path.join(checkpoint_dir, "best_checkpoint")
    if not os.path.exists(best_checkpoint_dir):
        os.makedirs(best_checkpoint_dir, exist_ok=True)
    write_config_file_to_yaml(
        os.path.join(checkpoint_dir, "experiment_config.yml"), config
    )
    write_config_file_to_yaml(
        os.path.join(best_checkpoint_dir, "experiment_config.yml"), config
    )

    for epoch in range(config.trainer_config.num_epochs):
        start = t.time()
        with accelerator.autocast():
            model.train()
            train_single_epoch(
                train_dl,
                epoch,
                accelerator,
                optimizer,
                lr_scheduler,
                device,
                model,
                config,
                logger,
                log_writer=log_writer,
            )

            test_loss, _, _ = test_single_epoch(
                test_dl, epoch, device, model, config, logger, log_writer=log_writer
            )

            end = t.time()

            logger.info(
                "Epoch " + str(epoch) + " done. Time elapsed: " + str(end - start)
            )

        # save model checkpoints
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }

        # Save a different checkpoint for every epoch.
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"{epoch}_checkpoint.pth"))

        # Check to see if this is the best checkpoint.
        if test_loss < best_loss:
            torch.save(
                model.state_dict(), os.path.join(best_checkpoint_dir, "model.pth")
            )

    return model
