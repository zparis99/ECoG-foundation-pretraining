# TODO implement learning rate scheduler for better performance - initially Paul had implemented one, but we are using a fixed LR for now ...

import os
import math
import torch
import numpy as np
from accelerate import Accelerator

import utils
from ecog_foundation_model.config import VideoMAEExperimentConfig
from ecog_foundation_model import constants
from ecog_foundation_model.ecog_setup import create_model


def system_setup(mixed_precision="no"):
    """
    Sets up accelerator, device, datatype precision and local rank

    Args:

    Returns:
        accelerator: an accelerator instance - https://huggingface.co/docs/accelerate/en/index
        device: the gpu to be used for model training
        data_type: the data type to be used, we use "fp16" mixed precision - https://towardsdatascience.com/understanding-mixed-precision-training-4b246679c7c4
        local_rank: the local rank environment variable (only needed for multi-gpu training)
    """

    # tf32 data type is faster than standard float32
    torch.backends.cuda.matmul.allow_tf32 = True

    # seed all random functions
    seed = 42
    utils.seed_everything(seed)

    accelerator = Accelerator(split_batches=False, mixed_precision=mixed_precision)

    device = "cuda:0"

    # set data_type to match your mixed precision
    if accelerator.mixed_precision == "bf16":
        data_type = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        data_type = torch.float16
    else:
        data_type = torch.float32

    # only need this if we want to set up multi GPU training
    local_rank = os.getenv("RANK")
    if local_rank is None:
        local_rank = 0
    else:
        local_rank = int(local_rank)

    return accelerator, device, data_type, local_rank


def model_setup(config: VideoMAEExperimentConfig, num_train_samples):
    """
    Sets up model config

    Args:
        config: experiment config
        mean: mean of input data
        std: std of input data

    Returns:
        model: an untrained model instance with randomly initialized parameters
        optimizer: an Adam optimizer instance - https://www.analyticsvidhya.com/blog/2023/12/adam-optimizer/
        lr_scheduler: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        num_patches: the number of patches in which the input data is segmented
    """
    model_config = config.video_mae_task_config.vit_config

    num_frames = int(
        config.ecog_data_config.sample_length * config.ecog_data_config.new_fs
    )
    frame_patch_size = model_config.frame_patch_size
    num_patches = int(  # Defining the number of patches
        constants.GRID_SIZE**2
        * num_frames
        // model_config.patch_size
        // frame_patch_size
    )

    num_encoder_patches = int(
        num_patches * (1 - config.video_mae_task_config.encoder_mask_ratio)
    )
    num_decoder_patches = int(
        num_patches * config.video_mae_task_config.pct_masks_to_decode
    )
    print("num_patches", num_patches)
    print("num_encoder_patches", num_encoder_patches)
    print("num_decoder_patches", num_decoder_patches)

    model = create_model(config)
    utils.count_params(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    opt_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.trainer_config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        opt_grouped_parameters,
        lr=config.trainer_config.max_learning_rate,
    )

    num_batches = num_train_samples / config.ecog_data_config.batch_size
    accum_steps = config.trainer_config.gradient_accumulation_steps
    optimizer_steps_per_epoch = math.ceil(num_batches / accum_steps)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.trainer_config.max_learning_rate,
        epochs=config.trainer_config.num_epochs,
        steps_per_epoch=optimizer_steps_per_epoch,
    )

    print("\nDone with model preparations!")

    return model, optimizer, lr_scheduler, num_patches
