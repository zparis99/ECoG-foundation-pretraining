from typing import Any, Union
import time as t
from dataclasses import is_dataclass
from dataclasses import is_dataclass
import argparse
import yaml
from copy import deepcopy

from ecog_foundation_model.config import (
    dict_to_config,
    VideoMAEExperimentConfig,
    ViTConfig,
)
from ecog_foundation_model import model_registry
from ecog_foundation_model.mae_st_util.logging import setup_logging

from training_setup import system_setup, model_setup
from loader import dl_setup
from train import train_model


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


def arg_parser():
    parser = argparse.ArgumentParser()
    # General config
    parser.add_argument(
        "--config-file", type=str, default="configs/video_mae_train.yml"
    )
    return parser.parse_args()


def parse_known_args():
    parser = argparse.ArgumentParser(description="Run decoding model over lag range")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    args, unknown_args = parser.parse_known_args()
    overrides = parse_override_args(unknown_args)
    return args, overrides


def parse_override_args(unknown_args):
    """
    Parse args like --model_params.checkpoint_dir=some_path into a dictionary.
    """
    overrides = {}
    for arg in unknown_args:
        if arg.startswith("--") and "=" in arg:
            key, val = arg[2:].split("=", 1)
            overrides[key] = yaml.safe_load(val)  # preserve types like int, float, bool
    return overrides


def set_nested_attr(obj, key_path, value):
    keys = key_path.split(".")
    target = obj
    for key in keys[:-1]:
        if is_dataclass(target):
            target = getattr(target, key)
        elif isinstance(target, dict):
            target = target[key]
        else:
            raise TypeError(
                f"Unsupported type {type(target)} for intermediate key: {key}"
            )

    final_key = keys[-1]
    if is_dataclass(target):
        setattr(target, final_key, value)
    elif isinstance(target, dict):
        target[final_key] = value
    else:
        raise TypeError(f"Unsupported type {type(target)} for final key: {final_key}")


def apply_overrides(config, overrides):
    config = deepcopy(config)  # Avoid mutating original
    for key_path, value in overrides.items():
        set_nested_attr(config, key_path, value)
    return config


def format_config_job_name(config: VideoMAEExperimentConfig):
    if config.format_fields:
        format_values = [get_nested_value(config, s) for s in config.format_fields]
        config.job_name = config.job_name.format(*format_values)
    return config


def load_config_with_overrides(config_path: str, overrides: dict):
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    base_config = dict_to_config(raw_cfg, VideoMAEExperimentConfig)
    override_config = apply_overrides(base_config, overrides)
    # Use model registry to construct model if provided.
    if override_config.video_mae_task_config.model_name:
        override_config.video_mae_task_config.vit_config = (
            model_registry.model_registry[
                override_config.video_mae_task_config.model_name
            ]()
        )
    override_config = format_config_job_name(override_config)
    override_config.job_name = override_config.job_name + "_" + str(int(t.time() // 60))
    return override_config


@model_registry.register_model()
def test_model():
    return ViTConfig(
        dim=128,
        decoder_embed_dim=64,
        mlp_ratio=4.0,
        depth=6,
        decoder_depth=4,
        num_heads=4,
        decoder_num_heads=4,
        patch_size=1,
        frame_patch_size=4,
        use_cls_token=False,
        sep_pos_embed=False,
        trunc_init=False,
        no_qkv_bias=False,
        proj_drop=0.0,
        drop_path=0.0,
    )


@model_registry.register_model()
def test_model_2():
    return ViTConfig(
        dim=128,
        decoder_embed_dim=64,
        mlp_ratio=4.0,
        depth=6,
        decoder_depth=4,
        num_heads=4,
        decoder_num_heads=4,
        patch_size=2,
        frame_patch_size=4,
        use_cls_token=False,
        sep_pos_embed=False,
        trunc_init=False,
        no_qkv_bias=False,
        proj_drop=0.0,
        drop_path=0.0,
    )


@model_registry.register_model()
def test_model_small():
    return ViTConfig(
        dim=144,
        decoder_embed_dim=144,
        mlp_ratio=4.0,
        depth=24,  # Increased significantly from 10
        decoder_depth=12,  # Increased from 6
        num_heads=12,  # Increased from 6
        decoder_num_heads=12,  # Increased from 4, often good to match num_heads
        patch_size=1,  # Keep this unless you want to change how patches are formed
        frame_patch_size=8,  # Keep this unless you want to change how patches are formed
        use_cls_token=False,  # Keep as is, unless you have a specific reason to use it
        sep_pos_embed=True,  # Keep as is
        trunc_init=False,  # Keep as is
        no_qkv_bias=False,  # Keep as is
        proj_drop=0.0,  # Crucial for overfitting - keep at 0.0
        drop_path=0.0,  # Crucial for overfitting - keep at 0.0
    )


@model_registry.register_model()
def test_model_small_short():
    return ViTConfig(
        dim=144,
        decoder_embed_dim=144,
        mlp_ratio=4.0,
        depth=16,  # Increased significantly from 10
        decoder_depth=10,  # Increased from 6
        num_heads=8,  # Increased from 6
        decoder_num_heads=8,  # Increased from 4, often good to match num_heads
        patch_size=1,  # Keep this unless you want to change how patches are formed
        frame_patch_size=8,  # Keep this unless you want to change how patches are formed
        use_cls_token=False,  # Keep as is, unless you have a specific reason to use it
        sep_pos_embed=True,  # Keep as is
        trunc_init=False,  # Keep as is
        no_qkv_bias=False,  # Keep as is
        proj_drop=0.0,  # Crucial for overfitting - keep at 0.0
        drop_path=0.0,  # Crucial for overfitting - keep at 0.0
    )


@model_registry.register_model()
def test_model_large():
    return ViTConfig(
        dim=384,
        decoder_embed_dim=384,
        mlp_ratio=4.0,
        depth=24,  # Increased significantly from 10
        decoder_depth=12,  # Increased from 6
        num_heads=12,  # Increased from 6
        decoder_num_heads=12,  # Increased from 4, often good to match num_heads
        patch_size=1,  # Keep this unless you want to change how patches are formed
        frame_patch_size=8,  # Keep this unless you want to change how patches are formed
        use_cls_token=False,  # Keep as is, unless you have a specific reason to use it
        sep_pos_embed=True,  # Keep as is
        trunc_init=False,  # Keep as is
        no_qkv_bias=False,  # Keep as is
        proj_drop=0.0,  # Crucial for overfitting - keep at 0.0
        drop_path=0.0,  # Crucial for overfitting - keep at 0.0
    )


@model_registry.register_model()
def test_model_very_large():
    return ViTConfig(
        dim=768,  # Increased significantly from 384
        decoder_embed_dim=384,  # Increased from 192, consider making it equal to dim if encoder and decoder are equally important
        mlp_ratio=4.0,  # Keep or slightly increase
        depth=24,  # Increased significantly from 10
        decoder_depth=12,  # Increased from 6
        num_heads=12,  # Increased from 6
        decoder_num_heads=12,  # Increased from 4, often good to match num_heads
        patch_size=1,  # Keep this unless you want to change how patches are formed
        frame_patch_size=8,  # Keep this unless you want to change how patches are formed
        use_cls_token=False,  # Keep as is, unless you have a specific reason to use it
        sep_pos_embed=True,  # Keep as is
        trunc_init=False,  # Keep as is
        no_qkv_bias=False,  # Keep as is
        proj_drop=0.0,  # Crucial for overfitting - keep at 0.0
        drop_path=0.0,  # Crucial for overfitting - keep at 0.0
    )


@model_registry.register_model()
def test_model_very_large_more_heads():
    return ViTConfig(
        dim=768,  # Increased significantly from 384
        decoder_embed_dim=384,  # Increased from 192, consider making it equal to dim if encoder and decoder are equally important
        mlp_ratio=4.0,  # Keep or slightly increase
        depth=24,  # Increased significantly from 10
        decoder_depth=12,  # Increased from 6
        num_heads=24,  # Increased from 6
        decoder_num_heads=24,  # Increased from 4, often good to match num_heads
        patch_size=1,  # Keep this unless you want to change how patches are formed
        frame_patch_size=8,  # Keep this unless you want to change how patches are formed
        use_cls_token=False,  # Keep as is, unless you have a specific reason to use it
        sep_pos_embed=True,  # Keep as is
        trunc_init=False,  # Keep as is
        no_qkv_bias=False,  # Keep as is
        proj_drop=0.0,  # Crucial for overfitting - keep at 0.0
        drop_path=0.0,  # Crucial for overfitting - keep at 0.0
    )


@model_registry.register_model()
def test_model_very_large_patch_size_2_frame_4():
    return ViTConfig(
        dim=768,  # Increased significantly from 384
        decoder_embed_dim=384,  # Increased from 192, consider making it equal to dim if encoder and decoder are equally important
        mlp_ratio=4.0,  # Keep or slightly increase
        depth=24,  # Increased significantly from 10
        decoder_depth=12,  # Increased from 6
        num_heads=12,  # Increased from 6
        decoder_num_heads=12,  # Increased from 4, often good to match num_heads
        patch_size=2,  # Keep this unless you want to change how patches are formed
        frame_patch_size=4,  # Keep this unless you want to change how patches are formed
        use_cls_token=False,  # Keep as is, unless you have a specific reason to use it
        sep_pos_embed=True,  # Keep as is
        trunc_init=False,  # Keep as is
        no_qkv_bias=False,  # Keep as is
        proj_drop=0.0,  # Crucial for overfitting - keep at 0.0
        drop_path=0.0,  # Crucial for overfitting - keep at 0.0
    )


def main():

    setup_logging()

    args, overrides = parse_known_args()
    experiment_config = load_config_with_overrides(args.config, overrides)

    accelerator, device, data_type, local_rank = system_setup(
        experiment_config.trainer_config.mixed_precision
    )
    train_dl, test_dl, num_train_samples = dl_setup(experiment_config)
    model, optimizer, lr_scheduler, _ = model_setup(
        experiment_config, num_train_samples
    )

    model = train_model(
        experiment_config,
        device,
        model,
        train_dl,
        test_dl,
        optimizer,
        lr_scheduler,
        accelerator,
        data_type,
        local_rank,
    )


if __name__ == "__main__":
    main()
