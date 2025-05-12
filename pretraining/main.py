import sys
from training_setup import system_setup, model_setup
from ecog_foundation_model.config import dict_to_config, VideoMAEExperimentConfig
from ecog_foundation_model import model_registry
from loader import dl_setup
from ecog_foundation_model.mae_st_util.logging import setup_logging
from train import train_model


import argparse
import yaml


def arg_parser():
    parser = argparse.ArgumentParser()
    # General config
    parser.add_argument(
        "--config-file", type=str, default="configs/video_mae_train.yml"
    )
    return parser.parse_args()


def create_experiment_config_from_yaml(
    yaml_file_path: str,
) -> VideoMAEExperimentConfig:
    with open(yaml_file_path, "r") as f:
        config_dict = yaml.safe_load(f)
    experiment_config = dict_to_config(config_dict, VideoMAEExperimentConfig)
    # Use model registry to construct model if provided.
    if experiment_config.video_mae_task_config.model_name:
        experiment_config.video_mae_task_config.vit_config = (
            model_registry.model_registry[
                experiment_config.video_mae_task_config.model_name
            ]()
        )

    return experiment_config


def main(args):

    setup_logging()

    experiment_config = create_experiment_config_from_yaml(args.config_file)

    accelerator, device, data_type, local_rank = system_setup(
        experiment_config.trainer_config.mixed_precision
    )
    train_dl, test_dl, num_train_samples = dl_setup(experiment_config)
    model, optimizer, lr_scheduler, _ = model_setup(
        experiment_config, device, num_train_samples
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

    args = arg_parser()
    main(args)
