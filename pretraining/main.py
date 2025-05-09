import sys
from pretraining.training_setup import system_setup, model_setup
from config import create_video_mae_experiment_config_from_yaml
from loader import dl_setup
from mae_st_util.logging import setup_logging
from train import train_model


import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    # General config
    parser.add_argument(
        "--config-file", type=str, default="configs/video_mae_train.yml"
    )
    return parser.parse_args()


def main(args):

    setup_logging()

    experiment_config = create_video_mae_experiment_config_from_yaml(args.config_file)

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
