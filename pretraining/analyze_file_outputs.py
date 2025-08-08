import os
import json
import torch
import psutil
import pandas as pd
from tqdm import tqdm

from ecog_foundation_model.mae_st_util.models_mae import MaskedAutoencoderViT
from ecog_foundation_model.ecog_setup import create_model, CheckpointManager
from ecog_foundation_model.config import (
    create_video_mae_experiment_config_from_yaml,
    VideoMAEExperimentConfig,
)

from loader import SequentialMultiFileECoGDataset
from mask import get_padding_mask


def get_system_ram_usage():
    """Prints overall system RAM usage."""
    ram = psutil.virtual_memory()
    print(f"Total RAM: {round(ram.total / (1024**3), 2)} GB")
    print(f"Used RAM: {round(ram.used / (1024**3), 2)} GB")
    print(f"Free RAM: {round(ram.free / (1024**3), 2)} GB")
    print(f"RAM Usage Percentage: {ram.percent}%")
    print(f"Available RAM: {round(ram.available / (1024**3), 2)} GB")


def get_current_process_ram_usage():
    """Prints RAM usage of the current Python script."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_used_mb = mem_info.rss / (1024 * 1024)
    print(f"Current Python process RAM usage: {round(ram_used_mb, 2)} MB")


def model_forward(model, signal, mask_ratio, alpha):
    """Pass signal through model after converting nan's to 0."""
    signal = torch.nan_to_num(signal)
    return model(signal, mask_ratio=mask_ratio, alpha=alpha)


def get_list_of_tensors_size_mb(tensor_list):
    """
    Calculates the total memory size of a list of PyTorch tensors in megabytes.

    Args:
        tensor_list (list): A list containing torch.Tensor objects.

    Returns:
        float: The total memory size in megabytes.
    """
    total_bytes = 0
    for tensor in tensor_list:
        total_bytes += tensor.nelement() * tensor.element_size()

    total_megabytes = total_bytes / (1024 * 1024)
    return total_megabytes


def stack_and_pad(tensor_list, padding_value=0):
    """
    Pads tensors to the maximum first dimension and then stacks them.

    Args:
        tensor_list (list): A list of torch.Tensor objects.
        padding_value (int or float): The value to use for padding.

    Returns:
        torch.Tensor: A single stacked tensor.
    """
    if not tensor_list:
        return torch.tensor([])

    max_dim0 = max(t.shape[0] for t in tensor_list)
    padded_tensors = []

    for t in tensor_list:
        if t.shape[0] < max_dim0:
            # Create a padding tensor
            padding_shape = (max_dim0 - t.shape[0], *t.shape[1:])
            padding = torch.full(
                padding_shape,
                padding_value,
                dtype=t.dtype,
                device=t.device,
            )
            padded_tensors.append(torch.cat((t, padding), dim=0))
        else:
            padded_tensors.append(t)

    return torch.stack(padded_tensors)


def collect_model_outputs(model, config: VideoMAEExperimentConfig, dataloader, device):
    all_embeddings = []
    all_preds = []
    all_targets = []
    all_latents = None

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Collecting outputs")):
            batch = batch.to(device)

            padding_mask = get_padding_mask(batch, device)
            model.initialize_mask(padding_mask)

            loss, mse, pred, mask, latent, correlation = model_forward(
                model,
                batch,
                mask_ratio=config.video_mae_task_config.encoder_mask_ratio,
                alpha=config.video_mae_task_config.alpha,
            )
            # Sample out latents since there's too many
            num_elements_to_sample = int(latent.shape[1] * 0.25)
            permuted_indices = torch.randperm(latent.shape[1])
            latent = latent[:, permuted_indices[:num_elements_to_sample], :]
            if i == 0:
                all_latents = torch.zeros(
                    len(dataloader) * dataloader.batch_size,
                    num_elements_to_sample,
                    latent.shape[2],
                )
            all_latents[i * latent.shape[0] : (i + 1) * latent.shape[0], :, :] = latent
            # Get unpatchified original target and prediction
            _imgs = torch.index_select(
                batch,
                2,
                torch.linspace(0, batch.shape[2] - 1, model.pred_t_dim)
                .long()
                .to(device),
            )
            target = model.patchify(_imgs)

            # Filter to only masked patches
            B, L, C = target.shape
            expanded_mask = mask.repeat_interleave(C, dim=1).view(B, L, C).bool()

            masked_pred = pred.masked_fill(~expanded_mask, torch.nan)
            masked_target = target.masked_fill(~expanded_mask, torch.nan)
            all_preds.append(masked_pred.cpu())
            all_targets.append(masked_target.cpu())
            if i % 100 == 0:
                print("=" * 60)
                # print(all_embeddings[0].shape)
                # print("embeddings:", get_list_of_tensors_size_mb(all_embeddings))
                print("all_preds:", get_list_of_tensors_size_mb(all_preds))
                print("all_targets:", get_list_of_tensors_size_mb(all_targets))
                get_system_ram_usage()
                get_current_process_ram_usage()

    print("Stacking and Padding")
    return {
        "embeddings": all_latents,
        "preds": torch.cat(all_preds),
        "targets": torch.cat(all_targets),
    }


def main(checkpoint_dir, samples_dir, output_path, device="cuda"):
    # Paths
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    config_path = os.path.join(checkpoint_dir, "experiment_config.yml")

    # Load config
    config = create_video_mae_experiment_config_from_yaml(config_path)

    # Load test filepaths directly from the 'name' column
    test_csv_path = os.path.join(samples_dir, f"{config.job_name}_test_samples.csv")
    # test_csv_path = os.path.join(
    #     samples_dir, "model={}-grad_accum={}-encoder_mask_ratio={}_test_samples.csv"
    # )
    # test_csv_path = samples_dir
    test_df = pd.read_csv(test_csv_path)
    metadatas = []
    for meta_path in test_df["metadata"]:
        with open(meta_path, "r") as f:
            metadatas.append(json.load(f))
    filepaths_and_metadata = list(zip(test_df["name"].tolist(), metadatas))

    # Create test dataloader directly
    dataset = SequentialMultiFileECoGDataset(
        filepaths_and_metadata,
        config.ecog_data_config,
        # use_cache=True,
        load_on_init=False,
    )
    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.ecog_data_config.batch_size,
    )

    # Load model
    model = create_model(config)
    ckpt_mgr = CheckpointManager(model=model)
    ckpt_mgr.load(ckpt_path, strict=False)

    # Analyze
    output = collect_model_outputs(model, config, test_dl, device)

    # Save
    torch.save(output, output_path)
    print(f"✅ Saved analysis outputs to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Path to directory containing checkpoint and config",
    )
    parser.add_argument(
        "--samples_dir", required=True, help="Directory containing *_test_samples.csv"
    )
    parser.add_argument("--out", required=True, help="Path to output .pt file")
    parser.add_argument(
        "--device", default="cuda", help="Device to run on (default: cuda)"
    )
    args = parser.parse_args()

    main(args.checkpoint_dir, args.samples_dir, args.out, device=args.device)
