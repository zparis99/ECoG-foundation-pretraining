import os
import torch
import numpy as np
import random
import scipy
from einops import rearrange
from typing import Optional

from ecog_foundation_model.config import ViTConfig
from ecog_foundation_model import constants


def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patchify(imgs, patch_size, frame_patch_size):
    N, C, T, H, W = imgs.shape
    ph, pw = patch_size
    assert H % ph == 0 and W % pw == 0 and T % frame_patch_size == 0
    h = H // ph
    w = W // pw
    t = T // frame_patch_size

    x = imgs.reshape(shape=(N, C, t, frame_patch_size, h, ph, w, pw))
    x = torch.einsum("nctuhpwq->nthwupqc", x)
    x = x.reshape(shape=(N, t * h * w, frame_patch_size * ph * pw * C))
    return x


def unpatchify(x, patch_size, frame_patch_size, grid_size):
    N, L, D = x.shape

    ph, pw = patch_size
    h, w = grid_size
    t = L // h // w
    C = D // frame_patch_size // ph // pw

    x = x.reshape(shape=(N, t, h, w, frame_patch_size, ph, pw, C))

    x = torch.einsum("nthwupqc->nctuhpwq", x)
    T, H, W = t * frame_patch_size, h * ph, w * pw
    imgs = x.reshape(shape=(N, C, T, H, W))
    return imgs


# TODO: Add tests, and review if we need this or any of these other functions.
def apply_mask_to_batch(model, batch, mask, frame_patch_size):
    """Replaces values in batch with nan where mask is True. Also returns mask in same shape as batch.
    model: MaskedAutencoderViT
    mask: (batch_size, frames // frame_patch_size)
    batch: (batch_size, num_channels, frames, electrode_height, electrode_width)
    frame_patch_size: The number of
    """
    patch_size = model.patch_embed.patch_size
    grid_size = model.patch_embed.grid_size
    batch_patch = patchify(batch, patch_size, frame_patch_size)
    # Mask is for every frame_patch_size frames, so expand to align with batch
    # tensor and fill in masked out information with nan.
    _, _, C = batch_patch.shape
    B, T = mask.shape
    # Repeats mask values to align with batch dimensions.
    mask = mask.repeat_interleave(C, axis=1).view(B, T, C).to(torch.bool)
    masked_batch = unpatchify(
        batch_patch.masked_fill(mask, torch.nan),
        patch_size,
        frame_patch_size,
        grid_size,
    )
    mask = unpatchify(mask, patch_size, frame_patch_size, grid_size)

    return masked_batch, mask


def get_signal_stats(signal: np.array) -> tuple[np.array, np.array]:
    """Generate means and standard deviations for all electrodes in signal.

    Args:
        signal (np.array): Shape [num_electrodes, num_samples].

    Returns:
        tuple[np.array, np.array]: (means, stds) each array has shape [num_electrodes].
    """
    return np.mean(signal, axis=1), np.std(signal, axis=1)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("param counts:\n{:,} total\n{:,} trainable".format(total, trainable))
    return trainable


def get_signal(
    patches: torch.Tensor,
    batch_size: int,
    num_bands: int,
    num_frames: int,
    model_config: ViTConfig,
) -> torch.Tensor:
    """Convert patches into a signal of shape [electrodes, num_bands, num_frames]

    Args:
        patches (torch.Tensor): Patch transformed signal of model.
        batch_size (int): The number of examples in a batch.
        num_bands (int): Number of bands in patches.
        num_frames (int): Number of frames in patches.
        model_config (ViTConfig): Config for model.
    """
    return rearrange(
        patches,
        "b (f d s) (pd ph pw pf c) -> b (d pd s ph pw) c (f pf)",
        c=num_bands,
        d=1,
        f=num_frames // model_config.frame_patch_size,
        pd=model_config.patch_dims[0],
        ph=model_config.patch_dims[1],
        pw=model_config.patch_dims[2],
        pf=model_config.frame_patch_size,
    )


def rearrange_signals(
    model_config: ViTConfig,
    model,
    device,
    signal,
    decoder_out,
    padding_mask,
    tube_mask,
    decoder_mask,
    decoder_padding_mask,
):
    batch_size = signal.shape[0]
    num_bands = signal.shape[1]
    num_frames = signal.shape[2]

    # parts of the reconstructed signal that were not seen by the encoder
    unseen_output = decoder_out[:, len(tube_mask.nonzero()) :]

    # parts of the reconstructed signal that were seen by the encoder
    seen_output = decoder_out[:, : len(tube_mask.nonzero())]

    # rearrange original signal into patches
    target_patches = model.patchify(signal)
    target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")

    # parts of the original signal not seen by the encoder
    unseen_target = target_patches_vit[:, decoder_mask][:, decoder_padding_mask]

    # parts of the original signal seen by the encoder
    seen_target = target_patches_vit[:, ~decoder_mask]

    # rearranging seen and unseen parts of the reconstructed signal into original position
    full_recon_patches = (
        torch.zeros(target_patches_vit.shape).fill_(float("nan")).to(device)
    )

    tube_idx = torch.nonzero(tube_mask).squeeze()
    decoder_idx = torch.nonzero(decoder_mask & padding_mask).squeeze()

    full_recon_patches[:, tube_idx, :] = seen_output
    full_recon_patches[:, decoder_idx, :] = unseen_output

    full_recon_signal = get_signal(
        full_recon_patches, batch_size, num_bands, num_frames, model_config
    )

    full_target_signal = rearrange(signal, "b c f d h w -> b (h w d) c f")

    # rearrange unseen patches into signal
    unseen_recon_signal = get_signal(
        unseen_output, batch_size, num_bands, num_frames, model_config
    )

    unseen_target_signal = get_signal(
        unseen_target, batch_size, num_bands, num_frames, model_config
    )

    return (
        full_recon_signal,
        full_target_signal,
        unseen_output,
        unseen_target,
        seen_output,
        seen_target,
        unseen_target_signal,
        unseen_recon_signal,
    )


def contrastive_loss(
    cls_token1: torch.Tensor, cls_token2: torch.Tensor, temperature: torch.Tensor
):
    feat1 = cls_token1 / cls_token1.norm(dim=1, keepdim=True)
    feat2 = cls_token2 / cls_token2.norm(dim=1, keepdim=True)

    cosine_sim = feat1 @ feat2.T
    logit_scale = temperature.exp()  # log scale, learned during training
    feat1 = cosine_sim * logit_scale
    feat2 = feat1.T

    labels = torch.arange(feat1.shape[0]).to(feat1.device)
    loss = (
        torch.nn.functional.cross_entropy(feat1, labels)
        + torch.nn.functional.cross_entropy(feat2, labels)
    ) / 2
    return loss


def overwrite_model_padding_mask(model, padding_mask, using_padding_mask):
    has_padded_pixels = not padding_mask.all()
    if has_padded_pixels:
        model.initialize_mask(padding_mask)
        return True
    # Stop initializing mask when there is no more need for padding.
    elif using_padding_mask:
        model.initialize_mask(None)
        return False


def get_signal_correlations(signal_a, signal_b):
    """Get correlation coefficients between channels of signal_a and signal_b averaged over batch.

    Args:
        signal_a (tensor): shape [batch, num_electrodes, channels, observations]
        signal_b (tensor): shape [batch, num_electrodes, channels, observations]

    Returns:
        tensor of shape [num_electrodes, channels] where each entry is the correlation found between
        the two signals for that electrode and channel.
    """
    correlation_matrix = torch.zeros(
        signal_a.shape[0], signal_a.shape[1], signal_a.shape[2]
    ).fill_(torch.nan)

    for batch in range(signal_a.shape[0]):
        for electrode in range(signal_a.shape[1]):
            for channel in range(signal_a.shape[2]):
                correlation_matrix[batch, electrode, channel] = torch.corrcoef(
                    torch.stack(
                        [
                            signal_a[batch, electrode, channel],
                            signal_b[batch, electrode, channel],
                        ]
                    )
                )[0, 1]

    return correlation_matrix.mean(dim=0)
