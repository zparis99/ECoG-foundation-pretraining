import numpy as np
import pytest
import torch

from ecog_foundation_model import constants
from ecog_foundation_model.config import ECoGDataConfig
from ecog_foundation_model.mae_st_util.models_mae import MaskedAutoencoderViT
from pretrain_engine import model_forward
from loader import BufferedFileRandomSampler, MultiFileECoGDataset

EMBEDDING_DIM = 64
FRAMES_PER_SAMPLE = 64
NUM_BANDS = 5
FRAME_PATCH_SIZE = 4


def create_dataloader(filepaths, data_config, use_cache=False):
    dataset = MultiFileECoGDataset(filepaths, data_config, use_cache=use_cache)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        sampler=BufferedFileRandomSampler(dataset),
    )
    return dataloader


@pytest.fixture
def model():
    return MaskedAutoencoderViT(
        img_size=constants.GRID_SIZE,
        patch_size=1,
        in_chans=NUM_BANDS,
        norm_pix_loss=False,
        num_frames=FRAMES_PER_SAMPLE,
        t_patch_size=FRAME_PATCH_SIZE,
        cls_embed=False,
        pred_t_dim=FRAMES_PER_SAMPLE // FRAME_PATCH_SIZE,
        embed_dim=EMBEDDING_DIM,
        depth=1,
        num_heads=1,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=1,
        mlp_ratio=2.0,
    )


def test_model_with_data_loader_input_succeeds(model, create_fake_mne_file_fn):
    data_config = ECoGDataConfig(
        batch_size=16,
        bands=[[i + 1, i + 2] for i in range(NUM_BANDS)],
        new_fs=FRAMES_PER_SAMPLE // 2,
        sample_length=2,
        max_open_files=2,
    )
    # Two batches of data per file. Create 5 files.
    ch_names = ["G" + str(i + 1) for i in range(64 + 1)]
    fake_data = np.ones((65, int(32 * 2 * 512)))
    filepaths = []
    for i in range(5):
        filepaths.append(
            create_fake_mne_file_fn(ch_names, fake_data, 512, str(i) + "_raw.fif")
        )
    dataloader = create_dataloader(filepaths, data_config, use_cache=False)

    for data in dataloader:
        loss, mse, pred, mask, latent, correlations = model_forward(
            model, data, mask_ratio=0.8, alpha=0.25
        )

        num_patches = (
            FRAMES_PER_SAMPLE
            * constants.GRID_SIZE
            * constants.GRID_SIZE
            // FRAME_PATCH_SIZE
        )
        assert loss.detach().numpy().shape == ()
        assert mse.detach().numpy().shape == ()
        assert not torch.isnan(loss)
        assert pred.detach().numpy().shape == (16, num_patches, NUM_BANDS)
        assert mask.detach().numpy().shape == (16, num_patches)
        assert latent.detach().numpy().shape == (
            16,
            int(num_patches * (1 - 0.8)),
            EMBEDDING_DIM,
        )
        assert correlations.detach().numpy().shape == ()

        # Check that loss is set as expected
        assert torch.isclose((1 - correlations) / 2 * 0.25 + mse * 0.75, loss)
