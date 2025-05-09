from einops.layers.torch import Rearrange
import torch

from mask import get_padding_mask, get_tube_mask, get_decoder_mask


def test_get_padding_mask():
    # [batch, channels, frames, depth, height, width]
    fake_signal = torch.ones((4, 5, 8, 3, 3))
    # Set some parts of electrodes to nan implying padding
    fake_signal[:, :, :, 0, 0] = torch.nan
    fake_signal[:, :, :, 0, 1] = torch.nan
    fake_signal[:, 1, :, 1, 0] = torch.nan
    fake_signal[:, :, 3:, 1, 1] = torch.nan
    fake_signal[2, :, :, 1, 2] = torch.nan
    
    actual_padding_mask = get_padding_mask(fake_signal, "cpu")
    assert torch.all(actual_padding_mask == torch.tensor([[False, False, True],
                                                          [False, False, False],
                                                          [True, True, True]]))
