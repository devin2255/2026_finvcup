import torch
from src.data.dataset import front_pad_or_trim


def test_front_pads_short_wave():
    wave = torch.arange(1, 11, dtype=torch.float32).reshape(2, 5)  # [2,5]
    out = front_pad_or_trim(wave, target=8)
    assert out.shape == (2, 8)
    assert torch.count_nonzero(out[:, :3]) == 0     # zeros at FRONT
    assert torch.allclose(out[:, 3:], wave)         # content at END


def test_trims_long_wave_keeping_most_recent():
    wave = torch.arange(1, 21, dtype=torch.float32).reshape(2, 10)  # [2,10]
    out = front_pad_or_trim(wave, target=8)
    assert out.shape == (2, 8)
    assert torch.allclose(out, wave[:, -8:])        # keeps the last 8 (most recent)
