import torch
from src.models.audio_temporal import end_aligned_mask, zero_mean_unit_var


def test_end_aligned_mask_is_right_aligned():
    mask = end_aligned_mask(5, torch.tensor([3, 5]))
    assert mask.shape == (2, 5)
    assert mask[0].tolist() == [False, False, True, True, True]   # last 3 valid
    assert mask[1].tolist() == [True, True, True, True, True]
    # clamp: 0 valid -> at least the last position
    assert end_aligned_mask(5, torch.tensor([0]))[0].tolist() == [False, False, False, False, True]


def test_zero_mean_unit_var_over_valid_region():
    x = torch.tensor([[100.0, 100.0, 1.0, 2.0, 3.0]])           # first 2 are padding
    mask = torch.tensor([[False, False, True, True, True]])
    out = zero_mean_unit_var(x, mask)
    valid = out[0, 2:]
    assert torch.allclose(valid.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(valid.std(unbiased=False), torch.tensor(1.0), atol=1e-4)
    assert out[0, 0].item() == 0.0 and out[0, 1].item() == 0.0   # padding zeroed


def test_zero_mean_unit_var_invariant_to_padding_content():
    mask = torch.tensor([[False, False, True, True, True]])
    x1 = torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0]])
    x2 = torch.tensor([[9.0, -7.0, 1.0, 2.0, 3.0]])             # different padding values
    assert torch.allclose(zero_mean_unit_var(x1, mask), zero_mean_unit_var(x2, mask))
