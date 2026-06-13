import torch
from src.models.audio_temporal import gather_boundary_tail


def test_shapes_lengths_and_boundary_alignment():
    B, n_frames, D, K = 2, 30, 4, 10
    audio_len = 30  # 1 sample == 1 frame for easy arithmetic
    hidden = torch.randn(2 * B, n_frames, D)
    valid = torch.tensor([6, 20])  # -> valid_frames [6, 20] -> Lc [6, 10]

    tail, lengths, mask = gather_boundary_tail(hidden, B, valid, K, audio_len)

    assert tail.shape == (B, K, 2 * D)
    assert lengths.tolist() == [6, 10]
    assert mask.sum(dim=1).tolist() == [6, 10]
    # boundary (last valid, chronological) == most recent frame of each channel
    # sample 0: last valid index = lengths[0]-1 = 5
    assert torch.allclose(tail[0, 5, :D], hidden[0, n_frames - 1, :])      # ch0
    assert torch.allclose(tail[0, 5, D:], hidden[B + 0, n_frames - 1, :])  # ch1
    # pad positions are zeroed
    assert torch.count_nonzero(tail[0, 6:, :]) == 0


def test_invariant_to_leading_pad_region():
    B, n_frames, D, K = 2, 30, 4, 10
    audio_len = 30
    hidden = torch.randn(2 * B, n_frames, D)
    valid = torch.tensor([6, 6])           # Lc = 6 -> gathered region is frames [24, 30)
    tail1, _, _ = gather_boundary_tail(hidden, B, valid, K, audio_len)

    hidden2 = hidden.clone()
    hidden2[:, :24, :] = torch.randn_like(hidden2[:, :24, :])  # perturb ungathered leading region
    tail2, _, _ = gather_boundary_tail(hidden2, B, valid, K, audio_len)

    assert torch.allclose(tail1, tail2)
