import torch
from torch import Tensor


def gather_boundary_tail(
    hidden_2b: Tensor,
    batch_size: int,
    valid_samples: Tensor,
    tail_frames: int,
    audio_len_samples: int,
):
    """Gather the most-recent `tail_frames` content frames per sample, left-justified.

    Audio is front-padded to a fixed length, so content sits at the END of the
    `n_frames` axis. We gather the last `Lc = min(valid_frames, K)` frames in
    chronological order into a [B, K, 2D] buffer (pad at the END), so the
    boundary (most recent) frame lands at index `Lc-1` and packing/masking are clean.

    Returns (tail [B,K,2D], lengths [B] long, mask [B,K] bool).
    """
    B = batch_size
    n_frames = hidden_2b.shape[1]
    D = hidden_2b.shape[2]
    device = hidden_2b.device
    K = tail_frames

    vs = valid_samples.to(device=device, dtype=torch.float32)
    valid_frames = torch.round(vs / float(audio_len_samples) * n_frames).long()
    valid_frames = valid_frames.clamp(min=1, max=n_frames)
    lengths = valid_frames.clamp(max=K)  # [B]

    ar = torch.arange(K, device=device).unsqueeze(0)        # [1,K]
    mask = ar < lengths.unsqueeze(1)                         # [B,K] bool
    start = (n_frames - lengths).unsqueeze(1)               # [B,1] first gathered frame
    idx = (start + ar).clamp(max=n_frames - 1)              # [B,K]
    idx_e = idx.unsqueeze(-1).expand(-1, -1, D)            # [B,K,D]

    h0 = hidden_2b[:B]
    h1 = hidden_2b[B:]
    t0 = torch.gather(h0, 1, idx_e)
    t1 = torch.gather(h1, 1, idx_e)
    tail = torch.cat([t0, t1], dim=-1)                      # [B,K,2D]
    tail = tail * mask.unsqueeze(-1).to(tail.dtype)         # zero the trailing pad
    return tail, lengths, mask


import torch.nn as nn


class _MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale = dim ** -0.5

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        scores = (self.query * x).sum(dim=-1) * self.scale       # [B,K]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)    # [B,K,1]
        return (x * weights).sum(dim=1)                          # [B,D]


class TemporalHead(nn.Module):
    """Temporal aggregation over the boundary-aligned tail.

    head_type='gru'        -> BiGRU with packing (pad at end is ignored)
    head_type='transformer'-> TransformerEncoder with key_padding_mask
    Aggregation: [boundary frame] ++ [masked attention pool] -> Linear -> out_dim.
    """

    def __init__(self, in_dim, model_dim, out_dim,
                 head_type="gru", hidden_dim=256, num_layers=1, dropout=0.1):
        super().__init__()
        self.head_type = head_type
        self.model_dim = model_dim
        self.proj_in = nn.Linear(in_dim, model_dim)

        if head_type == "gru":
            self.rnn = nn.GRU(model_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0.0)
            feat_dim = 2 * hidden_dim
        elif head_type == "transformer":
            self.pos = nn.Parameter(torch.randn(1, 4096, model_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=8, dim_feedforward=4 * model_dim,
                dropout=dropout, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            feat_dim = model_dim
        else:
            raise ValueError(f"unknown head_type {head_type}")

        self.pool = _MaskedAttentionPool(feat_dim)
        self.out = nn.Linear(2 * feat_dim, out_dim)

    def forward(self, tail: Tensor, lengths: Tensor, mask: Tensor) -> Tensor:
        B, K, _ = tail.shape
        x = self.proj_in(tail)  # [B,K,model_dim]

        if self.head_type == "gru":
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, h_n = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True, total_length=K)   # [B,K,2H]
            boundary = torch.cat([h_n[-2], h_n[-1]], dim=-1)    # [B,2H]
        else:
            x = x + self.pos[:, :K, :]
            out = self.encoder(x, src_key_padding_mask=~mask)   # [B,K,model_dim]
            idx = (lengths - 1).clamp(min=0).view(B, 1, 1).expand(-1, 1, out.shape[-1])
            boundary = out.gather(1, idx).squeeze(1)            # [B,model_dim]

        pooled = self.pool(out, mask)                           # [B,feat_dim]
        return self.out(torch.cat([boundary, pooled], dim=-1))  # [B,out_dim]


def end_aligned_mask(seq_len: int, valid_lengths: Tensor, device=None) -> Tensor:
    """Boolean mask [B, seq_len] with the LAST `valid_lengths` positions True.

    Audio is front-padded (content at the end), so valid samples/frames are
    right-aligned. Used for the SSL backbone attention_mask and for normalization.
    """
    if device is None:
        device = valid_lengths.device
    ar = torch.arange(seq_len, device=device).unsqueeze(0)                      # [1, L]
    vl = valid_lengths.to(device=device).clamp(min=1, max=seq_len).unsqueeze(1)  # [B, 1]
    return ar >= (seq_len - vl)                                                 # [B, L] bool


def zero_mean_unit_var(x: Tensor, mask: Tensor, eps: float = 1e-7) -> Tensor:
    """Per-row zero-mean unit-variance normalization over masked (valid) positions.

    x: [B, L] float; mask: [B, L] (True = valid). Padding positions are excluded
    from the statistics and set to 0 in the output. Matches HF zero_mean_unit_var_norm
    (population variance over the valid region).
    """
    m = mask.to(dtype=x.dtype)
    cnt = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * m).sum(dim=1, keepdim=True) / cnt
    var = (((x - mean) * m) ** 2).sum(dim=1, keepdim=True) / cnt
    return (x - mean) / torch.sqrt(var + eps) * m
