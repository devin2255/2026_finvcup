# Audio Branch Dual-Channel + Temporal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mono-averaged, single-vector Whisper audio branch with a per-channel, boundary-aligned, temporal-sequence encoder, fixing the dynamic-context padding bug — without changing the fusion, the 5-label head, or the submission format.

**Architecture:** In the Whisper audio branch, encode the two speaker channels separately (shared weights, one `2B` forward), gather a boundary-aligned **left-justified** tail of the most-recent frames per sample (pad at end → clean masking/packing), run a small temporal head (BiGRU default, Transformer optional), and pool to the same `[B, proj_dim]` vector the existing `MultimodalFusion` already consumes. The fixed-length front-padding of the waveform guarantees the prediction boundary is the last frame.

**Tech Stack:** PyTorch, torchaudio, HuggingFace transformers (Whisper), pytest.

**Design spec:** [docs/superpowers/specs/2026-06-13-audio-branch-dualchannel-temporal-design.md](../specs/2026-06-13-audio-branch-dualchannel-temporal-design.md)

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/models/audio_temporal.py` | Pure helpers (`gather_boundary_tail`) + `TemporalHead` module. No Whisper dependency → unit-testable in isolation. | Create |
| `src/models/multimodal_baseline.py` | `WhisperAudioEncoder` dual-channel path; `MultimodalTurnTakingModel.forward` threads `audio_valid_samples`. | Modify |
| `src/data/dataset.py` | `front_pad_or_trim` helper; datasets emit `audio_valid_samples`; collate stacks it. | Modify |
| `src/train.py` | Two `model(...)` call sites pass `audio_valid_samples`. | Modify |
| `src/infer_test.py` | One `model(...)` call site passes `audio_valid_samples`. | Modify |
| `configs/whisper_qwen0_6b_dualchannel_temporal.yaml` | New training config with the `audio_encoder` block + bumped batch size. | Create |
| `tests/` | pytest unit tests. | Create |
| `requirements.txt` | Add `pytest`. | Modify |

**Key shared signatures (must match across tasks):**

```python
# src/models/audio_temporal.py
def gather_boundary_tail(
    hidden_2b: Tensor,        # [2B, n_frames, D]
    batch_size: int,          # B
    valid_samples: Tensor,    # [B] long
    tail_frames: int,         # K
    audio_len_samples: int,   # fixed front-pad target, e.g. 480000
) -> tuple[Tensor, Tensor, Tensor]:   # tail [B,K,2D], lengths [B] long, mask [B,K] bool

class TemporalHead(nn.Module):
    def __init__(self, in_dim, model_dim, out_dim,
                 head_type="gru", hidden_dim=256, num_layers=1, dropout=0.1): ...
    def forward(self, tail, lengths, mask) -> Tensor:  # [B, out_dim]

# src/data/dataset.py
def front_pad_or_trim(wave: Tensor, target: int) -> Tensor:   # [C,T] -> [C,target]

# src/models/multimodal_baseline.py
class WhisperAudioEncoder(nn.Module):
    def __init__(self, model_name, sample_rate, proj_dim, freeze=True,
                 tail_ratio=0.2, unfreeze_layers=0,
                 dual_channel=False, tail_frames=400,
                 audio_len_samples=480000, temporal_head_cfg=None): ...
    def forward(self, wave, valid_samples=None) -> Tensor:  # [B, proj_dim]
```

---

## Task 1: Test infrastructure

**Files:**
- Modify: `requirements.txt`
- Create: `tests/test_infra_smoke.py`

- [ ] **Step 1: Add pytest to requirements**

Append one line to `requirements.txt`:

```
pytest
```

- [ ] **Step 2: Install it**

Run: `pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 3: Write a trivial test**

Create `tests/test_infra_smoke.py`:

```python
import torch


def test_torch_imports_and_runs():
    x = torch.zeros(2, 3)
    assert x.shape == (2, 3)
```

- [ ] **Step 4: Run it**

Run: `python -m pytest tests/test_infra_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/test_infra_smoke.py
git commit -m "test: add pytest infra and smoke test"
```

---

## Task 2: `gather_boundary_tail` helper

**Files:**
- Create: `src/models/audio_temporal.py`
- Test: `tests/test_gather_boundary_tail.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gather_boundary_tail.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gather_boundary_tail.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.audio_temporal'`

- [ ] **Step 3: Write minimal implementation**

Create `src/models/audio_temporal.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gather_boundary_tail.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/audio_temporal.py tests/test_gather_boundary_tail.py
git commit -m "feat: add gather_boundary_tail for boundary-aligned audio tail"
```

---

## Task 3: `TemporalHead` module (GRU + Transformer)

**Files:**
- Modify: `src/models/audio_temporal.py`
- Test: `tests/test_temporal_head.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_temporal_head.py`:

```python
import torch
from src.models.audio_temporal import TemporalHead


def _inputs(B=2, K=10, in_dim=8):
    tail = torch.randn(B, K, in_dim)
    lengths = torch.tensor([K, 6])
    ar = torch.arange(K).unsqueeze(0)
    mask = ar < lengths.unsqueeze(1)
    tail = tail * mask.unsqueeze(-1)  # pad positions zeroed, mirroring gather output
    return tail, lengths, mask


def test_gru_output_shape():
    head = TemporalHead(in_dim=8, model_dim=16, out_dim=5,
                        head_type="gru", hidden_dim=8, num_layers=1, dropout=0.0)
    head.eval()
    tail, lengths, mask = _inputs()
    out = head(tail, lengths, mask)
    assert out.shape == (2, 5)


def test_transformer_output_shape():
    head = TemporalHead(in_dim=8, model_dim=16, out_dim=5,
                        head_type="transformer", num_layers=2, dropout=0.0)
    head.eval()
    tail, lengths, mask = _inputs()
    out = head(tail, lengths, mask)
    assert out.shape == (2, 5)


def test_gru_invariant_to_padded_positions():
    head = TemporalHead(in_dim=8, model_dim=16, out_dim=5,
                        head_type="gru", hidden_dim=8, num_layers=1, dropout=0.0)
    head.eval()
    tail, lengths, mask = _inputs()
    out1 = head(tail, lengths, mask)
    tail2 = tail.clone()
    tail2[~mask] = torch.randn_like(tail2[~mask])  # garbage in padded slots
    out2 = head(tail2, lengths, mask)
    assert torch.allclose(out1, out2, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_temporal_head.py -v`
Expected: FAIL with `ImportError: cannot import name 'TemporalHead'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/models/audio_temporal.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_temporal_head.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/audio_temporal.py tests/test_temporal_head.py
git commit -m "feat: add TemporalHead (BiGRU + Transformer) with masked aggregation"
```

---

## Task 4: `front_pad_or_trim` waveform helper

**Files:**
- Modify: `src/data/dataset.py`
- Test: `tests/test_front_pad.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_front_pad.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_front_pad.py -v`
Expected: FAIL with `ImportError: cannot import name 'front_pad_or_trim'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/data/dataset.py` (module level, near `_read_wav_slice`):

```python
def front_pad_or_trim(wave: torch.Tensor, target: int) -> torch.Tensor:
    """Make wave [C, target] by zero-padding at the FRONT or keeping the last `target`.

    Front-padding guarantees the most-recent sample (prediction boundary) is the
    last column, so downstream tail-slicing is boundary-aligned.
    """
    content = wave.shape[1]
    if content < target:
        return torch.nn.functional.pad(wave, (target - content, 0))
    return wave[:, -target:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_front_pad.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset.py tests/test_front_pad.py
git commit -m "feat: add front_pad_or_trim for boundary-aligned waveforms"
```

---

## Task 5: Datasets emit `audio_valid_samples`; collate stacks it

**Files:**
- Modify: `src/data/dataset.py` (`_load_wave_segment`, both `__getitem__`, `CollateFn.__call__`)
- Test: `tests/test_collate_valid_samples.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_collate_valid_samples.py`:

```python
import torch
from src.data.dataset import CollateFn


class _StubTokenizer:
    truncation_side = "right"
    padding_side = "right"

    def __call__(self, texts, max_length, truncation, padding, return_tensors):
        n = len(texts)
        return {"input_ids": torch.ones(n, 3, dtype=torch.long),
                "attention_mask": torch.ones(n, 3, dtype=torch.long)}


def _item(valid, T):
    return {
        "conv_id": "c",
        "end_idx": 0,
        "waveform": torch.zeros(2, T),
        "text": "hi",
        "context_labels": torch.zeros(375, dtype=torch.long),
        "audio_valid_samples": torch.tensor(valid, dtype=torch.long),
        "label": torch.zeros(5),
    }


def test_collate_includes_audio_valid_samples():
    collate = CollateFn(_StubTokenizer(), text_max_length=8)
    batch = [_item(480000, 480000), _item(160000, 480000)]
    out = collate(batch)
    assert "audio_valid_samples" in out
    assert out["audio_valid_samples"].shape == (2,)
    assert out["audio_valid_samples"].tolist() == [480000, 160000]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collate_valid_samples.py -v`
Expected: FAIL with `KeyError: 'audio_valid_samples'` (or assertion error on missing key)

- [ ] **Step 3: Implement the three edits in `src/data/dataset.py`**

(a) In `_load_wave_segment`, replace the trailing pad/trim block (currently the `expected_frames` logic at the end of the method) with a fixed front-pad to the full context length, and remember the true content length. Replace:

```python
        expected_frames = int((end_ms - start_ms) * self.sample_rate / 1000)
        if wave.shape[1] < expected_frames:
            pad = expected_frames - wave.shape[1]
            wave = torch.nn.functional.pad(wave, (0, pad))
        elif wave.shape[1] > expected_frames:
            wave = wave[:, :expected_frames]
        return wave
```

with:

```python
        target = self.context_chunks * self.chunk_ms * self.sample_rate // 1000
        valid_samples = min(wave.shape[1], target)
        wave = front_pad_or_trim(wave, target)
        return wave, valid_samples
```

(b) In `TurnTakingTrainDataset.__getitem__`, update the wave call + output dict. Replace:

```python
        wave = self._load_wave_segment(sample.conv_id, start_ms, end_ms)
        wave = self._augment_audio(wave)

        out = {
            "conv_id": sample.conv_id,
            "end_idx": end_idx,
            "waveform": wave,
            "text": text,
            "context_labels": torch.from_numpy(context_labels),
        }
```

with:

```python
        wave, valid_samples = self._load_wave_segment(sample.conv_id, start_ms, end_ms)
        wave = self._augment_audio(wave)

        out = {
            "conv_id": sample.conv_id,
            "end_idx": end_idx,
            "waveform": wave,
            "text": text,
            "context_labels": torch.from_numpy(context_labels),
            "audio_valid_samples": torch.tensor(valid_samples, dtype=torch.long),
        }
```

(c) In `TurnTakingTestDataset.__getitem__`, the test set builds the wave inline (it does not call `_load_wave_segment`). Replace:

```python
        wav_path = self.audio_dir / f"{seg_id}.wav"
        audio, src_sr = _read_wav_slice(wav_path, start_ms, end_ms)
        wave = torch.from_numpy(audio.T)
        if wave.shape[0] == 1:
            wave = wave.repeat(2, 1)
        elif wave.shape[0] > 2:
            wave = wave[:2]
        if src_sr != self.sample_rate:
            wave = torchaudio.functional.resample(wave, src_sr, self.sample_rate)

        return {
            "segment_id": seg_id,
            "waveform": wave,
            "text": text,
            "context_labels": torch.from_numpy(context_labels),
        }
```

with:

```python
        wav_path = self.audio_dir / f"{seg_id}.wav"
        audio, src_sr = _read_wav_slice(wav_path, start_ms, end_ms)
        wave = torch.from_numpy(audio.T)
        if wave.shape[0] == 1:
            wave = wave.repeat(2, 1)
        elif wave.shape[0] > 2:
            wave = wave[:2]
        if src_sr != self.sample_rate:
            wave = torchaudio.functional.resample(wave, src_sr, self.sample_rate)

        target = 375 * 80 * self.sample_rate // 1000  # context_chunks(375) * chunk_ms(80) -> 30s
        valid_samples = min(wave.shape[1], target)
        wave = front_pad_or_trim(wave, target)

        return {
            "segment_id": seg_id,
            "waveform": wave,
            "text": text,
            "context_labels": torch.from_numpy(context_labels),
            "audio_valid_samples": torch.tensor(valid_samples, dtype=torch.long),
        }
```

> Note: `TurnTakingTestDataset` has no `context_chunks`/`chunk_ms` attributes today, so the `375 * 80` constants are inlined here; they match the competition config (`context_chunks: 375`, `chunk_ms: 80`). If you later run a config with different chunking, thread those values into the test dataset constructor instead of hardcoding.

(d) In `CollateFn.__call__`, add `audio_valid_samples` to BOTH output branches. After the existing `out = {...}` dict is built (the block starting `out = { ... "context_labels": ... }`), insert before the `if "label" in batch[0]:` check:

```python
        if "audio_valid_samples" in batch[0]:
            out["audio_valid_samples"] = torch.stack(
                [b["audio_valid_samples"] for b in batch], dim=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_collate_valid_samples.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full suite so far**

Run: `python -m pytest tests/ -v`
Expected: PASS (all green)

- [ ] **Step 6: Commit**

```bash
git add src/data/dataset.py tests/test_collate_valid_samples.py
git commit -m "feat: datasets emit audio_valid_samples and front-pad waveforms"
```

---

## Task 6: Wire `WhisperAudioEncoder` dual-channel path + model forward

**Files:**
- Modify: `src/models/multimodal_baseline.py` (`WhisperAudioEncoder`, `MultimodalTurnTakingModel`)
- Test: `tests/test_whisper_encoder_dual.py`

- [ ] **Step 1: Write the failing test (stubbed Whisper, no download)**

Create `tests/test_whisper_encoder_dual.py`:

```python
import torch
import torch.nn as nn
from src.models import multimodal_baseline as mb


class _StubFE:
    def __call__(self, arrays, sampling_rate, return_tensors):
        n = len(arrays)
        return {"input_features": torch.zeros(n, 128, 3000)}


class _StubEnc(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()
        self.config = type("C", (), {"d_model": d_model})()
        self.layers = nn.ModuleList([nn.Linear(1, 1)])

    def forward(self, input_features=None):
        b = input_features.shape[0]
        hs = torch.randn(b, 1500, self.config.d_model)
        return type("O", (), {"last_hidden_state": hs})()


class _StubWhisperModel:
    @staticmethod
    def from_pretrained(*a, **k):
        m = nn.Module()
        m.encoder = _StubEnc()
        return m


def _make_encoder(monkeypatch, proj_dim=8):
    monkeypatch.setattr(mb.WhisperFeatureExtractor, "from_pretrained",
                        staticmethod(lambda *a, **k: _StubFE()))
    monkeypatch.setattr(mb, "WhisperModel", _StubWhisperModel)
    return mb.WhisperAudioEncoder(
        model_name="stub", sample_rate=16000, proj_dim=proj_dim,
        dual_channel=True, tail_frames=20, audio_len_samples=480000,
        temporal_head_cfg={"type": "gru", "hidden_dim": 8, "num_layers": 1, "dropout": 0.0},
    )


def test_dual_forward_shape(monkeypatch):
    enc = _make_encoder(monkeypatch, proj_dim=8)
    enc.eval()
    wave = torch.randn(3, 2, 480000)
    vs = torch.tensor([480000, 240000, 48000])
    out = enc(wave, valid_samples=vs)
    assert out.shape == (3, 8)
    assert enc.out_dim == 8


def test_forward_runs_without_valid_samples(monkeypatch):
    enc = _make_encoder(monkeypatch, proj_dim=8)
    enc.eval()
    wave = torch.randn(2, 2, 480000)
    out = enc(wave)  # valid_samples=None -> treat as fully valid
    assert out.shape == (2, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_whisper_encoder_dual.py -v`
Expected: FAIL — `WhisperAudioEncoder.__init__` does not accept `dual_channel`.

- [ ] **Step 3: Implement the dual path in `WhisperAudioEncoder`**

Add the import near the top of `src/models/multimodal_baseline.py`:

```python
from src.models.audio_temporal import gather_boundary_tail, TemporalHead
```

Replace the entire `WhisperAudioEncoder.__init__` signature and body, and its `_build_input_features` / `forward`, with:

```python
class WhisperAudioEncoder(nn.Module):
    def __init__(
        self, model_name: str, sample_rate: int, proj_dim: int,
        freeze: bool = True, tail_ratio: float = 0.2,
        unfreeze_layers: int = 0,
        dual_channel: bool = False, tail_frames: int = 400,
        audio_len_samples: int = 480000, temporal_head_cfg: dict | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.freeze = freeze
        self.tail_ratio = tail_ratio
        self.dual_channel = dual_channel
        self.tail_frames = tail_frames
        self.audio_len_samples = audio_len_samples
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.encoder = WhisperModel.from_pretrained(model_name).encoder
        if self.freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if unfreeze_layers > 0 and self.freeze:
            total_layers = len(self.encoder.layers)
            for layer_idx in range(max(0, total_layers - unfreeze_layers), total_layers):
                for p in self.encoder.layers[layer_idx].parameters():
                    p.requires_grad = True
        self.encoder_has_trainable_layers = any(p.requires_grad for p in self.encoder.parameters())
        hidden_size = int(self.encoder.config.d_model)

        if self.dual_channel:
            cfg = temporal_head_cfg or {}
            self.temporal_head = TemporalHead(
                in_dim=2 * hidden_size, model_dim=proj_dim, out_dim=proj_dim,
                head_type=str(cfg.get("type", "gru")),
                hidden_dim=int(cfg.get("hidden_dim", 256)),
                num_layers=int(cfg.get("num_layers", 1)),
                dropout=float(cfg.get("dropout", 0.1)),
            )
        else:
            self.attn_pool = AttentionPooling(hidden_size)
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
            )
        self.out_dim = proj_dim

    def _features_from_channel(self, wave_1ch: torch.Tensor) -> torch.Tensor:
        arr = wave_1ch.detach().float().cpu().numpy()
        return [x for x in arr]

    def _run_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        if self.freeze and not self.encoder_has_trainable_layers:
            with torch.no_grad():
                return self.encoder(input_features=input_features).last_hidden_state
        return self.encoder(input_features=input_features).last_hidden_state

    def forward(self, wave: torch.Tensor, valid_samples: torch.Tensor | None = None) -> torch.Tensor:
        B = wave.shape[0]
        if not self.dual_channel:
            mono = wave.mean(dim=1)
            with torch.amp.autocast("cuda", enabled=False):
                arrays = [x for x in mono.detach().float().cpu().numpy()]
                feats = self.feature_extractor(arrays, sampling_rate=self.sample_rate,
                                               return_tensors="pt")["input_features"]
            hidden = self._run_encoder(feats.to(wave.device))
            T = hidden.shape[1]
            tail_start = max(0, T - int(T * self.tail_ratio))
            pooled = self.attn_pool(hidden[:, tail_start:, :])
            return self.proj(pooled)

        # dual-channel path
        with torch.amp.autocast("cuda", enabled=False):
            arrays = self._features_from_channel(wave[:, 0, :]) + \
                     self._features_from_channel(wave[:, 1, :])
            feats = self.feature_extractor(arrays, sampling_rate=self.sample_rate,
                                           return_tensors="pt")["input_features"]  # [2B,128,3000]
        hidden = self._run_encoder(feats.to(wave.device))                          # [2B,T,D]

        if valid_samples is None:
            valid_samples = torch.full((B,), self.audio_len_samples,
                                       dtype=torch.long, device=wave.device)
        tail, lengths, mask = gather_boundary_tail(
            hidden, B, valid_samples, self.tail_frames, self.audio_len_samples)
        return self.temporal_head(tail, lengths, mask)
```

Then update `MultimodalTurnTakingModel.__init__` where it constructs the whisper encoder. Replace:

```python
            self.audio_encoder = WhisperAudioEncoder(
                model_name=cfg["audio_encoder"]["model_name"],
                sample_rate=cfg["sample_rate"],
                proj_dim=int(cfg["audio_encoder"]["proj_dim"]),
                freeze=bool(cfg["audio_encoder"].get("freeze", True)),
                tail_ratio=float(cfg["audio_encoder"].get("tail_ratio", 0.2)),
                unfreeze_layers=int(cfg["audio_encoder"].get("unfreeze_layers", 0)),
            )
```

with:

```python
            audio_len_samples = int(cfg["context_chunks"]) * int(cfg["chunk_ms"]) \
                * int(cfg["sample_rate"]) // 1000
            self.audio_encoder = WhisperAudioEncoder(
                model_name=cfg["audio_encoder"]["model_name"],
                sample_rate=cfg["sample_rate"],
                proj_dim=int(cfg["audio_encoder"]["proj_dim"]),
                freeze=bool(cfg["audio_encoder"].get("freeze", True)),
                tail_ratio=float(cfg["audio_encoder"].get("tail_ratio", 0.2)),
                unfreeze_layers=int(cfg["audio_encoder"].get("unfreeze_layers", 0)),
                dual_channel=bool(cfg["audio_encoder"].get("dual_channel", False)),
                tail_frames=int(cfg["audio_encoder"].get("tail_frames", 400)),
                audio_len_samples=audio_len_samples,
                temporal_head_cfg=cfg["audio_encoder"].get("temporal_head"),
            )
```

Finally, update `MultimodalTurnTakingModel.forward` to accept and forward `audio_valid_samples`. Replace:

```python
    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_labels: torch.Tensor,
    ) -> torch.Tensor:
        audio_feat = self.audio_encoder(waveform)
```

with:

```python
    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_labels: torch.Tensor,
        audio_valid_samples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(self.audio_encoder, WhisperAudioEncoder):
            audio_feat = self.audio_encoder(waveform, valid_samples=audio_valid_samples)
        else:
            audio_feat = self.audio_encoder(waveform)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_whisper_encoder_dual.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: PASS (all green)

- [ ] **Step 6: Commit**

```bash
git add src/models/multimodal_baseline.py tests/test_whisper_encoder_dual.py
git commit -m "feat: dual-channel Whisper encoder with temporal head + model wiring"
```

---

## Task 7: Thread `audio_valid_samples` through train/infer call sites

**Files:**
- Modify: `src/train.py` (two `model(...)` sites: eval ~L62-74, train ~L392-404)
- Modify: `src/infer_test.py` (one `model(...)` site ~L102-114)

- [ ] **Step 1: Edit the eval loop in `src/train.py`**

In `evaluate(...)`, after the `context_labels = ...` line and before the `with torch.amp.autocast(...)` block, add:

```python
        audio_valid_samples = batch["audio_valid_samples"].to(device, non_blocking=True)
```

and change the `model(` call to include the new kwarg:

```python
            logits = model(
                waveform=waveform,
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_labels=context_labels,
                audio_valid_samples=audio_valid_samples,
            )
```

- [ ] **Step 2: Edit the training loop in `src/train.py`**

In the training step (the block around L392), after `context_labels = ...` add the same `audio_valid_samples = batch["audio_valid_samples"].to(...)` line, and add `audio_valid_samples=audio_valid_samples,` to that `model(` call (the one at ~L399).

- [ ] **Step 3: Edit the inference loop in `src/infer_test.py`**

After `context_labels = ...` (L105) add:

```python
            audio_valid_samples = batch["audio_valid_samples"].to(device, non_blocking=True)
```

and add `audio_valid_samples=audio_valid_samples,` to the `model(` call at ~L109.

- [ ] **Step 4: Sanity-check imports compile**

Run: `python -c "import ast; ast.parse(open('src/train.py').read()); ast.parse(open('src/infer_test.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add src/train.py src/infer_test.py
git commit -m "feat: pass audio_valid_samples to model in train and infer loops"
```

---

## Task 8: New training config + end-to-end smoke run

**Files:**
- Create: `configs/whisper_qwen0_6b_dualchannel_temporal.yaml`

- [ ] **Step 1: Create the config**

Create `configs/whisper_qwen0_6b_dualchannel_temporal.yaml` (copy of the optimized config with the new `audio_encoder` block and a bumped batch size — adjust the `paths.*` roots to your machine's real data paths before running):

```yaml
seed: 42
chunk_ms: 80
context_chunks: 375
target_chunks: 25
stride: 2
sample_rate: 16000
max_train_samples: null
max_valid_samples: null
num_workers: 2

data_augmentation:
  dynamic_context: true
  min_context_chunks: 125
  max_context_chunks: 375
  context_prob: 0.5

paths:
  project_root: D:/bisai/V2/2026_finvcup_baseline
  train_audio_dir: D:/bisai/V2/2026_finvcup_baseline/train/audio
  train_text_dir: D:/bisai/V2/2026_finvcup_baseline/train/text
  train_labels_dir: D:/bisai/V2/2026_finvcup_baseline/train/labels
  output_root: D:/bisai/V2/2026_finvcup_baseline/outputs/dualchannel_temporal
  checkpoints_dir: D:/bisai/V2/2026_finvcup_baseline/outputs/dualchannel_temporal/checkpoints
  logs_dir: D:/bisai/V2/2026_finvcup_baseline/outputs/dualchannel_temporal/logs
  cache_root: D:/bisai/V2/2026_finvcup_baseline/.cache

labels:
  C: 0
  T: 1
  BC: 2
  I: 3
  NA: 4
  positive_ids: [1, 2, 3]
  multi_targets: [C, NA, I, BC, T]

split:
  valid_ratio: 0.1
  by_conversation: true

audio_encoder:
  type: whisper
  model_name: openai/whisper-large-v3
  proj_dim: 512
  freeze: true
  unfreeze_layers: 2
  dual_channel: true
  tail_frames: 400
  temporal_head:
    type: gru
    hidden_dim: 256
    num_layers: 1
    dropout: 0.1

text_encoder:
  model_name: Qwen/Qwen3-0.6B
  max_length: 256
  freeze_backbone: true

context_encoder:
  vocab_size: 5
  embed_dim: 24
  channels: [48, 96]
  tail_k: 75

fusion:
  hidden_dim: 320
  bilinear_rank: 64
  dropout: 0.25

train:
  multi_label: true
  pos_weight_mode: capped_per_label
  pos_weight_cap: 8.0
  focal_gamma: 2.0
  label_smoothing: 0.05
  epochs: 80
  max_steps_per_epoch: 20000
  eval_valid_sample_count: 8000
  eval_valid_max_batches: null
  eval_valid_shuffle: false
  batch_size: 4
  eval_batch_size: 4
  log_every_steps: 40
  learning_rate: 3.0e-5
  warmup_ratio: 0.05
  ema_decay: 0.995
  weight_decay: 0.01
  grad_clip_norm: 1.0
  use_amp: true
  gradient_accumulation_steps: 4
  save_metric: best_f1
  best_checkpoint_name: best_dualchannel_temporal.pt
  early_stop_patience: 15

env:
  HF_HOME: D:/bisai/V2/2026_finvcup_baseline/.cache/huggingface
  TRANSFORMERS_CACHE: D:/bisai/V2/2026_finvcup_baseline/.cache/huggingface
  TORCH_HOME: D:/bisai/V2/2026_finvcup_baseline/.cache/torch
  HF_ENDPOINT: https://hf-mirror.com
```

> `batch_size` raised 1 → 4 and `gradient_accumulation_steps` lowered 8 → 4 to use the 2×L20 headroom (effective batch unchanged at 16 if both GPUs are used via DDP). Tune up further if VRAM allows.

- [ ] **Step 2: End-to-end smoke run (downloads/uses cached whisper-large-v3)**

Run (very short, just to prove the pipeline trains + evals + saves without error):

```bash
python -m src.train --config configs/whisper_qwen0_6b_dualchannel_temporal.yaml --epochs 1 --max_steps_per_epoch 5
```

Expected: training logs print, an eval runs, a checkpoint is written under `outputs/dualchannel_temporal/checkpoints/`, no exceptions. (If `--epochs`/`--max_steps_per_epoch` are not CLI flags in your `train.py`, set them temporarily in the YAML instead.)

- [ ] **Step 3: Commit**

```bash
git add configs/whisper_qwen0_6b_dualchannel_temporal.yaml
git commit -m "feat: add dual-channel temporal training config"
```

---

## Task 9: Full regression run + leaderboard validation (manual)

**No code — verification only.**

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all green.

- [ ] **Step 2: Ablation A/B (optional but recommended)**

Train two short runs differing only in `audio_encoder.dual_channel` (`true` vs `false`) for a few epochs; compare `valid/macro_best_f1` in the logs. New path should be ≥ old path.

- [ ] **Step 3: Full train + infer + submit**

Train to convergence with the new config, run `scripts/run_infer.sh` (or `python -m src.infer_test ...`) to produce `pred_test1.csv`, and submit. Compare against the 0.736767 baseline.

---

## Self-Review Notes (author)

- **Spec coverage:** §5.1 boundary-align/mask → Tasks 2, 4, 5; §5.2 per-channel 2B forward → Task 6; §5.3 temporal head (GRU default + Transformer) → Task 3; §6 interface table → Tasks 5, 6, 7; §7 config → Task 8; §9 tests → Tasks 1-6, 9; §12 acceptance → Task 9.
- **Refinement vs spec:** the tail is **left-justified (pad at end)** rather than the spec's right-aligned sketch, so the BiGRU can use `pack_padded_sequence` cleanly and leading silence cannot leak into the boundary representation. Architecture, interface, and boundary semantics are unchanged; this is the faithful implementation of "boundary-aligned + masked tail."
- **Type consistency:** `gather_boundary_tail` returns `(tail, lengths, mask)` consumed identically by `TemporalHead.forward(tail, lengths, mask)`; `WhisperAudioEncoder(..., dual_channel, tail_frames, audio_len_samples, temporal_head_cfg)` matches the constructor call in `MultimodalTurnTakingModel`; `audio_valid_samples` is the dict key in dataset → collate → train/infer → model.
```
