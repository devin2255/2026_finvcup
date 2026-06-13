# SSL Audio Encoder (HuBERT/WavLM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable `SSLAudioEncoder` (AutoModel: chinese-hubert-large by default) as a new `type: ssl` audio branch — raw-waveform input + GPU normalization (no CPU mel) — reusing the existing `gather_boundary_tail` + `TemporalHead` unchanged.

**Architecture:** Two new pure helpers (`end_aligned_mask`, `zero_mean_unit_var`) in `audio_temporal.py`; a new `SSLAudioEncoder` in `multimodal_baseline.py` that stacks the two channels to `[2B, T]`, GPU-normalizes over the valid (front-padded) region, runs an `AutoModel` backbone with an end-aligned `attention_mask`, then feeds the existing boundary-tail + temporal head. Whisper and CNN paths are retained for A/B.

**Tech Stack:** PyTorch, HuggingFace transformers (`AutoModel` → Hubert/WavLM/Wav2Vec2 Model), pytest.

**Design spec:** [docs/superpowers/specs/2026-06-13-ssl-audio-encoder-design.md](../specs/2026-06-13-ssl-audio-encoder-design.md)

**Interpreter:** use the project conda env for ALL python/pytest: `D:\anaconda\envs\finvcup\python.exe` (Python 3.12 / torch 2.12). The default `python` (Anaconda base 3.8.5) will fail.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/models/audio_temporal.py` | Add `end_aligned_mask` + `zero_mean_unit_var` (pure, encoder-agnostic, unit-testable). | Modify |
| `src/models/multimodal_baseline.py` | Add `SSLAudioEncoder`; add `type: ssl` branch in model `__init__`; generalize `forward` audio routing. | Modify |
| `configs/ssl_hubert_dualchannel_temporal.yaml` | New training config (`type: ssl`, chinese-hubert-large, server paths, bs16). | Create |
| `tests/test_ssl_audio_helpers.py` | Tests for the two pure helpers. | Create |
| `tests/test_ssl_audio_encoder.py` | Stub-based tests for `SSLAudioEncoder` + routing invariant. | Create |

**Shared signatures (must match across tasks):**

```python
# src/models/audio_temporal.py
def end_aligned_mask(seq_len: int, valid_lengths: Tensor, device=None) -> Tensor:   # [B, seq_len] bool
def zero_mean_unit_var(x: Tensor, mask: Tensor, eps: float = 1e-7) -> Tensor:        # [B, L]

# src/models/multimodal_baseline.py
class SSLAudioEncoder(nn.Module):
    def __init__(self, model_name, sample_rate, proj_dim, freeze=True,
                 unfreeze_layers=0, dual_channel=True, tail_frames=400,
                 audio_len_samples=480000, temporal_head_cfg=None): ...
    def forward(self, wave, valid_samples=None) -> Tensor:   # [B, proj_dim]
```

---

## Task 1: Pure helpers `end_aligned_mask` + `zero_mean_unit_var`

**Files:**
- Modify: `src/models/audio_temporal.py`
- Test: `tests/test_ssl_audio_helpers.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ssl_audio_helpers.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\anaconda\envs\finvcup\python.exe -m pytest tests/test_ssl_audio_helpers.py -v`
Expected: FAIL with `ImportError: cannot import name 'end_aligned_mask'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/models/audio_temporal.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\anaconda\envs\finvcup\python.exe -m pytest tests/test_ssl_audio_helpers.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/audio_temporal.py tests/test_ssl_audio_helpers.py
git commit -m "feat: add end_aligned_mask and zero_mean_unit_var audio helpers"
```

---

## Task 2: `SSLAudioEncoder`

**Files:**
- Modify: `src/models/multimodal_baseline.py`
- Test: `tests/test_ssl_audio_encoder.py`

- [ ] **Step 1: Write the failing test (stubbed AutoModel, no download)**

Create `tests/test_ssl_audio_encoder.py`:

```python
import torch
import torch.nn as nn
from src.models import multimodal_baseline as mb


class _StubEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])


class _StubBackbone(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden})()
        self.encoder = _StubEncoder()
        self.feature_extractor = nn.Linear(1, 1)

    def forward(self, input_values=None, attention_mask=None):
        b = input_values.shape[0]
        return type("O", (), {"last_hidden_state": torch.randn(b, 1500, self.config.hidden_size)})()


class _StubAutoModel:
    @staticmethod
    def from_pretrained(*a, **k):
        return _StubBackbone()


def _make(monkeypatch, **kw):
    monkeypatch.setattr(mb, "AutoModel", _StubAutoModel)
    defaults = dict(model_name="stub", sample_rate=16000, proj_dim=8,
                    dual_channel=True, tail_frames=20, audio_len_samples=480000,
                    temporal_head_cfg={"type": "gru", "hidden_dim": 8, "num_layers": 1, "dropout": 0.0})
    defaults.update(kw)
    return mb.SSLAudioEncoder(**defaults)


def test_ssl_dual_forward_shape(monkeypatch):
    enc = _make(monkeypatch)
    enc.eval()
    wave = torch.randn(3, 2, 480000)
    vs = torch.tensor([480000, 240000, 48000])
    out = enc(wave, valid_samples=vs)
    assert out.shape == (3, 8)
    assert enc.out_dim == 8


def test_ssl_mono_and_no_valid_samples(monkeypatch):
    enc = _make(monkeypatch, dual_channel=False)
    enc.eval()
    out = enc(torch.randn(2, 2, 480000))   # valid_samples=None -> fully valid
    assert out.shape == (2, 8)


def test_ssl_unfreeze_last_layers(monkeypatch):
    enc = _make(monkeypatch, freeze=True, unfreeze_layers=1)
    assert enc.encoder_has_trainable_layers is True
    assert all(p.requires_grad for p in enc.backbone.encoder.layers[-1].parameters())
    assert not any(p.requires_grad for p in enc.backbone.encoder.layers[0].parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\anaconda\envs\finvcup\python.exe -m pytest tests/test_ssl_audio_encoder.py -v`
Expected: FAIL with `AttributeError: module 'src.models.multimodal_baseline' has no attribute 'SSLAudioEncoder'`

- [ ] **Step 3: Write minimal implementation**

In `src/models/multimodal_baseline.py`, extend the audio_temporal import (find the existing line `from src.models.audio_temporal import gather_boundary_tail, TemporalHead`) to:

```python
from src.models.audio_temporal import (
    gather_boundary_tail, TemporalHead, end_aligned_mask, zero_mean_unit_var,
)
```

Then add the `SSLAudioEncoder` class immediately after the `WhisperAudioEncoder` class:

```python
class SSLAudioEncoder(nn.Module):
    """Self-supervised speech encoder (HuBERT/WavLM/wav2vec2 via AutoModel).

    Raw-waveform input + GPU normalization (no CPU mel). Per-channel encoding,
    then the shared boundary-tail + temporal head. Output is [B, proj_dim],
    interface-compatible with WhisperAudioEncoder.
    """

    def __init__(self, model_name: str, sample_rate: int, proj_dim: int,
                 freeze: bool = True, unfreeze_layers: int = 0,
                 dual_channel: bool = True, tail_frames: int = 400,
                 audio_len_samples: int = 480000, temporal_head_cfg: dict | None = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.freeze = freeze
        self.dual_channel = dual_channel
        self.tail_frames = tail_frames
        self.audio_len_samples = audio_len_samples
        self.backbone = AutoModel.from_pretrained(model_name)
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # Keep the CNN feature front-end frozen; optionally unfreeze last N transformer layers.
        if unfreeze_layers > 0 and self.freeze:
            layers = self.backbone.encoder.layers
            total = len(layers)
            for idx in range(max(0, total - unfreeze_layers), total):
                for p in layers[idx].parameters():
                    p.requires_grad = True
        self.encoder_has_trainable_layers = any(p.requires_grad for p in self.backbone.parameters())
        hidden_size = int(self.backbone.config.hidden_size)

        cfg = temporal_head_cfg or {}
        self.temporal_head = TemporalHead(
            in_dim=2 * hidden_size, model_dim=proj_dim, out_dim=proj_dim,
            head_type=str(cfg.get("type", "gru")),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            num_layers=int(cfg.get("num_layers", 1)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
        self.out_dim = proj_dim

    def _run_backbone(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        if self.freeze and not self.encoder_has_trainable_layers:
            with torch.no_grad():
                return self.backbone(input_values=x, attention_mask=attn).last_hidden_state
        return self.backbone(input_values=x, attention_mask=attn).last_hidden_state

    def forward(self, wave: torch.Tensor, valid_samples: torch.Tensor | None = None) -> torch.Tensor:
        B, _, T = wave.shape
        if self.dual_channel:
            ch0, ch1 = wave[:, 0, :], wave[:, 1, :]
        else:
            mono = wave.mean(dim=1)
            ch0 = ch1 = mono
        x = torch.cat([ch0, ch1], dim=0)                       # [2B, T]
        if valid_samples is None:
            valid_samples = torch.full((B,), self.audio_len_samples,
                                       dtype=torch.long, device=wave.device)
        vs2 = torch.cat([valid_samples, valid_samples], dim=0)
        attn = end_aligned_mask(T, vs2, wave.device)           # [2B, T] bool
        with torch.amp.autocast("cuda", enabled=False):
            x = zero_mean_unit_var(x.float(), attn)
        hidden = self._run_backbone(x, attn.long())            # [2B, n_frames, H]
        tail, lengths, mask = gather_boundary_tail(
            hidden, B, valid_samples, self.tail_frames, self.audio_len_samples)
        return self.temporal_head(tail, lengths, mask)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\anaconda\envs\finvcup\python.exe -m pytest tests/test_ssl_audio_encoder.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/multimodal_baseline.py tests/test_ssl_audio_encoder.py
git commit -m "feat: add SSLAudioEncoder (HuBERT/WavLM via AutoModel, GPU normalization)"
```

---

## Task 3: Wire `type: ssl` branch + generalize forward routing

**Files:**
- Modify: `src/models/multimodal_baseline.py` (`MultimodalTurnTakingModel.__init__` and `.forward`)
- Test: `tests/test_ssl_audio_encoder.py` (append a routing-invariant test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ssl_audio_encoder.py`:

```python
def test_audio_routing_invariant():
    # The model.forward routes ONLY the CNN AudioEncoder without valid_samples.
    # Whisper and SSL encoders must NOT be subclasses of AudioEncoder.
    from src.models.multimodal_baseline import AudioEncoder, WhisperAudioEncoder, SSLAudioEncoder
    assert not issubclass(WhisperAudioEncoder, AudioEncoder)
    assert not issubclass(SSLAudioEncoder, AudioEncoder)
    assert issubclass(AudioEncoder, AudioEncoder)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\anaconda\envs\finvcup\python.exe -m pytest tests/test_ssl_audio_encoder.py::test_audio_routing_invariant -v`
Expected: PASS already (classes exist and are unrelated) — this guards the invariant. If it ERRORS on import, Task 2 is incomplete. Proceed to wire the branch so the invariant is actually used.

- [ ] **Step 3: Wire the model**

In `MultimodalTurnTakingModel.__init__`, the audio encoder is currently built as `if audio_type == "whisper": ... else: <CNN AudioEncoder>`. Change the `else:` that builds the CNN into an `elif`/`else` so SSL slots in. Replace this exact block:

```python
        else:
            self.audio_encoder = AudioEncoder(
                sample_rate=cfg["sample_rate"],
                n_mels=cfg["audio_encoder"]["n_mels"],
                conv_channels=cfg["audio_encoder"]["conv_channels"],
                dropout=cfg["audio_encoder"]["dropout"],
            )
```

with:

```python
        elif audio_type == "ssl":
            audio_len_samples = int(cfg["context_chunks"]) * int(cfg["chunk_ms"]) \
                * int(cfg["sample_rate"]) // 1000
            self.audio_encoder = SSLAudioEncoder(
                model_name=cfg["audio_encoder"]["model_name"],
                sample_rate=cfg["sample_rate"],
                proj_dim=int(cfg["audio_encoder"]["proj_dim"]),
                freeze=bool(cfg["audio_encoder"].get("freeze", True)),
                unfreeze_layers=int(cfg["audio_encoder"].get("unfreeze_layers", 0)),
                dual_channel=bool(cfg["audio_encoder"].get("dual_channel", True)),
                tail_frames=int(cfg["audio_encoder"].get("tail_frames", 400)),
                audio_len_samples=audio_len_samples,
                temporal_head_cfg=cfg["audio_encoder"].get("temporal_head"),
            )
        else:
            self.audio_encoder = AudioEncoder(
                sample_rate=cfg["sample_rate"],
                n_mels=cfg["audio_encoder"]["n_mels"],
                conv_channels=cfg["audio_encoder"]["conv_channels"],
                dropout=cfg["audio_encoder"]["dropout"],
            )
```

Then in `MultimodalTurnTakingModel.forward`, replace the audio routing block:

```python
        if isinstance(self.audio_encoder, WhisperAudioEncoder):
            audio_feat = self.audio_encoder(waveform, valid_samples=audio_valid_samples)
        else:
            audio_feat = self.audio_encoder(waveform)
```

with (route only the CNN `AudioEncoder` without `valid_samples`; Whisper + SSL both take it):

```python
        if isinstance(self.audio_encoder, AudioEncoder):
            audio_feat = self.audio_encoder(waveform)
        else:
            audio_feat = self.audio_encoder(waveform, valid_samples=audio_valid_samples)
```

- [ ] **Step 4: Run tests + compile check**

Run: `D:\anaconda\envs\finvcup\python.exe -m pytest tests/ -v`
Expected: PASS (all green)

Run: `D:\anaconda\envs\finvcup\python.exe -m py_compile src/models/multimodal_baseline.py`
Expected: no output (success)

- [ ] **Step 5: Commit**

```bash
git add src/models/multimodal_baseline.py tests/test_ssl_audio_encoder.py
git commit -m "feat: add type:ssl branch and generalize audio encoder forward routing"
```

---

## Task 4: New training config

**Files:**
- Create: `configs/ssl_hubert_dualchannel_temporal.yaml`

- [ ] **Step 1: Create the config**

Create `configs/ssl_hubert_dualchannel_temporal.yaml`:

```yaml
seed: 42
chunk_ms: 80
context_chunks: 375
target_chunks: 25
stride: 2
sample_rate: 16000
max_train_samples: null
max_valid_samples: null
num_workers: 8

data_augmentation:
  dynamic_context: true
  min_context_chunks: 125
  max_context_chunks: 375
  context_prob: 0.5

paths:
  project_root: /mnt/workspace/dorihue/2026_finvcup
  train_audio_dir: /mnt/workspace/dorihue/2026_finvcup/train/audio
  train_text_dir: /mnt/workspace/dorihue/2026_finvcup/train/text
  train_labels_dir: /mnt/workspace/dorihue/2026_finvcup/train/labels
  output_root: /mnt/workspace/dorihue/2026_finvcup/outputs/ssl_hubert
  checkpoints_dir: /mnt/workspace/dorihue/2026_finvcup/outputs/ssl_hubert/checkpoints
  logs_dir: /mnt/workspace/dorihue/2026_finvcup/outputs/ssl_hubert/logs
  cache_root: /mnt/workspace/dorihue/2026_finvcup/.cache

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
  type: ssl
  model_name: TencentGameMate/chinese-hubert-large
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
  batch_size: 16
  eval_batch_size: 16
  log_every_steps: 40
  learning_rate: 3.0e-5
  warmup_ratio: 0.05
  ema_decay: 0.995
  weight_decay: 0.01
  grad_clip_norm: 1.0
  use_amp: true
  gradient_accumulation_steps: 1
  save_metric: best_f1
  best_checkpoint_name: best_ssl_hubert.pt
  early_stop_patience: 15

env:
  HF_HOME: /mnt/workspace/dorihue/2026_finvcup/.cache/huggingface
  TRANSFORMERS_CACHE: /mnt/workspace/dorihue/2026_finvcup/.cache/huggingface
  TORCH_HOME: /mnt/workspace/dorihue/2026_finvcup/.cache/torch
  HF_ENDPOINT: https://hf-mirror.com
```

- [ ] **Step 2: Validate the config**

Run:

```bash
D:\anaconda\envs\finvcup\python.exe -c "import yaml; c=yaml.safe_load(open('configs/ssl_hubert_dualchannel_temporal.yaml',encoding='utf-8')); ae=c['audio_encoder']; assert ae['type']=='ssl' and 'hubert' in ae['model_name'] and ae['dual_channel'] is True and ae['tail_frames']==400; print('ssl config ok', c['train']['batch_size'])"
```

Expected: `ssl config ok 16`

- [ ] **Step 3: Commit**

```bash
git add configs/ssl_hubert_dualchannel_temporal.yaml
git commit -m "feat: add chinese-hubert-large SSL training config"
```

---

## Task 5: Full suite + final review (manual train deferred to user)

**No code — verification only.**

- [ ] **Step 1: Run the full test suite**

Run: `D:\anaconda\envs\finvcup\python.exe -m pytest tests/ -v`
Expected: all green (Task 1 helpers + Task 2/3 SSL + the pre-existing ① tests).

- [ ] **Step 2: Document handoff**

The end-to-end SSL smoke train + A/B vs Whisper need the real data and the chinese-hubert-large download (~1.2 GB via `HF_ENDPOINT`), so they are user-run on the server:

```bash
# from /mnt/workspace/dorihue/2026_finvcup (with train/ data present):
bash scripts/run_train.sh configs/ssl_hubert_dualchannel_temporal.yaml 2
# A/B: same config with audio_encoder.model_name swapped to microsoft/wavlm-large
#      or TencentGameMate/chinese-wav2vec2-large; or audio_encoder.type: whisper.
```

Verify: training starts, GPU-Util is higher and s/it lower than the Whisper run (no CPU mel), logits `[B,5]`, a checkpoint is written.

---

## Self-Review Notes (author)

- **Spec coverage:** §5.1 SSLAudioEncoder → Task 2; §5.2 GPU normalization (`zero_mean_unit_var`) → Task 1; §5.3 attention_mask (`end_aligned_mask`) → Task 1; §5.4 forward (dual + mono via duplicated mix + valid_samples=None) → Task 2; §6 interface (`type: ssl` branch + forward routing) → Task 3; §7 config → Task 4; §9 tests → Tasks 1–3; §12 acceptance → Task 5.
- **Reuse unchanged:** `gather_boundary_tail`, `TemporalHead` consumed exactly as in the ① plan (`gather_boundary_tail(hidden, B, valid_samples, tail_frames, audio_len_samples)` → `(tail, lengths, mask)` → `TemporalHead.forward(tail, lengths, mask)`). `in_dim=2*hidden_size` auto-adapts to HuBERT's 1024.
- **Type consistency:** `SSLAudioEncoder(model_name, sample_rate, proj_dim, freeze, unfreeze_layers, dual_channel, tail_frames, audio_len_samples, temporal_head_cfg)` matches the `type: ssl` construction in Task 3. `end_aligned_mask(seq_len, valid_lengths, device)` and `zero_mean_unit_var(x, mask, eps)` match their call sites in `SSLAudioEncoder.forward`. The `forward` routing uses `isinstance(self.audio_encoder, AudioEncoder)` (CNN only) so Whisper + SSL both receive `audio_valid_samples`.
- **No placeholders:** every step has complete code/commands.
