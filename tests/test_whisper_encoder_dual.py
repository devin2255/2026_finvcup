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
