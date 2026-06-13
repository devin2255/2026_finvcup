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


def test_audio_routing_invariant():
    # The model.forward routes ONLY the CNN AudioEncoder without valid_samples.
    # Whisper and SSL encoders must NOT be subclasses of AudioEncoder.
    from src.models.multimodal_baseline import AudioEncoder, WhisperAudioEncoder, SSLAudioEncoder
    assert not issubclass(WhisperAudioEncoder, AudioEncoder)
    assert not issubclass(SSLAudioEncoder, AudioEncoder)
    assert issubclass(AudioEncoder, AudioEncoder)
