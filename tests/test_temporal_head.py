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
