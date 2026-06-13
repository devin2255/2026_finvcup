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
