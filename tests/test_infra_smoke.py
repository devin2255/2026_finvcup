import torch


def test_torch_imports_and_runs():
    x = torch.zeros(2, 3)
    assert x.shape == (2, 3)
