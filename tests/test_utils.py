import torch
import torch.nn as nn

from core.utils import hard_update, moving_average, set_seed, soft_update


def test_moving_average():
    result = moving_average([1.0, 2.0, 3.0, 4.0], window=2)
    assert len(result) == 4
    assert result[-1] == 3.5


def test_soft_and_hard_update():
    source = nn.Linear(10, 5)
    target = nn.Linear(10, 5)
    source.weight.data.fill_(1.0)
    target.weight.data.fill_(0.0)
    soft_update(target, source, tau=0.5)
    assert float(target.weight.mean()) == 0.5
    hard_update(target, source)
    assert float(target.weight.mean()) == 1.0


def test_set_seed():
    set_seed(123)
    a = torch.randn(3)
    set_seed(123)
    b = torch.randn(3)
    assert torch.allclose(a, b)
