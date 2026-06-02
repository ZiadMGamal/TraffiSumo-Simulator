import numpy as np
import pytest

from agents.replay_buffer import (
    CircularReplayBuffer,
    NStepReplayBuffer,
    PrioritizedReplayBuffer,
)


@pytest.fixture
def sample_transition():
    state = np.random.randn(26).astype(np.float32)
    next_state = np.random.randn(26).astype(np.float32)
    return state, 1, -1.0, next_state, False


def test_circular_buffer_push_and_sample(sample_transition):
    buffer = CircularReplayBuffer(100)
    for _ in range(50):
        buffer.push(*sample_transition)
    assert len(buffer) == 50
    batch = buffer.sample(16)
    assert batch[0].shape == (16, 26)
    assert batch[1].shape == (16,)


def test_prioritized_buffer_sample(sample_transition):
    buffer = PrioritizedReplayBuffer(100)
    for _ in range(64):
        buffer.push(*sample_transition)
    batch = buffer.sample(32)
    assert len(batch) == 7
    assert batch[0].shape[0] == 32


def test_nstep_buffer(sample_transition):
    buffer = NStepReplayBuffer(100, n_step=3)
    for _ in range(10):
        buffer.push(*sample_transition)
    assert len(buffer) >= 1
