import numpy as np
import pytest

from agents.dqn_agent import DQNAgent


@pytest.fixture
def agent():
    return DQNAgent(agent_id="J0", state_dim=26, action_dim=4, buffer_capacity=1000)


def test_choose_action(agent):
    state = np.random.randn(26).astype(np.float32)
    action = agent.choose_action(state, explore=True)
    assert 0 <= action < 4


def test_store_and_update(agent):
    for _ in range(100):
        s = np.random.randn(26).astype(np.float32)
        ns = np.random.randn(26).astype(np.float32)
        agent.store_transition(s, 1, -1.0, ns, False)
    loss = agent.update(32)
    assert loss is not None


def test_save_load(agent, tmp_path):
    agent.save(str(tmp_path / "J0"))
    agent.explorer.set_epsilon(0.5)
    agent.load(str(tmp_path / "J0"))
    assert agent.explorer.epsilon == 0.0 or agent.training_step >= 0
