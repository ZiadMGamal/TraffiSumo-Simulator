import bootstrap
from core.registry import AlgorithmRegistry, EnvironmentRegistry


def test_algorithms_registered():
    algorithms = AlgorithmRegistry.list_algorithms()
    assert "dqn" in algorithms
    assert "ppo" in algorithms
    assert len(algorithms) >= 4


def test_environments_registered():
    environments = EnvironmentRegistry.list_environments()
    assert "sumo" in environments
    assert "mock" in environments


def test_create_dqn_agent():
    agent = AlgorithmRegistry.get("dqn", agent_id="test", state_dim=26, action_dim=4)
    assert agent.agent_id == "test"


def test_create_mock_env():
    env = EnvironmentRegistry.get("mock", num_intersections=2, max_steps=10)
    obs, _ = env.reset()
    assert len(obs) == 2
    env.close()
