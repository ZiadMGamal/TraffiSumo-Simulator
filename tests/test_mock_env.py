import pytest

from environment.mock_env import MockTrafficEnv


@pytest.fixture
def env():
    e = MockTrafficEnv(num_intersections=3, max_steps=50)
    yield e
    e.close()


def test_reset(env):
    obs, info = env.reset(seed=42)
    assert len(obs) == 3
    assert info["mode"] == "mock"
    for aid in env.agent_ids:
        assert obs[aid].shape == (26,)


def test_step(env):
    obs, _ = env.reset(seed=0)
    actions = {aid: 0 for aid in env.agent_ids}
    next_obs, rewards, dones, truncated, infos = env.step(actions)
    assert len(rewards) == 3
    assert not truncated
    assert "global_state" in infos


def test_episode_terminates(env):
    obs, _ = env.reset()
    steps = 0
    while steps < 100:
        _, _, dones, _, _ = env.step({aid: 1 for aid in env.agent_ids})
        steps += 1
        if any(dones.values()):
            break
    assert steps <= 50
