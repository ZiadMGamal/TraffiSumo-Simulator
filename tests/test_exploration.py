import numpy as np

from agents.exploration import (
    BoltzmannExploration,
    CosineSchedule,
    EpsilonGreedy,
    LinearSchedule,
    OrnsteinUhlenbeckNoise,
)


def test_epsilon_greedy_decay():
    explorer = EpsilonGreedy(4, epsilon_start=1.0, epsilon_decay=0.5)
    explorer.decay()
    assert explorer.epsilon == 0.5


def test_boltzmann_select():
    explorer = BoltzmannExploration(4, temperature=1.0)
    action = explorer.select(np.array([1.0, 2.0, 3.0, 0.5]))
    assert 0 <= action < 4


def test_ou_noise():
    noise = OrnsteinUhlenbeckNoise(3)
    sample = noise.sample()
    assert sample.shape == (3,)


def test_schedules():
    linear = LinearSchedule(1.0, 0.1, duration=10)
    cosine = CosineSchedule(1.0, 0.1, duration=10)
    for _ in range(5):
        linear.tick()
        cosine.tick()
    assert linear.value() < 1.0
    assert cosine.value() < 1.0
