import math
import random
from typing import Optional

import numpy as np


class EpsilonGreedy:
    def __init__(
        self,
        action_dim: int,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select(self, q_values: Optional[np.ndarray] = None) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        if q_values is not None:
            return int(np.argmax(q_values))
        return random.randint(0, self.action_dim - 1)

    def decay(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_epsilon(self, value: float) -> None:
        self.epsilon = max(self.epsilon_min, min(1.0, value))


class BoltzmannExploration:
    def __init__(self, action_dim: int, temperature: float = 1.0, min_temp: float = 0.1):
        self.action_dim = action_dim
        self.temperature = temperature
        self.min_temp = min_temp

    def select(self, q_values: np.ndarray) -> int:
        scaled = q_values / max(self.temperature, 1e-8)
        exp_q = np.exp(scaled - np.max(scaled))
        probs = exp_q / exp_q.sum()
        return int(np.random.choice(self.action_dim, p=probs))

    def decay(self, factor: float = 0.995) -> None:
        self.temperature = max(self.min_temp, self.temperature * factor)


class OrnsteinUhlenbeckNoise:
    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            self.action_dim
        )
        self.state = self.state + dx
        return self.state.copy()

    def reset(self) -> None:
        self.state = np.ones(self.action_dim) * self.mu


class LinearSchedule:
    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = max(1, duration)
        self.step = 0

    def value(self) -> float:
        fraction = min(1.0, self.step / self.duration)
        return self.start + fraction * (self.end - self.start)

    def tick(self) -> float:
        self.step += 1
        return self.value()


class CosineSchedule:
    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = max(1, duration)
        self.step = 0

    def value(self) -> float:
        fraction = min(1.0, self.step / self.duration)
        return self.end + 0.5 * (self.start - self.end) * (
            1 + math.cos(math.pi * fraction)
        )

    def tick(self) -> float:
        self.step += 1
        return self.value()
