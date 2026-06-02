from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.config import get_settings
from core.registry import EnvironmentRegistry
from environment.observers import TrafficObserver
from environment.reward_shaping import RewardShaper


@EnvironmentRegistry.register("mock")
class MockTrafficEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_intersections: int = 4,
        max_steps: int = 3600,
        reward_shaper: Optional[RewardShaper] = None,
        **kwargs,
    ):
        super().__init__()
        settings = get_settings()
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_ids = [f"J{i}" for i in range(num_intersections)]
        self.obs_dim = settings.state_dim
        self.observer = TrafficObserver(max_lanes=settings.max_lanes)
        self.reward_shaper = reward_shaper or RewardShaper(
            cooperative_weight=settings.cooperative_weight,
            local_weight=settings.local_reward_weight,
        )
        self._queues = {aid: np.random.randint(0, 15, 4) for aid in self.agent_ids}
        self._waits = {aid: np.random.uniform(0, 60, 4) for aid in self.agent_ids}
        self._phases = {aid: 0 for aid in self.agent_ids}
        self._phase_durations = {aid: 0 for aid in self.agent_ids}
        self._throughput = 0
        self.action_spaces = {aid: spaces.Discrete(4) for aid in self.agent_ids}
        self.observation_spaces = {
            aid: spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)
            for aid in self.agent_ids
        }
        self._neighbor_map = {
            self.agent_ids[i]: [
                self.agent_ids[j]
                for j in range(len(self.agent_ids))
                if j != i and abs(i - j) <= 1
            ]
            for i in range(len(self.agent_ids))
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self._throughput = 0
        for aid in self.agent_ids:
            self._queues[aid] = np.random.randint(0, 10, 4)
            self._waits[aid] = np.random.uniform(0, 30, 4)
            self._phases[aid] = 0
            self._phase_durations[aid] = 0
        self.reward_shaper.reset()
        return self._get_observations(), {"agent_ids": self.agent_ids, "mode": "mock"}

    def step(self, actions: Dict[str, int]):
        self.current_step += 1
        for aid, action in actions.items():
            target = int(action) % 4
            if self._phases[aid] != target:
                self._phases[aid] = target
                self._phase_durations[aid] = 0
            else:
                self._phase_durations[aid] += 1
            effect = 1.0 if self._phase_durations[aid] > 5 else 0.3
            self._queues[aid] = np.clip(
                self._queues[aid]
                - np.random.poisson(2 * effect, size=self._queues[aid].shape)
                + np.random.poisson(1, size=self._queues[aid].shape),
                0,
                50,
            )
            self._waits[aid] = np.clip(
                self._waits[aid] - effect * 2 + np.random.uniform(0, 3, 4),
                0,
                300,
            )
        self._throughput += int(np.random.poisson(3))
        obs = self._get_observations()
        rewards = self._calculate_rewards()
        terminated = self.current_step >= self.max_steps
        dones = {aid: terminated for aid in self.agent_ids}
        infos = {aid: self._get_metrics(aid) for aid in self.agent_ids}
        infos["global_state"] = self.observer.build_global_state(obs)
        return obs, rewards, dones, False, infos

    def _get_observations(self) -> Dict[str, np.ndarray]:
        obs = {}
        for aid in self.agent_ids:
            state = self.observer.normalize_state(
                self._queues[aid].tolist(),
                self._waits[aid].tolist(),
                self._phases[aid],
                self._phase_durations[aid],
            )
            obs[aid] = state[: self.obs_dim]
        return obs

    def _calculate_rewards(self) -> Dict[str, float]:
        local = {}
        queues = {}
        for aid in self.agent_ids:
            pressure = float(self._queues[aid].sum() - self._queues[aid].mean())
            total_wait = float(self._waits[aid].sum())
            local[aid] = self.reward_shaper.compute_local_reward(
                aid, pressure, total_wait, np.random.randint(0, 3)
            )
            queues[aid] = float(self._queues[aid].sum())
        coop = self.reward_shaper.compute_cooperative_rewards(
            local, self._neighbor_map
        )
        return self.reward_shaper.apply_fairness_penalty(coop, queues)

    def _get_metrics(self, aid: str) -> Dict[str, float]:
        return {
            "queue_length": float(self._queues[aid].sum()),
            "waiting_time": float(self._waits[aid].sum()),
            "throughput": self._throughput,
            "pressure": float(self._queues[aid].sum() - 5),
            "density": float(self._queues[aid].mean() / 10),
            "phase": self._phases[aid],
            "phase_duration": self._phase_durations[aid],
        }

    def get_neighbor_map(self) -> Dict[str, list]:
        return self._neighbor_map

    def close(self):
        pass
