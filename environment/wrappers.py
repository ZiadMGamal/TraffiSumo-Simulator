from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


class NormalizeObservationWrapper(gym.Wrapper):
    def __init__(self, env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.running_mean: Dict[str, np.ndarray] = {}
        self.running_var: Dict[str, np.ndarray] = {}
        self.count = 0

    def _normalize(self, agent_id: str, obs: np.ndarray) -> np.ndarray:
        if agent_id not in self.running_mean:
            self.running_mean[agent_id] = np.zeros_like(obs)
            self.running_var[agent_id] = np.ones_like(obs)
        self.count += 1
        delta = obs - self.running_mean[agent_id]
        self.running_mean[agent_id] += delta / self.count
        delta2 = obs - self.running_mean[agent_id]
        self.running_var[agent_id] += delta * delta2
        std = np.sqrt(self.running_var[agent_id] / self.count + self.epsilon)
        return (obs - self.running_mean[agent_id]) / std

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return {k: self._normalize(k, v) for k, v in obs.items()}, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        obs = {k: self._normalize(k, v) for k, v in obs.items()}
        return obs, rewards, terminated, truncated, info


class EpisodeStatsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_rewards: Dict[str, float] = {}
        self.episode_steps = 0
        self.episode_metrics: Dict[str, list] = {}

    def reset(self, **kwargs):
        self.episode_rewards = {}
        self.episode_steps = 0
        self.episode_metrics = {}
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        self.episode_steps += 1
        for aid, reward in rewards.items():
            self.episode_rewards[aid] = self.episode_rewards.get(aid, 0.0) + reward
            if aid not in self.episode_metrics:
                self.episode_metrics[aid] = []
            self.episode_metrics[aid].append(info.get(aid, {}))
        if any(terminated.values()) if isinstance(terminated, dict) else terminated:
            info["episode_stats"] = {
                "steps": self.episode_steps,
                "rewards": self.episode_rewards.copy(),
                "metrics": self.episode_metrics,
            }
        return obs, rewards, terminated, truncated, info


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, stack_size: int = 4):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames: Dict[str, list] = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = {k: [v.copy() for _ in range(self.stack_size)] for k, v in obs.items()}
        return {k: np.concatenate(self.frames[k]) for k in obs}, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        stacked = {}
        for k, v in obs.items():
            self.frames[k].pop(0)
            self.frames[k].append(v.copy())
            stacked[k] = np.concatenate(self.frames[k])
        return stacked, rewards, terminated, truncated, info
