import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.utils import ensure_dir, moving_average


class MetricsTracker:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = ensure_dir(log_dir)
        self.episode_rewards: Dict[str, List[float]] = defaultdict(list)
        self.episode_losses: Dict[str, List[float]] = defaultdict(list)
        self.global_metrics: List[Dict[str, Any]] = []
        self.current_episode = 0

    def log_step(
        self,
        agent_id: str,
        reward: float,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        entry = {
            "episode": self.current_episode,
            "agent_id": agent_id,
            "reward": reward,
            "loss": loss,
            "metrics": metrics or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.global_metrics.append(entry)

    def end_episode(self, episode_rewards: Dict[str, float]) -> None:
        for aid, reward in episode_rewards.items():
            self.episode_rewards[aid].append(reward)
        self.current_episode += 1

    def get_summary(self) -> Dict[str, Any]:
        summary = {"episode": self.current_episode, "agents": {}}
        for aid, rewards in self.episode_rewards.items():
            if rewards:
                summary["agents"][aid] = {
                    "mean_reward": float(np.mean(rewards[-100:])),
                    "std_reward": float(np.std(rewards[-100:])),
                    "best_reward": float(max(rewards)),
                    "moving_avg": moving_average(rewards, 10)[-1]
                    if rewards
                    else 0.0,
                }
        return summary

    def save(self, filename: str = "metrics.json") -> Path:
        path = self.log_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": self.get_summary(),
                    "episode_rewards": dict(self.episode_rewards),
                    "global_metrics": self.global_metrics[-10000:],
                },
                f,
                indent=2,
            )
        return path

    def reset(self) -> None:
        self.episode_rewards.clear()
        self.episode_losses.clear()
        self.global_metrics.clear()
        self.current_episode = 0
