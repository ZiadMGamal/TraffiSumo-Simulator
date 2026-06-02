from typing import Dict, List, Optional

import numpy as np


class RewardShaper:
    def __init__(
        self,
        pressure_weight: float = 1.0,
        wait_weight: float = 0.05,
        throughput_weight: float = 0.1,
        cooperative_weight: float = 0.3,
        local_weight: float = 0.7,
        fairness_weight: float = 0.05,
    ):
        self.pressure_weight = pressure_weight
        self.wait_weight = wait_weight
        self.throughput_weight = throughput_weight
        self.cooperative_weight = cooperative_weight
        self.local_weight = local_weight
        self.fairness_weight = fairness_weight
        self.prev_throughput: Dict[str, int] = {}

    def compute_local_reward(
        self,
        agent_id: str,
        pressure: float,
        total_wait: float,
        throughput_delta: int,
    ) -> float:
        reward = -(
            abs(pressure) * self.pressure_weight
            + total_wait * self.wait_weight
        )
        reward += throughput_delta * self.throughput_weight
        return float(reward)

    def compute_cooperative_rewards(
        self,
        local_rewards: Dict[str, float],
        neighbor_map: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, float]:
        cooperative = {}
        agent_ids = list(local_rewards.keys())
        for aid in agent_ids:
            if neighbor_map and aid in neighbor_map:
                neighbors = neighbor_map[aid]
            else:
                neighbors = [nid for nid in agent_ids if nid != aid]
            if neighbors:
                neighbor_reward = sum(local_rewards[n] for n in neighbors) / len(
                    neighbors
                )
            else:
                neighbor_reward = 0.0
            cooperative[aid] = (
                local_rewards[aid] * self.local_weight
                + neighbor_reward * self.cooperative_weight
            )
        return cooperative

    def apply_fairness_penalty(
        self,
        rewards: Dict[str, float],
        queue_lengths: Dict[str, float],
    ) -> Dict[str, float]:
        if not queue_lengths:
            return rewards
        avg_queue = np.mean(list(queue_lengths.values()))
        adjusted = {}
        for aid, reward in rewards.items():
            deviation = abs(queue_lengths.get(aid, 0) - avg_queue)
            adjusted[aid] = reward - deviation * self.fairness_weight
        return adjusted

    def reset(self) -> None:
        self.prev_throughput.clear()
