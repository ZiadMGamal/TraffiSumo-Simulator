import random
from typing import Dict, List

import numpy as np

from core.logging import get_logger


class BaselineController:
    def __init__(self, strategy: str = "fixed_cycle"):
        self.strategy = strategy
        self.phase_timers: Dict[str, int] = {}
        self.current_phases: Dict[str, int] = {}

    def reset(self, agent_ids: List[str]) -> None:
        for aid in agent_ids:
            self.phase_timers[aid] = 0
            self.current_phases[aid] = 0

    def choose_actions(self, agent_ids: List[str], observations: Dict = None) -> Dict[str, int]:
        actions = {}
        for aid in agent_ids:
            if self.strategy == "random":
                actions[aid] = random.randint(0, 3)
            elif self.strategy == "max_pressure":
                actions[aid] = self._max_pressure_action(aid, observations)
            else:
                self.phase_timers[aid] += 1
                if self.phase_timers[aid] >= 30:
                    self.current_phases[aid] = (self.current_phases[aid] + 1) % 4
                    self.phase_timers[aid] = 0
                actions[aid] = self.current_phases[aid]
        return actions

    def _max_pressure_action(self, agent_id: str, observations: Dict) -> int:
        if not observations or agent_id not in observations:
            return self.current_phases.get(agent_id, 0)
        obs = observations[agent_id]
        phase_scores = []
        for phase in range(4):
            start = phase * 3
            score = np.sum(obs[start : start + 12]) if len(obs) > start + 12 else 0
            phase_scores.append(score)
        best = int(np.argmax(phase_scores))
        self.current_phases[agent_id] = best
        return best


class BenchmarkSuite:
    def __init__(self, env, strategies: List[str] = None):
        self.env = env
        self.strategies = strategies or [
            "fixed_cycle",
            "random",
            "max_pressure",
        ]
        self.logger = get_logger("benchmarks")
        self.results: Dict[str, list] = {}

    def run_all(self, episodes_per_strategy: int = 5) -> Dict[str, Dict]:
        for strategy in self.strategies:
            self.logger.info(f"Running baseline: {strategy}")
            controller = BaselineController(strategy)
            strategy_results = []
            for ep in range(episodes_per_strategy):
                obs, _ = self.env.reset()
                controller.reset(self.env.agent_ids)
                total_reward = {aid: 0.0 for aid in self.env.agent_ids}
                steps = 0
                while True:
                    actions = controller.choose_actions(self.env.agent_ids, obs)
                    obs, rewards, dones, _, _ = self.env.step(actions)
                    for aid in self.env.agent_ids:
                        total_reward[aid] += rewards[aid]
                    steps += 1
                    if any(dones.values()):
                        break
                strategy_results.append(
                    {"steps": steps, "rewards": total_reward}
                )
            self.results[strategy] = strategy_results
        return self._summarize()

    def _summarize(self) -> Dict[str, Dict]:
        summary = {}
        for strategy, results in self.results.items():
            all_rewards = []
            for r in results:
                all_rewards.extend(r["rewards"].values())
            summary[strategy] = {
                "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0,
                "episodes": len(results),
            }
        return summary
