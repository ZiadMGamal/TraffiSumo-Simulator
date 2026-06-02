import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agents.base_agent import MultiAgentCoordinator
from agents.dqn_agent import DQNAgent
from core.config import get_settings
from core.logging import get_logger
from core.utils import ensure_dir, save_json
from core.env_loader import load_environment


class MARLEvaluator:
    def __init__(
        self,
        config_file: Optional[str] = None,
        model_dir: Optional[str] = None,
        use_gui: bool = False,
    ):
        self.settings = get_settings()
        self.logger = get_logger("evaluation")
        self.env = load_environment(
            "sumo" if config_file else "auto",
            config_file=config_file,
            use_gui=use_gui,
            fallback_mock=True,
        )
        self.model_dir = model_dir or self.settings.model_dir
        self.agents = {
            aid: DQNAgent(
                agent_id=aid,
                state_dim=self.settings.state_dim,
                action_dim=self.settings.action_dim,
            )
            for aid in self.env.agent_ids
        }
        self.coordinator = MultiAgentCoordinator(self.agents)
        self.coordinator.set_eval_mode()
        self._load_models()

    def _load_models(self) -> None:
        model_path = Path(self.model_dir)
        for aid in self.env.agent_ids:
            path = model_path / aid
            if (model_path / f"{aid}_policy.pth").exists():
                import torch

                self.agents[aid].policy_net.load_state_dict(
                    torch.load(model_path / f"{aid}_policy.pth")
                )
            elif path.exists() or (Path(str(path) + "_dqn.pt")).exists():
                self.agents[aid].load(str(path))

    def run_episode(self) -> Dict[str, Any]:
        obs, _ = self.env.reset()
        step = 0
        total_rewards = {aid: 0.0 for aid in self.env.agent_ids}
        metrics_history: Dict[str, List[Dict]] = {aid: [] for aid in self.env.agent_ids}
        while True:
            actions = self.coordinator.choose_actions(obs, explore=False)
            obs, rewards, dones, _, infos = self.env.step(actions)
            for aid in self.env.agent_ids:
                total_rewards[aid] += rewards[aid]
                metrics_history[aid].append(infos.get(aid, {}))
            step += 1
            if any(dones.values()):
                break
        aggregated = {}
        for aid in self.env.agent_ids:
            history = metrics_history[aid]
            aggregated[aid] = {
                "total_reward": total_rewards[aid],
                "avg_queue": float(
                    np.mean([m.get("queue_length", 0) for m in history])
                ),
                "avg_wait": float(
                    np.mean([m.get("waiting_time", 0) for m in history])
                ),
                "max_queue": float(
                    max([m.get("queue_length", 0) for m in history] or [0])
                ),
                "throughput": history[-1].get("throughput", 0) if history else 0,
            }
        return {"steps": step, "agents": aggregated}

    def run_benchmark(self, num_episodes: int = 10) -> Dict[str, Any]:
        results = []
        for ep in range(num_episodes):
            self.logger.info(f"Evaluation episode {ep + 1}/{num_episodes}")
            results.append(self.run_episode())
        summary = self._aggregate_results(results)
        output_dir = ensure_dir("evaluation_results")
        save_json(summary, output_dir / "benchmark_summary.json")
        return summary

    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        agent_ids = list(results[0]["agents"].keys()) if results else []
        summary = {"episodes": len(results), "agents": {}}
        for aid in agent_ids:
            rewards = [r["agents"][aid]["total_reward"] for r in results]
            queues = [r["agents"][aid]["avg_queue"] for r in results]
            waits = [r["agents"][aid]["avg_wait"] for r in results]
            summary["agents"][aid] = {
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "mean_queue": float(np.mean(queues)),
                "mean_wait": float(np.mean(waits)),
            }
        return summary

    def close(self) -> None:
        self.env.close()
