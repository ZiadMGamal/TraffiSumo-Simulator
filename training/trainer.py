import bootstrap
from typing import Any, Dict, Optional

from agents.base_agent import MultiAgentCoordinator
from agents.dqn_agent import DQNAgent
from core.config import get_settings
from core.logging import get_logger
from core.registry import AlgorithmRegistry
from core.utils import ensure_dir, set_seed
from core.env_loader import load_environment
from training.callbacks import CallbackList, CheckpointCallback, MetricsCallback
from training.checkpoint import CheckpointManager
from training.metrics import MetricsTracker


class MARLTrainer:
    def __init__(
        self,
        algorithm: Optional[str] = None,
        config_file: Optional[str] = None,
        seed: int = 42,
        callbacks: Optional[CallbackList] = None,
    ):
        self.settings = get_settings()
        self.algorithm = algorithm or self.settings.algorithm
        self.logger = get_logger("training.trainer")
        set_seed(seed)
        env_name = "sumo" if config_file else "auto"
        self.env = load_environment(
            env_name,
            config_file=config_file,
            use_gui=False,
            fallback_mock=True,
        )
        self.agents = self._create_agents()
        self.coordinator = MultiAgentCoordinator(self.agents)
        self.metrics_tracker = MetricsTracker(self.settings.log_dir)
        self.checkpoint_manager = CheckpointManager(self.settings.checkpoint_dir)
        self.callbacks = callbacks or CallbackList(
            [
                CheckpointCallback(save_interval=25),
                MetricsCallback(log_interval=100),
            ]
        )
        self.current_episode = 0
        self._init_database()

    def _create_agents(self) -> Dict[str, Any]:
        agents = {}
        for aid in self.env.agent_ids:
            try:
                agent = AlgorithmRegistry.get(
                    self.algorithm,
                    agent_id=aid,
                    state_dim=self.settings.state_dim,
                    action_dim=self.settings.action_dim,
                    buffer_capacity=self.settings.buffer_capacity,
                    gamma=self.settings.gamma,
                    lr=self.settings.learning_rate,
                )
            except KeyError:
                agent = DQNAgent(
                    agent_id=aid,
                    state_dim=self.settings.state_dim,
                    action_dim=self.settings.action_dim,
                    buffer_capacity=self.settings.buffer_capacity,
                )
            agents[aid] = agent
        return agents

    def _init_database(self) -> None:
        try:
            from backend.models import SessionLocal, init_db

            init_db()
            self.db = SessionLocal()
        except Exception:
            self.db = None

    def _log_to_db(self, episode: int, step: int, aid: str, info: dict, reward: float):
        if self.db is None:
            return
        try:
            from backend.models import MetricsLog

            entry = MetricsLog(
                episode=episode,
                step=step,
                intersection_id=aid,
                queue_length=int(info.get("queue_length", 0)),
                waiting_time=float(info.get("waiting_time", 0)),
                throughput=int(info.get("throughput", 0)),
                reward=float(reward),
                pressure=float(info.get("pressure", 0)),
                density=float(info.get("density", 0)),
            )
            self.db.add(entry)
        except Exception as e:
            self.logger.warning(f"DB log failed: {e}")

    def train(self, total_episodes: Optional[int] = None) -> Dict[str, Any]:
        episodes = total_episodes or self.settings.total_episodes
        batch_size = self.settings.batch_size
        self.callbacks.on_train_start(self)
        self.logger.info(
            f"Starting MARL training | Algorithm: {self.algorithm} | Episodes: {episodes}"
        )
        for episode in range(episodes):
            self.current_episode = episode
            self.callbacks.on_episode_start(episode, self)
            obs, _ = self.env.reset()
            step = 0
            episode_rewards = {aid: 0.0 for aid in self.env.agent_ids}
            while True:
                actions = self.coordinator.choose_actions(obs, explore=True)
                next_obs, rewards, dones, _, infos = self.env.step(actions)
                losses = {}
                for aid in self.env.agent_ids:
                    self.agents[aid].store_transition(
                        obs[aid], actions[aid], rewards[aid], next_obs[aid], dones[aid]
                    )
                    loss = self.agents[aid].update(batch_size)
                    losses[aid] = loss
                    episode_rewards[aid] += rewards[aid]
                    if step % 100 == 0:
                        self._log_to_db(
                            episode, step, aid, infos.get(aid, {}), rewards[aid]
                        )
                self.callbacks.on_step_end(
                    step, self, {"losses": losses, "rewards": rewards}
                )
                obs = next_obs
                step += 1
                if any(dones.values()):
                    break
            if self.db:
                self.db.commit()
            self.coordinator.sync_targets()
            self.metrics_tracker.end_episode(episode_rewards)
            episode_metrics = {
                "rewards": episode_rewards,
                "steps": step,
                "summary": self.metrics_tracker.get_summary(),
            }
            self.callbacks.on_episode_end(episode, self, episode_metrics)
            self.logger.info(f"Episode {episode}/{episodes} | Steps: {step}")
        self._save_models()
        self.metrics_tracker.save()
        self.callbacks.on_train_end(self)
        if self.db:
            self.db.close()
        self.env.close()
        return self.metrics_tracker.get_summary()

    def _save_models(self) -> None:
        model_dir = ensure_dir(self.settings.model_dir)
        self.coordinator.save_all(str(model_dir))
        self.logger.info(f"Models saved to {model_dir}")
