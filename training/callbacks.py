from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.logging import get_logger


class BaseCallback(ABC):
    def on_train_start(self, trainer) -> None:
        pass

    def on_episode_start(self, episode: int, trainer) -> None:
        pass

    def on_step_end(self, step: int, trainer, info: Dict[str, Any]) -> None:
        pass

    def on_episode_end(self, episode: int, trainer, metrics: Dict[str, Any]) -> None:
        pass

    def on_train_end(self, trainer) -> None:
        pass


class CallbackList(BaseCallback):
    def __init__(self, callbacks: List[BaseCallback]):
        self.callbacks = callbacks

    def _call_all(self, method: str, *args, **kwargs) -> None:
        for cb in self.callbacks:
            getattr(cb, method)(*args, **kwargs)

    def on_train_start(self, trainer) -> None:
        self._call_all("on_train_start", trainer)

    def on_episode_start(self, episode: int, trainer) -> None:
        self._call_all("on_episode_start", episode, trainer)

    def on_step_end(self, step: int, trainer, info: Dict[str, Any]) -> None:
        self._call_all("on_step_end", step, trainer, info)

    def on_episode_end(self, episode: int, trainer, metrics: Dict[str, Any]) -> None:
        self._call_all("on_episode_end", episode, trainer, metrics)

    def on_train_end(self, trainer) -> None:
        self._call_all("on_train_end", trainer)


class CheckpointCallback(BaseCallback):
    def __init__(self, save_interval: int = 10):
        self.save_interval = save_interval

    def on_episode_end(self, episode: int, trainer, metrics: Dict[str, Any]) -> None:
        if episode % self.save_interval == 0:
            trainer.checkpoint_manager.save(
                trainer.coordinator.save_all,
                episode,
                metrics,
            )


class MetricsCallback(BaseCallback):
    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self.logger = get_logger("training.metrics")

    def on_step_end(self, step: int, trainer, info: Dict[str, Any]) -> None:
        if step % self.log_interval == 0:
            losses = info.get("losses", {})
            avg_loss = sum(v for v in losses.values() if v) / max(
                len([v for v in losses.values() if v]), 1
            )
            self.logger.info(
                f"Episode {trainer.current_episode} Step {step} Loss {avg_loss:.4f}"
            )

    def on_episode_end(self, episode: int, trainer, metrics: Dict[str, Any]) -> None:
        self.logger.info(f"Episode {episode} completed | Metrics: {metrics}")


class TensorBoardCallback(BaseCallback):
    def __init__(self, log_dir: str = "logs/tensorboard"):
        self.writer = None
        self.log_dir = log_dir

    def on_train_start(self, trainer) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            self.writer = None

    def on_episode_end(self, episode: int, trainer, metrics: Dict[str, Any]) -> None:
        if self.writer is None:
            return
        for aid, reward in metrics.get("rewards", {}).items():
            self.writer.add_scalar(f"reward/{aid}", reward, episode)
        self.writer.flush()

    def on_train_end(self, trainer) -> None:
        if self.writer:
            self.writer.close()
