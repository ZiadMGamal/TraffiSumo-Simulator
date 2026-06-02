import json
import threading
from typing import Any, Dict, Optional

from backend.models import SessionLocal, TrainingRun
from core.config import get_settings
from core.logging import get_logger


class TrainingService:
    _instance: Optional["TrainingService"] = None

    def __init__(self):
        self.logger = get_logger("training.service")
        self.thread: Optional[threading.Thread] = None
        self.status = "idle"
        self.current_episode = 0
        self.total_episodes = 0
        self.algorithm = "dqn"
        self._stop_event = threading.Event()

    @classmethod
    def get_instance(cls) -> "TrainingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_training(
        self,
        algorithm: str = "dqn",
        total_episodes: int = 100,
        config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if self.status == "running":
            return {"status": "already_running"}
        self.status = "running"
        self.algorithm = algorithm
        self.total_episodes = total_episodes
        self._stop_event.clear()
        settings = get_settings()
        settings.algorithm = algorithm
        settings.total_episodes = total_episodes
        db = SessionLocal()
        run = TrainingRun(
            algorithm=algorithm,
            total_episodes=total_episodes,
            status="running",
            config_json=json.dumps(config or {}),
        )
        db.add(run)
        db.commit()
        run_id = run.id
        db.close()

        def _train():
            try:
                import bootstrap
                from training.trainer import MARLTrainer

                trainer = MARLTrainer(algorithm=algorithm)
                trainer.train(total_episodes=total_episodes)
                self.status = "completed"
            except Exception as e:
                self.logger.error(f"Training failed: {e}")
                self.status = "failed"
            finally:
                db = SessionLocal()
                run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
                if run:
                    run.status = self.status
                    db.commit()
                db.close()

        self.thread = threading.Thread(target=_train, daemon=True)
        self.thread.start()
        return {"status": "started", "algorithm": algorithm, "episodes": total_episodes}

    def stop_training(self) -> Dict[str, str]:
        self._stop_event.set()
        self.status = "stopped"
        return {"status": "stopped"}

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "current_episode": self.current_episode,
            "total_episodes": self.total_episodes,
            "algorithm": self.algorithm,
        }
