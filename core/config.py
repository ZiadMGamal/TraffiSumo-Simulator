from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "Cooperative Smart Traffic MARL"
    version: str = "2.0.0"
    debug: bool = False

    sumo_home: Optional[str] = None
    sumo_config: str = "data/sumo/city_simulation.sumocfg"
    sumo_gui: bool = False
    simulation_steps: int = 3600
    max_lanes: int = 12

    database_url: str = "sqlite:///./traffic_analytics.db"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    algorithm: str = "dqn"
    state_dim: int = 26
    action_dim: int = 4
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_capacity: int = 100000
    total_episodes: int = 500
    target_sync_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    model_dir: str = "models"
    log_dir: str = "logs"

    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995

    cooperative_weight: float = 0.3
    local_reward_weight: float = 0.7

    redis_url: Optional[str] = None
    secret_key: str = "change-me-in-production"
    api_key_header: str = "X-API-Key"

    wandb_enabled: bool = False
    wandb_project: str = "traffic-marl"
    tensorboard_enabled: bool = True

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def sumo_config_path(self) -> Path:
        return self.project_root / self.sumo_config


@lru_cache
def get_settings() -> Settings:
    return Settings()
