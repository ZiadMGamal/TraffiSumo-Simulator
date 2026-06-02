from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IntersectionMetrics(BaseModel):
    id: str
    queue: int = 0
    wait: float = 0.0
    reward: float = 0.0
    throughput: int = 0
    pressure: float = 0.0
    density: float = 0.0
    phase: int = 0


class TrafficSnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    intersections: List[IntersectionMetrics]


class AnalyticsSummary(BaseModel):
    avg_queue: float
    avg_wait: float
    total_throughput: int
    avg_reward: float = 0.0
    active_intersections: int = 0


class TrainingConfigSchema(BaseModel):
    algorithm: str = "dqn"
    total_episodes: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-4
    gamma: float = 0.99


class TrainingStatusResponse(BaseModel):
    status: str
    current_episode: int = 0
    total_episodes: int = 0
    algorithm: str = "dqn"
    message: Optional[str] = None


class EpisodeMetricsResponse(BaseModel):
    episode: int
    agent_id: str
    mean_reward: float
    mean_queue: float
    mean_wait: float


class HealthResponse(BaseModel):
    status: str
    version: str
    sumo_available: bool
    database_connected: bool


class AlgorithmInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, str]


class WebSocketMessage(BaseModel):
    type: str
    payload: dict
