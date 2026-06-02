import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from core.config import get_settings

Base = declarative_base()
settings = get_settings()
engine = create_engine(
    settings.database_url, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class MetricsLog(Base):
    __tablename__ = "metrics_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    episode = Column(Integer, index=True)
    step = Column(Integer)
    intersection_id = Column(String, index=True)
    queue_length = Column(Integer)
    waiting_time = Column(Float)
    throughput = Column(Integer)
    reward = Column(Float)
    pressure = Column(Float, default=0.0)
    density = Column(Float, default=0.0)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    algorithm = Column(String)
    total_episodes = Column(Integer)
    status = Column(String, default="running")
    config_json = Column(String)
    final_metrics_json = Column(String, nullable=True)


class ModelCheckpoint(Base):
    __tablename__ = "model_checkpoints"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    agent_id = Column(String)
    path = Column(String)
    episode = Column(Integer)
    is_best = Column(Boolean, default=False)


def init_db():
    Base.metadata.create_all(bind=engine)
