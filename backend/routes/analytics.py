from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.dependencies import get_db
from backend.schemas import AnalyticsSummary, EpisodeMetricsResponse
from backend.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
def get_summary(
    limit: int = Query(100, ge=1, le=10000),
    db: Session = Depends(get_db),
):
    return AnalyticsService(db).get_summary(limit)


@router.get("/episodes/{episode}", response_model=List[EpisodeMetricsResponse])
def get_episode_metrics(episode: int, db: Session = Depends(get_db)):
    return AnalyticsService(db).get_episode_metrics(episode)


@router.get("/timeseries/{intersection_id}")
def get_time_series(
    intersection_id: str,
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db),
):
    return AnalyticsService(db).get_time_series(intersection_id, hours)


@router.get("/training-history")
def get_training_history(db: Session = Depends(get_db)):
    return AnalyticsService(db).get_training_history()


@router.get("/leaderboard")
def get_leaderboard(db: Session = Depends(get_db)):
    return AnalyticsService(db).get_leaderboard()
