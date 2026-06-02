from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models import MetricsLog, TrainingRun
from backend.schemas import AnalyticsSummary, EpisodeMetricsResponse


class AnalyticsService:
    def __init__(self, db: Session):
        self.db = db

    def get_summary(self, limit: int = 100) -> AnalyticsSummary:
        logs = (
            self.db.query(MetricsLog)
            .order_by(MetricsLog.id.desc())
            .limit(limit)
            .all()
        )
        if not logs:
            return AnalyticsSummary(
                avg_queue=0,
                avg_wait=0,
                total_throughput=0,
                avg_reward=0,
                active_intersections=0,
            )
        return AnalyticsSummary(
            avg_queue=round(sum(l.queue_length for l in logs) / len(logs), 2),
            avg_wait=round(sum(l.waiting_time for l in logs) / len(logs), 2),
            total_throughput=max(l.throughput for l in logs),
            avg_reward=round(sum(l.reward for l in logs) / len(logs), 4),
            active_intersections=len(set(l.intersection_id for l in logs)),
        )

    def get_episode_metrics(self, episode: int) -> List[EpisodeMetricsResponse]:
        logs = (
            self.db.query(MetricsLog).filter(MetricsLog.episode == episode).all()
        )
        by_agent: Dict[str, list] = {}
        for log in logs:
            by_agent.setdefault(log.intersection_id, []).append(log)
        results = []
        for aid, agent_logs in by_agent.items():
            results.append(
                EpisodeMetricsResponse(
                    episode=episode,
                    agent_id=aid,
                    mean_reward=sum(l.reward for l in agent_logs) / len(agent_logs),
                    mean_queue=sum(l.queue_length for l in agent_logs)
                    / len(agent_logs),
                    mean_wait=sum(l.waiting_time for l in agent_logs)
                    / len(agent_logs),
                )
            )
        return results

    def get_time_series(
        self, intersection_id: str, hours: int = 24
    ) -> List[Dict]:
        since = datetime.utcnow() - timedelta(hours=hours)
        logs = (
            self.db.query(MetricsLog)
            .filter(
                MetricsLog.intersection_id == intersection_id,
                MetricsLog.timestamp >= since,
            )
            .order_by(MetricsLog.timestamp.asc())
            .all()
        )
        return [
            {
                "timestamp": l.timestamp.isoformat(),
                "queue": l.queue_length,
                "wait": l.waiting_time,
                "reward": l.reward,
                "throughput": l.throughput,
            }
            for l in logs
        ]

    def get_training_history(self) -> List[Dict]:
        runs = (
            self.db.query(TrainingRun)
            .order_by(TrainingRun.started_at.desc())
            .limit(20)
            .all()
        )
        return [
            {
                "id": r.id,
                "algorithm": r.algorithm,
                "status": r.status,
                "episodes": r.total_episodes,
                "started_at": r.started_at.isoformat() if r.started_at else None,
            }
            for r in runs
        ]

    def get_leaderboard(self) -> List[Dict]:
        subq = (
            self.db.query(
                MetricsLog.intersection_id,
                func.avg(MetricsLog.reward).label("avg_reward"),
                func.avg(MetricsLog.queue_length).label("avg_queue"),
            )
            .group_by(MetricsLog.intersection_id)
            .all()
        )
        return sorted(
            [
                {
                    "intersection_id": row[0],
                    "avg_reward": round(float(row[1] or 0), 4),
                    "avg_queue": round(float(row[2] or 0), 2),
                }
                for row in subq
            ],
            key=lambda x: x["avg_reward"],
            reverse=True,
        )
