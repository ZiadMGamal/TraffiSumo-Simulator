import os

from fastapi import APIRouter

from backend.schemas import HealthResponse
from core.config import get_settings
from core.registry import AlgorithmRegistry, EnvironmentRegistry

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    settings = get_settings()
    sumo_available = bool(os.environ.get("SUMO_HOME"))
    try:
        from sqlalchemy import text

        from backend.models import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return HealthResponse(
        status="healthy",
        version=settings.version,
        sumo_available=sumo_available,
        database_connected=db_ok,
    )


@router.get("/info")
def system_info():
    settings = get_settings()
    return {
        "project": settings.project_name,
        "version": settings.version,
        "algorithms": AlgorithmRegistry.list_algorithms(),
        "environments": EnvironmentRegistry.list_environments(),
        "config": {
            "state_dim": settings.state_dim,
            "action_dim": settings.action_dim,
            "sumo_config": settings.sumo_config,
        },
    }
