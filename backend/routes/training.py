from fastapi import APIRouter, BackgroundTasks

from backend.schemas import TrainingConfigSchema, TrainingStatusResponse
from backend.services.training_service import TrainingService
from core.registry import AlgorithmRegistry

router = APIRouter(prefix="/api/training", tags=["training"])


@router.get("/algorithms")
def list_algorithms():
    return [
        {
            "name": name,
            "description": f"Multi-agent {name.upper()} traffic signal control",
        }
        for name in AlgorithmRegistry.list_algorithms()
    ]


@router.post("/start", response_model=TrainingStatusResponse)
def start_training(config: TrainingConfigSchema):
    service = TrainingService.get_instance()
    result = service.start_training(
        algorithm=config.algorithm,
        total_episodes=config.total_episodes,
        config=config.model_dump(),
    )
    return TrainingStatusResponse(
        status=result.get("status", "started"),
        total_episodes=config.total_episodes,
        algorithm=config.algorithm,
    )


@router.post("/stop")
def stop_training():
    return TrainingService.get_instance().stop_training()


@router.get("/status", response_model=TrainingStatusResponse)
def get_training_status():
    data = TrainingService.get_instance().get_status()
    return TrainingStatusResponse(**data)
