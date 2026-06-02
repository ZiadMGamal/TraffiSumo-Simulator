from fastapi import APIRouter

from backend.services.model_service import ModelService

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/")
def list_models():
    return ModelService().get_model_info()


@router.get("/checkpoints")
def list_checkpoints():
    return ModelService().list_checkpoints()
