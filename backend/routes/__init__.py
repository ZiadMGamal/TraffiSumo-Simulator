from fastapi import APIRouter

from backend.routes.analytics import router as analytics_router
from backend.routes.training import router as training_router
from backend.routes.system import router as system_router
from backend.routes.simulation import router as simulation_router
from backend.routes.models import router as models_router

api_router = APIRouter()
api_router.include_router(analytics_router)
api_router.include_router(training_router)
api_router.include_router(system_router)
api_router.include_router(simulation_router)
api_router.include_router(models_router)
