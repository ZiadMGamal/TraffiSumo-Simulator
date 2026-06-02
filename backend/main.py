from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.middleware.rate_limit import RateLimitMiddleware
from backend.middleware.request_logging import RequestLoggingMiddleware
from backend.models import init_db
from backend.routes import api_router
from backend.websocket.traffic import register_websocket_routes
from core.config import get_settings
from core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging("traffic_marl.api")
    init_db()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description="Cooperative Multi-Agent Reinforcement Learning for Smart Traffic Management",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=200)
    app.include_router(api_router)
    register_websocket_routes(app)

    @app.get("/")
    def root():
        return {
            "service": settings.project_name,
            "version": settings.version,
            "docs": "/docs",
            "health": "/api/system/health",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug,
    )
