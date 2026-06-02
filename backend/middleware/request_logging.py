import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.logging import get_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger("api.request")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        self.logger.info(
            f"{request.method} {request.url.path} {response.status_code} {duration_ms:.2f}ms"
        )
        return response
