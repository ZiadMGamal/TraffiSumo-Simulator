import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 120):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.clients: dict = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60
        self.clients[client_ip] = [
            t for t in self.clients[client_ip] if t > window_start
        ]
        if len(self.clients[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
            )
        self.clients[client_ip].append(now)
        return await call_next(request)
