"""Basic token authentication middleware."""

from __future__ import annotations

from typing import Callable, Awaitable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Optional bearer-token authentication.

    If *token* is empty, all requests are allowed (no-op).
    """

    # Paths that never require auth
    EXEMPT_PATHS = frozenset({"/api/health", "/docs", "/openapi.json"})

    def __init__(self, app: object, token: str = "") -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._token = token

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # No-op when token is unset
        if not self._token:
            return await call_next(request)

        # Skip auth for exempt paths and static files
        path = request.url.path
        if path in self.EXEMPT_PATHS or path.startswith("/static"):
            return await call_next(request)

        # WebSocket connections carry token as query param
        if path.startswith("/ws"):
            token = request.query_params.get("token", "")
        else:
            auth_header = request.headers.get("authorization", "")
            token = auth_header.removeprefix("Bearer ").strip()

        if token != self._token:
            return JSONResponse(
                {"detail": "Unauthorized"}, status_code=401
            )

        return await call_next(request)
