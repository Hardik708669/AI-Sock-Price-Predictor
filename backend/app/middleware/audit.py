import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("stockvision.audit")


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request method=%s path=%s status=%s duration_ms=%.2f client=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request.client.host if request.client else "-",
        )
        return response
