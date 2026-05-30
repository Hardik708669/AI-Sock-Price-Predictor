from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.firebase import initialize_firebase

settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=["180/minute"])

app = FastAPI(
    title="StockVision AI API",
    description="AI-powered stock intelligence, prediction, sentiment, risk, portfolio, and assistant APIs.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(_request, exc):
    return JSONResponse(status_code=429, content={"detail": f"Rate limit exceeded: {exc.detail}"})


@app.on_event("startup")
def startup() -> None:
    initialize_firebase()


app.include_router(router, prefix=settings.api_prefix)
