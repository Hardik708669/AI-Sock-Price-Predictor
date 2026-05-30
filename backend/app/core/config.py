from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "StockVision AI"
    api_prefix: str = "/api/v1"
    environment: str = "development"
    app_debug: bool = False
    database_url: str = "sqlite:///./stockvision_local.db"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    model_registry_path: str = "app/ml/model_registry"
    alpha_vantage_api_key: str | None = None
    financial_news_api_key: str | None = None
    enable_transformer_sentiment: bool = False
    jwt_secret: str = Field(default="dev-secret-change-me", min_length=12)
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 30
    refresh_token_days: int = 14
    frontend_url: str = "http://localhost:5173"
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    firebase_project_id: str | None = None
    firebase_client_email: str | None = None
    firebase_private_key: str | None = None
    smtp_from_email: str = "noreply@stockvision.ai"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
