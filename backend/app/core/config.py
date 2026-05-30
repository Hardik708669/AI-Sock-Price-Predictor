from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "StockVision AI"
    api_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://stockvision:stockvision@localhost:5432/stockvision"
    jwt_secret: str = Field(default="dev-secret-change-me", min_length=12)
    jwt_algorithm: str = "HS256"
    frontend_url: str = "http://localhost:5173"
    firebase_project_id: str | None = None
    firebase_client_email: str | None = None
    firebase_private_key: str | None = None
    news_api_key: str | None = None
    redis_url: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
