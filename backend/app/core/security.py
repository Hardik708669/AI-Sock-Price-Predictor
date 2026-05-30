from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt
from jose.exceptions import JWTError
from passlib.context import CryptContext

from app.core.config import get_settings

password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return password_context.verify(password, password_hash)


def create_token(subject: str, token_type: str, expires_delta: timedelta, extra: dict[str, Any] | None = None) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "exp": now + expires_delta,
        "iat": now,
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(subject: str, extra: dict[str, Any] | None = None) -> str:
    settings = get_settings()
    return create_token(subject, "access", timedelta(minutes=settings.access_token_minutes), extra)


def create_refresh_token(subject: str, extra: dict[str, Any] | None = None) -> str:
    settings = get_settings()
    return create_token(subject, "refresh", timedelta(days=settings.refresh_token_days), extra)


def create_token_pair(subject: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = get_settings()
    return {
        "access_token": create_access_token(subject, extra),
        "refresh_token": create_refresh_token(subject, extra),
        "expires_in": settings.access_token_minutes * 60,
        "token_type": "bearer",
    }


def decode_token(token: str, expected_type: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise ValueError("Invalid token") from exc
    if expected_type and payload.get("type") != expected_type:
        raise ValueError("Invalid token type")
    return payload


def decode_access_token(token: str) -> dict[str, Any]:
    return decode_token(token, "access")


def decode_refresh_token(token: str) -> dict[str, Any]:
    return decode_token(token, "refresh")
