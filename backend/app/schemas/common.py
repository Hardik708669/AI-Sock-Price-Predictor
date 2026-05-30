from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Role(StrEnum):
    user = "user"
    premium = "premium"
    admin = "admin"


class ResponseMessage(BaseModel):
    message: str


class PaginatedResponse(BaseModel):
    items: list
    total: int
    limit: int
    offset: int


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class AuditLogRead(ORMModel):
    id: int
    actor_user_id: int | None
    action: str
    resource_type: str | None
    resource_id: str | None
    ip_address: str | None
    user_agent: str | None
    created_at: datetime
