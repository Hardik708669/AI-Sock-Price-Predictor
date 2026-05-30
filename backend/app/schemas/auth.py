from pydantic import BaseModel, EmailStr, Field

from app.schemas.common import Role


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenResponse(TokenPair):
    pass


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str


class FirebaseLoginRequest(BaseModel):
    id_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class CurrentUser(BaseModel):
    id: int
    email: EmailStr
    role: Role
    is_active: bool
