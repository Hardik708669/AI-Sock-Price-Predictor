from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database import get_db
from app.models.domain import User
from app.schemas import FirebaseLoginRequest, ForgotPasswordRequest, LoginRequest, RefreshRequest, RegisterRequest, ResponseMessage, TokenResponse
from app.services.auth import AuthError, AuthService

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> dict:
    try:
        return AuthService(db).register(payload)
    except AuthError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> dict:
    try:
        return AuthService(db).login(payload)
    except AuthError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error


@router.post("/google", response_model=TokenResponse)
@router.post("/firebase", response_model=TokenResponse)
def google_login(payload: FirebaseLoginRequest, db: Session = Depends(get_db)) -> dict:
    try:
        return AuthService(db).google_login(payload)
    except Exception as error:
        raise HTTPException(status_code=401, detail=str(error)) from error


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)) -> dict:
    try:
        return AuthService(db).refresh(payload)
    except Exception as error:
        raise HTTPException(status_code=401, detail="Invalid refresh token") from error


@router.post("/forgot-password", response_model=ResponseMessage)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)) -> ResponseMessage:
    AuthService(db).forgot_password(payload.email)
    return ResponseMessage(message="If the email exists, password reset instructions will be sent.")


@router.post("/logout", response_model=ResponseMessage)
def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ResponseMessage:
    AuthService(db).logout(current_user)
    return ResponseMessage(message="Logged out successfully.")
