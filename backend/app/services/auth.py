from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.firebase import verify_firebase_token
from app.core.security import create_token_pair, decode_refresh_token, hash_password, verify_password
from app.models.domain import User
from app.repositories import ActivityRepository, UserRepository
from app.schemas import FirebaseLoginRequest, LoginRequest, RefreshRequest, RegisterRequest


class AuthError(ValueError):
    pass


class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.users = UserRepository(db)
        self.activity = ActivityRepository(db)

    def _tokens_for(self, user: User) -> dict:
        return create_token_pair(str(user.id), {"email": user.email, "role": user.role.value})

    def register(self, payload: RegisterRequest) -> dict:
        if self.users.get_by_email(payload.email):
            raise AuthError("Email is already registered")
        user = self.users.create(payload.email, hash_password(payload.password), payload.name)
        self.activity.audit("auth.register", actor_user_id=user.id, resource_type="user", resource_id=str(user.id))
        self.db.commit()
        return self._tokens_for(user)

    def login(self, payload: LoginRequest) -> dict:
        user = self.users.get_by_email(payload.email)
        if not user or not user.password_hash or not verify_password(payload.password, user.password_hash):
            raise AuthError("Invalid email or password")
        if not user.is_active:
            raise AuthError("Account is disabled")
        user.last_login_at = datetime.now(timezone.utc)
        self.activity.audit("auth.login", actor_user_id=user.id, resource_type="user", resource_id=str(user.id))
        self.db.commit()
        return self._tokens_for(user)

    def google_login(self, payload: FirebaseLoginRequest) -> dict:
        decoded = verify_firebase_token(payload.id_token)
        email = decoded.get("email")
        uid = decoded["uid"]
        if not email:
            raise AuthError("Firebase account has no email")
        user = self.users.get_by_firebase_uid(uid) or self.users.get_by_email(email)
        if not user:
            user = self.users.create(email, None, decoded.get("name") or email.split("@")[0], firebase_uid=uid)
        else:
            user.firebase_uid = user.firebase_uid or uid
            user.is_verified = True
        user.last_login_at = datetime.now(timezone.utc)
        self.activity.audit("auth.google_login", actor_user_id=user.id, resource_type="user", resource_id=str(user.id))
        self.db.commit()
        return self._tokens_for(user)

    def refresh(self, payload: RefreshRequest) -> dict:
        decoded = decode_refresh_token(payload.refresh_token)
        user = self.users.get_by_id(int(decoded["sub"]))
        if not user or not user.is_active:
            raise AuthError("Invalid refresh token")
        return self._tokens_for(user)

    def forgot_password(self, email: str) -> None:
        user = self.users.get_by_email(email)
        if user:
            self.activity.audit("auth.forgot_password", actor_user_id=user.id, resource_type="user", resource_id=str(user.id))
            self.db.commit()

    def logout(self, user: User) -> None:
        self.activity.audit("auth.logout", actor_user_id=user.id, resource_type="user", resource_id=str(user.id))
        self.db.commit()
