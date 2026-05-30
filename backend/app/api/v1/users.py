from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database import get_db
from app.models.domain import User
from app.repositories import ActivityRepository
from app.schemas import ProfileUpdate, UserRead

router = APIRouter()


@router.get("/profile", response_model=UserRead)
def get_profile(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@router.put("/profile", response_model=UserRead)
def update_profile(payload: ProfileUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> User:
    profile = current_user.profile
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(profile, field, value)
    ActivityRepository(db).audit("users.profile_update", actor_user_id=current_user.id, resource_type="profile", resource_id=str(profile.id))
    db.commit()
    db.refresh(current_user)
    return current_user
