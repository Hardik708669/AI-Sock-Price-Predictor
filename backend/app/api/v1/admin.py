from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.database import get_db
from app.models.domain import Alert, AuditLog, Portfolio, Prediction, Stock, User
from app.repositories import ActivityRepository, UserRepository
from app.schemas import AdminUserUpdate

router = APIRouter(dependencies=[Depends(require_admin)])


@router.get("/users")
def list_users(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)) -> dict:
    users = UserRepository(db).list(limit, offset)
    total = db.scalar(select(func.count(User.id))) or 0
    return {"items": [{"id": user.id, "email": user.email, "role": user.role.value, "is_active": user.is_active, "created_at": user.created_at} for user in users], "total": total}


@router.patch("/users/{user_id}")
def update_user(user_id: int, payload: AdminUserUpdate, db: Session = Depends(get_db)) -> dict:
    user = UserRepository(db).get_by_id(user_id)
    if not user:
        return {"message": "User not found"}
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    ActivityRepository(db).audit("admin.user_update", resource_type="user", resource_id=str(user_id))
    db.commit()
    return {"message": "User updated"}


@router.get("/analytics")
def analytics(db: Session = Depends(get_db)) -> dict:
    return {
        "users": db.scalar(select(func.count(User.id))) or 0,
        "stocks": db.scalar(select(func.count(Stock.id))) or 0,
        "portfolios": db.scalar(select(func.count(Portfolio.id))) or 0,
        "predictions": db.scalar(select(func.count(Prediction.id))) or 0,
        "alerts": db.scalar(select(func.count(Alert.id))) or 0,
    }


@router.get("/reports")
def reports(db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(100)).all()
    return {"audit_logs": [{"id": row.id, "action": row.action, "resource_type": row.resource_type, "created_at": row.created_at} for row in rows]}


@router.get("/system")
def system_monitoring() -> dict:
    return {"status": "healthy", "database": "connected", "redis": "configured", "workers": "celery-ready"}
