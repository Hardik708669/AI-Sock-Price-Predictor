from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database import get_db
from app.models.domain import User
from app.repositories import ActivityRepository, AlertRepository, StockRepository
from app.schemas import AlertRequest

router = APIRouter()


@router.post("", status_code=status.HTTP_201_CREATED)
def create_alert(payload: AlertRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    stock = StockRepository(db).get_or_create(payload.symbol)
    alert = AlertRepository(db).create(current_user.id, stock, payload.metric, payload.operator, payload.threshold, payload.channel)
    ActivityRepository(db).audit("alerts.create", current_user.id, "alert", str(alert.id))
    db.commit()
    return {"id": alert.id, "message": f"Alert created: {stock.ticker} {payload.metric} {payload.operator} {payload.threshold}", "active": alert.is_active}


@router.get("")
def list_alerts(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    alerts = AlertRepository(db).list_for_user(current_user.id)
    return {"items": [{"id": alert.id, "metric": alert.metric, "operator": alert.operator, "threshold": alert.threshold, "channel": alert.channel, "is_active": alert.is_active} for alert in alerts]}
