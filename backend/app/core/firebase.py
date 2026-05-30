import firebase_admin
from firebase_admin import auth, credentials

from app.core.config import get_settings


def initialize_firebase() -> None:
    settings = get_settings()
    if firebase_admin._apps:
        return
    if not settings.firebase_project_id:
        return
    if settings.firebase_client_email and settings.firebase_private_key:
        private_key = settings.firebase_private_key.replace("\\n", "\n")
        cred = credentials.Certificate(
            {
                "type": "service_account",
                "project_id": settings.firebase_project_id,
                "client_email": settings.firebase_client_email,
                "private_key": private_key,
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )
        firebase_admin.initialize_app(cred)
    else:
        firebase_admin.initialize_app(options={"projectId": settings.firebase_project_id})


def verify_firebase_token(id_token: str) -> dict:
    initialize_firebase()
    if not firebase_admin._apps:
        raise ValueError("Firebase is not configured")
    return auth.verify_id_token(id_token)
