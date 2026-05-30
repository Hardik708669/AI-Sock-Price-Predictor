# Render Deployment Guide Without Docker

This project is configured for source-based Render deployment. No Dockerfile is required.

## Web Service

Create a Render Web Service from the repository.

Build command:

```bash
cd backend && pip install -r requirements-local.txt
```

Start command:

```bash
cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Background Worker

Create a Render Background Worker if you want Celery tasks.

Build command:

```bash
cd backend && pip install -r requirements-local.txt
```

Start command:

```bash
cd backend && celery -A app.jobs.worker.celery_app worker --loglevel=INFO
```

## Managed Services

Use Render managed PostgreSQL and Redis.

Set `DATABASE_URL` to the Render PostgreSQL internal URL using the SQLAlchemy driver format:

```txt
postgresql+psycopg://USER:PASSWORD@HOST:PORT/DATABASE
```

Set Redis variables:

```txt
REDIS_URL=<Render Redis URL>
CELERY_BROKER_URL=<Render Redis URL>/1
CELERY_RESULT_BACKEND=<Render Redis URL>/2
```

## Required Environment Variables

```txt
ENVIRONMENT=production
DATABASE_URL=postgresql+psycopg://...
REDIS_URL=redis://...
CELERY_BROKER_URL=redis://.../1
CELERY_RESULT_BACKEND=redis://.../2
JWT_SECRET=<long random secret>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_MINUTES=30
REFRESH_TOKEN_DAYS=14
ALLOWED_ORIGINS=https://your-frontend-domain.com
FRONTEND_URL=https://your-frontend-domain.com
FIREBASE_PROJECT_ID=<project>
FIREBASE_CLIENT_EMAIL=<service-account-email>
FIREBASE_PRIVATE_KEY=<service-account-private-key>
```

## Database

Run `database/schema.sql` once against the production PostgreSQL database, or add Alembic migrations from the SQLAlchemy models before launch.
