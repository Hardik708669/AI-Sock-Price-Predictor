# StockVision AI

StockVision AI is an AI-powered stock intelligence platform with a React frontend and FastAPI backend.

The project is configured to run **without Docker**. Local development starts with PowerShell scripts and a Python virtual environment.

## Tech Stack

### Frontend

- React
- TypeScript
- Vite
- Tailwind CSS
- Framer Motion
- React Query
- Zustand
- Recharts
- TradingView widget integration

### Backend

- FastAPI
- Python 3.13+
- SQLAlchemy 2.0
- Pydantic
- JWT access and refresh tokens
- Firebase Authentication support
- Role based access control
- Rate limiting
- Security headers
- WebSockets
- Celery-ready background jobs
- Redis-ready caching
- Advanced AI engine with prediction, technical analysis, sentiment, risk, optimizer, explainability, anomaly detection, forecasting, and copilot orchestration

### Database

- Local development default: SQLite file at `backend/stockvision_local.db`
- Production/local advanced option: PostgreSQL via `DATABASE_URL`
- PostgreSQL schema: `database/schema.sql`

## Run Without Docker

Start backend in Terminal 1:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-backend.ps1
```

Start frontend in Terminal 2:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-frontend.ps1
```

Open:

```txt
http://127.0.0.1:5173
```

API docs:

```txt
http://127.0.0.1:8000/docs
```

More details are in `RUN_WITHOUT_DOCKER.md`.

## Backend Structure

```txt
backend/
  app/
    api/
    core/
    database/
    jobs/
    middleware/
    ml/
    models/
    repositories/
    schemas/
    services/
    utils/
    websocket/
    main.py
  requirements-local.txt
```

## Main API Areas

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/logout`
- `POST /api/v1/auth/refresh`
- `POST /api/v1/auth/forgot-password`
- `GET /api/v1/users/profile`
- `PUT /api/v1/users/profile`
- `GET /api/v1/stocks/search`
- `GET /api/v1/stocks/{ticker}`
- `GET /api/v1/stocks/history`
- `GET /api/v1/stocks/live`
- `GET /api/v1/portfolio`
- `POST /api/v1/portfolio/add`
- `POST /api/v1/portfolio/remove`
- `GET /api/v1/watchlist`
- `POST /api/v1/watchlist/add`
- `DELETE /api/v1/watchlist/remove`
- `POST /api/v1/predict`
- `GET /api/v1/prediction/history`
- `GET /api/v1/news`
- `GET /api/v1/news/sentiment`
- `GET /api/v1/risk-analysis`
- `POST /api/v1/alerts`
- `GET /api/v1/alerts`
- `POST /api/v1/copilot/chat`
- `GET /api/v1/admin/analytics`
- `GET /api/v1/ai/intelligence/{ticker}`
- `GET /api/v1/ai/technical-analysis/{ticker}`
- `GET /api/v1/ai/sentiment/{ticker}`
- `GET /api/v1/ai/trend/{ticker}`
- `GET /api/v1/ai/risk/{ticker}`
- `GET /api/v1/ai/anomalies/{ticker}`
- `GET /api/v1/ai/forecast/{ticker}`
- `POST /api/v1/ai/portfolio/optimize`
- `POST /api/v1/ai/retrain`

## AI Engine

The production AI modules live in:

```txt
backend/app/ml/
```

Documentation:

```txt
backend/app/ml/README.md
```

Optional native packages like TensorFlow, Prophet, and TA-Lib are listed in:

```txt
backend/requirements-ml-optional.txt
```

## WebSockets

- `/ws/stocks`
- `/ws/portfolio`
- `/ws/alerts`

## Optional PostgreSQL

For production-like local development, install PostgreSQL directly on your machine and set:

```powershell
$env:DATABASE_URL="postgresql+psycopg://stockvision:stockvision@localhost:5432/stockvision"
```

Then apply:

```txt
database/schema.sql
```

## Legacy Desktop App

`Stock.py` is kept as the legacy desktop version. The main project is the StockVision AI web app.

## Disclaimer

This project is educational. It is not financial advice.
