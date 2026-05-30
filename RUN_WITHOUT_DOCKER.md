# Run StockVision AI Without Docker

StockVision AI is configured to run locally without Docker.

## Prerequisites

- Python 3.13 or newer
- Node.js 18 or newer
- Internet connection for live stock data from `yfinance`

PostgreSQL and Redis are optional for local development. The backend defaults to a local SQLite database file so it can start immediately.

## Terminal 1: Start Backend

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-backend.ps1
```

Backend URL:

```txt
http://127.0.0.1:8000
```

Health check:

```txt
http://127.0.0.1:8000/api/v1/health
```

Swagger docs:

```txt
http://127.0.0.1:8000/docs
```

## Terminal 2: Start Frontend

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-frontend.ps1
```

Frontend URL:

```txt
http://127.0.0.1:5173
```

## Manual Backend Commands

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Optional PostgreSQL

For a production-like local database, install PostgreSQL normally on your machine, create a database, then set:

```powershell
$env:DATABASE_URL="postgresql+psycopg://stockvision:stockvision@localhost:5432/stockvision"
```

You can apply the schema from:

```txt
database/schema.sql
```

## Optional Redis And Celery

Redis is only needed if you want to run cache-backed features or Celery workers locally.

Start a worker after Redis is running:

```powershell
cd backend
.\.venv\Scripts\activate
celery -A app.jobs.worker.celery_app worker --loglevel=INFO
```

## Optional Advanced ML Packages

The AI engine supports TensorFlow LSTM, Prophet, TA-Lib, SHAP, and Transformers. The backend is designed to run even if native-heavy packages are missing.

Core ML packages are in:

```txt
backend/requirements-local.txt
```

Native optional packages are in:

```txt
backend/requirements-ml-optional.txt
```

Install them only if your machine has compatible wheels/tooling:

```powershell
cd backend
.\.venv\Scripts\activate
python -m pip install -r requirements-ml-optional.txt
```

To enable FinBERT transformer sentiment after installing Transformers/PyTorch, set:

```powershell
$env:ENABLE_TRANSFORMER_SENTIMENT="true"
```

## Notes

- No Docker is required.
- Local development defaults to `backend/stockvision_local.db`.
- PostgreSQL remains supported through `DATABASE_URL`.
- Firebase Google login needs real Firebase environment values.
- Stock endpoints use live `yfinance` data, so they need internet access.
