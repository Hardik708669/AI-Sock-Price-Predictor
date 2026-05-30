# Run StockVision AI Without Docker

You do not need Docker to run this project locally.

## Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer
- Internet connection for stock data from `yfinance`

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

## Terminal 2: Start Frontend

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-frontend.ps1
```

Frontend URL:

```txt
http://127.0.0.1:5173
```

## Manual Commands

If you do not want to use the scripts, run these commands manually.

Backend:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

## Notes

- This no-Docker mode runs the current API and UI without local PostgreSQL.
- The PostgreSQL schema is still included in `database/schema.sql` for Supabase or future deployment.
- Firebase Google login needs real Firebase values in `frontend/.env`; the rest of the app can be explored without it.
- Stock endpoints use live `yfinance` data, so they need internet access.
