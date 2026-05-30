# StockVision AI

StockVision AI is a full-stack AI-powered stock intelligence platform built from the original Python stock predictor.

It includes a React/TypeScript frontend, FastAPI backend, live stock data, AI prediction APIs, portfolio tools, risk analytics, sentiment intelligence, alerts, and an AI stock assistant.

## Tech Stack

### Frontend
- React.js
- TypeScript
- Vite
- TailwindCSS
- Framer Motion
- React Query
- Zustand
- Recharts
- ApexCharts
- Firebase Authentication wiring

### Backend
- FastAPI
- JWT authentication
- Firebase token verification
- Rate limiting
- Service-layer architecture
- PostgreSQL-ready schema

### AI/ML
- Scikit-Learn Linear Regression
- Random Forest
- XGBoost
- Explainable AI feature importance
- Risk analytics
- Sentiment analysis

## Features

- Premium landing page
- Login, register, forgot password, and Google login UI
- AI dashboard with portfolio value, P/L, confidence score, sentiment, watchlist, news, and predictions
- Stock analysis for symbols like `AAPL`, `TSLA`, `MSFT`, `NVDA`, `RELIANCE.NS`, and `TCS.NS`
- Candlestick, line, area, and volume charts
- RSI, MACD, EMA, SMA, Bollinger Bands, and VWAP indicator toggles
- Multi-model prediction comparison
- AI BUY/SELL/HOLD recommendation engine
- Portfolio management with health score, risk score, allocation, and rebalance actions
- Market heatmap
- Alert API
- AI stock assistant
- No-Docker Windows run scripts

## Folder Structure

```txt
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── services/
│   │   ├── main.py
│   │   └── schemas.py
│   └── requirements-local.txt
├── database/
│   └── schema.sql
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── package-lock.json
│   ├── tailwind.config.ts
│   └── vercel.json
├── Stock.py
├── run-backend.ps1
├── run-frontend.ps1
├── RUN_WITHOUT_DOCKER.md
└── README.md
```

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

More details are in `RUN_WITHOUT_DOCKER.md`.

## API Routes

- `GET /api/v1/health`
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/firebase`
- `GET /api/v1/stocks/search?q=AAPL`
- `GET /api/v1/stocks/{symbol}/overview`
- `GET /api/v1/stocks/{symbol}/candles`
- `GET /api/v1/stocks/{symbol}/technicals`
- `GET /api/v1/predictions/{symbol}`
- `GET /api/v1/sentiment/{symbol}`
- `GET /api/v1/risk/{symbol}`
- `POST /api/v1/portfolio/analyze`
- `GET /api/v1/watchlist`
- `POST /api/v1/alerts`
- `GET /api/v1/heatmap`
- `GET /api/v1/dashboard`
- `POST /api/v1/assistant`

## Legacy Desktop App

`Stock.py` is kept as the legacy desktop version. The main project is now the full-stack StockVision AI web app.

## Disclaimer

This project is educational. It is not financial advice.
