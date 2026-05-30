# StockVision AI

StockVision AI transforms the original Python desktop stock predictor into a modern full-stack AI-powered stock intelligence platform.

It is built to look and feel like a startup-grade fintech SaaS product with a React/TypeScript frontend, FastAPI backend, PostgreSQL schema, Firebase authentication flow, multi-model prediction engine, portfolio tools, risk analytics, sentiment intelligence, alerts, and an AI stock assistant.

## Tech Stack

### Frontend
- React.js
- TypeScript
- Vite
- TailwindCSS
- ShadCN-style reusable UI components
- Framer Motion
- React Query
- Zustand
- Recharts
- ApexCharts
- Firebase Authentication

### Backend
- FastAPI
- PostgreSQL
- SQLAlchemy-ready architecture
- JWT authentication
- Firebase token verification
- Rate limiting
- Service layer architecture
- Repository-ready database design

### AI/ML
- Scikit-Learn Linear Regression
- Random Forest
- XGBoost
- Prophet-style forecasting path
- TensorFlow LSTM-ready forecasting path
- Explainable AI feature importance
- Risk analytics
- Sentiment analysis

## Product Features

- Premium landing page with animated stock chart background
- Login, register, forgot password, and Google login flow
- Main dashboard with portfolio value, daily P/L, confidence score, market sentiment, watchlist, news feed, and prediction center
- Customizable dashboard widgets
- Stock search for symbols such as `AAPL`, `TSLA`, `MSFT`, `NVDA`, `RELIANCE.NS`, and `TCS.NS`
- Company overview, live price, market cap, volume, PE ratio, EPS, and dividend yield
- TradingView-style charting with candlestick, line, area, and volume modes
- Timeframes and indicator toggles for RSI, MACD, EMA, SMA, Bollinger Bands, and VWAP
- Multi-model prediction comparison
- AI BUY/SELL/HOLD recommendation engine
- News sentiment analysis with bullish, bearish, and neutral classification
- AI stock assistant
- Portfolio management with holdings, allocation, P/L, health score, risk score, and rebalance actions
- Watchlist API
- Market heatmap
- Risk engine for volatility, Sharpe ratio, maximum drawdown, and beta
- Prediction history database schema
- Alert system API
- PostgreSQL schema for users, stocks, watchlists, portfolios, transactions, predictions, news, and alerts
- Docker setup
- Deployment guide for Vercel, Render, and Supabase

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
│   ├── Dockerfile
│   ├── render.yaml
│   └── requirements.txt
├── database/
│   └── schema.sql
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── lib/
│   │   ├── store/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tailwind.config.ts
│   └── vercel.json
├── Stock.py
├── docker-compose.yml
├── DEPLOYMENT.md
└── README.md
```

## Run Locally

Start backend dependencies and API:

```sh
docker compose up --build
```

Start frontend:

```sh
cd frontend
npm install
npm run dev
```

Open:

```txt
http://localhost:5173
```

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

The original `Stock.py` CustomTkinter app is still included as a legacy desktop version. The new resume-ready product is the full-stack `StockVision AI` platform.

## Disclaimer

This project is educational. It is not financial advice.
