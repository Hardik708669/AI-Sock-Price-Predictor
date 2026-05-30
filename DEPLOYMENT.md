# StockVision AI Deployment Guide

## Local Development

1. Copy environment files:

```sh
copy .env.example .env
copy frontend\.env.example frontend\.env
```

2. Start PostgreSQL, Redis, and the backend:

```sh
docker compose up --build
```

3. Start the frontend:

```sh
cd frontend
npm install
npm run dev
```

4. Open:

```txt
http://localhost:5173
```

## Frontend on Vercel

1. Import the repository in Vercel.
2. Set root directory to `frontend`.
3. Add environment variables:
   - `VITE_API_URL`
   - `VITE_FIREBASE_API_KEY`
   - `VITE_FIREBASE_AUTH_DOMAIN`
   - `VITE_FIREBASE_PROJECT_ID`
   - `VITE_FIREBASE_APP_ID`
4. Deploy with the included `vercel.json`.

## Backend on Render

1. Create a Render Web Service.
2. Set root directory to `backend`.
3. Use Docker deployment.
4. Add environment variables:
   - `DATABASE_URL`
   - `JWT_SECRET`
   - `FRONTEND_URL`
   - Firebase service account values
5. Health check path:

```txt
/api/v1/health
```

## Database on Supabase

1. Create a Supabase project.
2. Open SQL editor.
3. Run `database/schema.sql`.
4. Copy the PostgreSQL connection string into `DATABASE_URL`.

## Firebase Authentication

1. Create a Firebase project.
2. Enable Email/Password and Google providers.
3. Add frontend web app credentials to `frontend/.env`.
4. Add Admin SDK credentials to backend environment variables.

## Production Notes

- Use a long random `JWT_SECRET`.
- Restrict CORS to your Vercel domain.
- Configure rate limits per subscription tier.
- Move news and prediction jobs to background workers for high traffic.
- Add SHAP artifact caching for expensive explainability calls.
