CREATE TYPE user_role AS ENUM ('user', 'premium', 'admin');
CREATE TYPE transaction_type AS ENUM ('buy', 'sell', 'dividend', 'deposit', 'withdrawal');

CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(320) NOT NULL UNIQUE,
    password_hash VARCHAR(255),
    firebase_uid VARCHAR(128) UNIQUE,
    role user_role NOT NULL DEFAULT 'user',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE profiles (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    full_name VARCHAR(120),
    avatar_url VARCHAR(500),
    phone VARCHAR(40),
    country VARCHAR(80),
    timezone VARCHAR(80) DEFAULT 'UTC',
    risk_preference VARCHAR(32) DEFAULT 'balanced',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE stocks (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(24) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    exchange VARCHAR(64),
    sector VARCHAR(120),
    industry VARCHAR(120),
    currency VARCHAR(8) NOT NULL DEFAULT 'USD',
    country VARCHAR(80),
    metadata_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE watchlists (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(120) NOT NULL DEFAULT 'Default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, name)
);

CREATE TABLE watchlist_items (
    id BIGSERIAL PRIMARY KEY,
    watchlist_id BIGINT NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
    stock_id BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (watchlist_id, stock_id)
);

CREATE TABLE portfolios (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(120) NOT NULL DEFAULT 'Primary Portfolio',
    base_currency VARCHAR(8) NOT NULL DEFAULT 'USD',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE portfolio_holdings (
    id BIGSERIAL PRIMARY KEY,
    portfolio_id BIGINT NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    stock_id BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    quantity DOUBLE PRECISION NOT NULL CHECK (quantity >= 0),
    average_price DOUBLE PRECISION NOT NULL CHECK (average_price >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (portfolio_id, stock_id)
);

CREATE TABLE transactions (
    id BIGSERIAL PRIMARY KEY,
    portfolio_id BIGINT NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    stock_id BIGINT REFERENCES stocks(id) ON DELETE SET NULL,
    type transaction_type NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    fees DOUBLE PRECISION NOT NULL DEFAULT 0,
    executed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE prediction_models (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(120) NOT NULL UNIQUE,
    version VARCHAR(40) NOT NULL DEFAULT '1.0.0',
    metrics JSONB NOT NULL DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
    stock_id BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    model_id BIGINT REFERENCES prediction_models(id) ON DELETE SET NULL,
    horizon_days INTEGER NOT NULL DEFAULT 30,
    predicted_price DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    recommendation VARCHAR(16) NOT NULL,
    explanation TEXT NOT NULL,
    features JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE market_news (
    id BIGSERIAL PRIMARY KEY,
    stock_id BIGINT REFERENCES stocks(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    source VARCHAR(120) NOT NULL,
    url VARCHAR(800),
    summary TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE sentiment_analysis (
    id BIGSERIAL PRIMARY KEY,
    stock_id BIGINT REFERENCES stocks(id) ON DELETE CASCADE,
    news_id BIGINT UNIQUE REFERENCES market_news(id) ON DELETE CASCADE,
    sentiment VARCHAR(24) NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(40) NOT NULL DEFAULT 'lexicon-v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE risk_reports (
    id BIGSERIAL PRIMARY KEY,
    stock_id BIGINT REFERENCES stocks(id) ON DELETE CASCADE,
    user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
    volatility DOUBLE PRECISION NOT NULL,
    beta DOUBLE PRECISION NOT NULL,
    sharpe_ratio DOUBLE PRECISION NOT NULL,
    max_drawdown DOUBLE PRECISION NOT NULL,
    value_at_risk DOUBLE PRECISION NOT NULL,
    report JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE alerts (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    stock_id BIGINT REFERENCES stocks(id) ON DELETE CASCADE,
    metric VARCHAR(64) NOT NULL,
    operator VARCHAR(4) NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    channel VARCHAR(32) NOT NULL DEFAULT 'email',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE alert_logs (
    id BIGSERIAL PRIMARY KEY,
    alert_id BIGINT NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    triggered_value DOUBLE PRECISION NOT NULL,
    message TEXT NOT NULL,
    delivered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chat_sessions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(180) NOT NULL DEFAULT 'New Copilot Chat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    context JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE audit_logs (
    id BIGSERIAL PRIMARY KEY,
    actor_user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(120) NOT NULL,
    resource_type VARCHAR(120),
    resource_id VARCHAR(120),
    ip_address VARCHAR(80),
    user_agent VARCHAR(500),
    metadata_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE notifications (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(180) NOT NULL,
    body TEXT NOT NULL,
    channel VARCHAR(32) NOT NULL DEFAULT 'in_app',
    is_read BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ix_users_email ON users(email);
CREATE INDEX ix_users_firebase_uid ON users(firebase_uid);
CREATE INDEX ix_stocks_ticker ON stocks(ticker);
CREATE INDEX ix_transactions_portfolio_executed ON transactions(portfolio_id, executed_at);
CREATE INDEX ix_predictions_stock_created ON predictions(stock_id, created_at);
CREATE INDEX ix_market_news_stock_published ON market_news(stock_id, published_at);
CREATE INDEX ix_alerts_user_active ON alerts(user_id, is_active);
CREATE INDEX ix_chat_messages_session_created ON chat_messages(session_id, created_at);
CREATE INDEX ix_audit_logs_actor_created ON audit_logs(actor_user_id, created_at);
