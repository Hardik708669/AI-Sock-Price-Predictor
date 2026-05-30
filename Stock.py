import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import customtkinter as ctk
import logging
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from threading import Thread

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

app = ctk.CTk()
app.geometry("980x780")
app.title("AI Stock Price Predictor")

scaler = MinMaxScaler(feature_range=(0, 1))
model = None
data = None
trained = False
DOWNLOAD_RETRY_DELAYS = (0, 15, 45)

PORTFOLIO_TEMPLATES = {
    10000: {
        "risk": "Moderate",
        "risk_score": 46,
        "expected_return": "9-12% yearly",
        "allocation": {
            "Nifty 50 Index Fund": 45,
            "Large Cap Quality Stocks": 20,
            "Flexi Cap Mutual Fund": 15,
            "Gold ETF": 10,
            "Liquid Fund": 10,
        },
        "sectors": {
            "Financial Services": 25,
            "Technology": 18,
            "Consumer": 16,
            "Healthcare": 12,
            "Energy": 9,
            "Gold/Debt": 20,
        },
        "why": [
            "Keeps the base diversified because a smaller capital amount should avoid concentrated bets.",
            "Uses liquid and gold exposure to soften market dips.",
            "Adds flexi-cap exposure so the portfolio can still participate in mid-cap growth.",
        ],
    },
    50000: {
        "risk": "Balanced Growth",
        "risk_score": 58,
        "expected_return": "11-14% yearly",
        "allocation": {
            "Nifty 50 Index Fund": 35,
            "Large Cap Quality Stocks": 20,
            "Mid Cap Index Fund": 15,
            "Banking & Financial ETF": 10,
            "Gold ETF": 10,
            "Liquid Fund": 10,
        },
        "sectors": {
            "Financial Services": 28,
            "Technology": 17,
            "Consumer": 14,
            "Industrials": 12,
            "Healthcare": 9,
            "Gold/Debt": 20,
        },
        "why": [
            "Balances core index exposure with selective growth from mid-cap and banking themes.",
            "Keeps 20% in gold and liquid funds to reduce volatility and support future rebalancing.",
            "Avoids overloading one sector by spreading equity exposure across broad market buckets.",
        ],
    },
    100000: {
        "risk": "Growth",
        "risk_score": 67,
        "expected_return": "12-16% yearly",
        "allocation": {
            "Nifty 50 Index Fund": 30,
            "Large Cap Quality Stocks": 20,
            "Mid Cap Index Fund": 18,
            "Small Cap Fund": 10,
            "International ETF": 7,
            "Gold ETF": 8,
            "Liquid Fund": 7,
        },
        "sectors": {
            "Financial Services": 24,
            "Technology": 18,
            "Consumer": 13,
            "Industrials": 13,
            "Healthcare": 10,
            "International": 7,
            "Gold/Debt": 15,
        },
        "why": [
            "Higher capital can support more diversification, including international and small-cap exposure.",
            "The portfolio still keeps defensive assets so it is not fully dependent on equity momentum.",
            "Core index funds remain the anchor while satellite funds add growth potential.",
        ],
    },
}


def format_inr(amount):
    rounded_amount = str(int(round(amount)))
    if len(rounded_amount) <= 3:
        formatted_amount = rounded_amount
    else:
        last_three = rounded_amount[-3:]
        remaining = rounded_amount[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted_amount = ",".join(groups + [last_three])
    return f"₹{formatted_amount}"


def build_investment_plan(amount):
    template = PORTFOLIO_TEMPLATES[amount]
    allocation_rows = []
    rebalance_rows = []

    for index, (asset, percent) in enumerate(template["allocation"].items()):
        target_value = amount * percent / 100
        drift = [1.08, 0.94, 1.03, 0.90, 1.12, 0.96, 1.05][index % 7]
        current_value = target_value * drift
        trade_value = target_value - current_value
        action = "Buy" if trade_value > 0 else "Trim"

        allocation_rows.append(
            f"{asset}: {percent}% ({format_inr(target_value)})"
        )
        rebalance_rows.append(
            f"{action} {format_inr(abs(trade_value))} in {asset}"
        )

    sector_rows = [
        f"{sector}: {percent}%"
        for sector, percent in template["sectors"].items()
    ]

    return {
        "risk": template["risk"],
        "risk_score": template["risk_score"],
        "expected_return": template["expected_return"],
        "allocation_rows": allocation_rows,
        "sector_rows": sector_rows,
        "why": template["why"],
        "rebalance_rows": rebalance_rows,
    }


def update_result(message, color="white"):
    app.after(0, lambda: result_label.configure(text=message, text_color=color))


def set_button_state(train_state, predict_state):
    def update():
        train_button.configure(state=train_state)
        predict_button.configure(state=predict_state)
    app.after(0, update)


def download_stock_data(stock_symbol, start_date, end_date):
    last_error = None

    for attempt, delay in enumerate(DOWNLOAD_RETRY_DELAYS, start=1):
        if delay:
            update_result(
                f"Yahoo is busy/rate limiting. Retrying in {delay} seconds...",
                "yellow",
            )
            time.sleep(delay)

        update_result(
            f"Downloading data for {stock_symbol}... attempt {attempt}/{len(DOWNLOAD_RETRY_DELAYS)}",
            "yellow",
        )

        try:
            stock_data = yf.download(
                stock_symbol,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                threads=False,
                multi_level_index=False,
            )
        except Exception as error:
            last_error = error
            continue

        if not stock_data.empty:
            return stock_data, None

    update_result("Yahoo is still rate limiting. Trying Stooq fallback...", "yellow")
    try:
        stock_data = download_from_stooq(stock_symbol, start_date, end_date)
    except Exception as error:
        return None, last_error or error

    if stock_data is not None and not stock_data.empty:
        return stock_data, None

    return None, last_error


def download_from_stooq(stock_symbol, start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
    end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")

    stooq_symbols = [stock_symbol.lower()]
    if "." not in stock_symbol:
        stooq_symbols.insert(0, f"{stock_symbol.lower()}.us")

    for stooq_symbol in stooq_symbols:
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&d1={start}&d2={end}&i=d"
        stock_data = pd.read_csv(url)

        if stock_data.empty:
            continue

        stock_data.columns = [column.strip() for column in stock_data.columns]
        if "Date" not in stock_data.columns or "Close" not in stock_data.columns:
            continue

        stock_data["Date"] = pd.to_datetime(stock_data["Date"], errors="coerce")
        stock_data["Close"] = pd.to_numeric(stock_data["Close"], errors="coerce")
        stock_data = stock_data.dropna(subset=["Date", "Close"])

        if stock_data.empty:
            continue

        return stock_data.set_index("Date").sort_index()

    return pd.DataFrame()


def fetch_and_train(symbol):
    global data, model, scaler, trained
    stock_symbol = symbol.strip().upper()
    if not stock_symbol:
        update_result("Enter a stock symbol.", "red")
        set_button_state("normal", "disabled")
        return
    set_button_state("disabled", "disabled")
    update_result(f"Downloading data for {stock_symbol}...", "yellow")

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    stock_data, download_error = download_stock_data(stock_symbol, start_date, end_date)

    if download_error is not None:
        update_result(f"Could not download data:\n{download_error}", "red")
        set_button_state("normal", "disabled")
        return

    if stock_data is None or stock_data.empty:
        update_result(
            "No price data downloaded. Yahoo may be rate limiting you; wait a few minutes and try again.",
            "red",
        )
        set_button_state("normal", "disabled")
        return

    if len(stock_data) < 70:
        update_result(
            "Not enough historical data. Try another symbol.",
            "red",
        )
        set_button_state("normal", "disabled")
        return

    update_result(f"Training model for {stock_symbol}...", "yellow")

    close_data = stock_data[['Close']].values
    scaled_data = scaler.fit_transform(close_data)

    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    if not X_train:
        update_result("Not enough data to train the model.", "red")
        set_button_state("normal", "disabled")
        return

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    data = close_data
    trained = True
    update_result(f"Model trained for {stock_symbol}", "green")
    set_button_state("normal", "normal")


def predict_price():
    if not trained or model is None or data is None:
        update_result("Train the model first!", "red")
        return

    if len(data) < 60:
        update_result("Not enough data to predict.", "red")
        return

    last_60_days = data[-60:]
    last_60_scaled = scaler.transform(last_60_days)
    X_test = last_60_scaled.flatten().reshape(1, -1)

    predicted_scaled = model.predict(X_test)[0]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]
    last_close = float(data[-1][0])

    update_result(
        f"Predicted Next Close: ${predicted_price:.2f}\nLast Close: ${last_close:.2f}",
        "cyan",
    )

    plt.figure(figsize=(10, 5))
    historical_prices = data[-100:].flatten()
    extended_prices = np.append(historical_prices, predicted_price)
    x_values = list(range(len(extended_prices)))

    plt.plot(x_values[:-1], extended_prices[:-1], label="Last Close Price", color="green")
    plt.scatter(x_values[-1], extended_prices[-1], color="red", label="Predicted Next Close", zorder=5)
    plt.plot(
        [x_values[-2], x_values[-1]],
        [extended_prices[-2], extended_prices[-1]],
        linestyle='dashed',
        color='red',
    )
    plt.xlabel("Recent Time")
    plt.ylabel("Stock Price (USD)")
    plt.title("Stock Price Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def update_copilot(amount=None):
    if amount is not None:
        investment_entry.delete(0, "end")
        investment_entry.insert(0, str(amount))

    raw_amount = (
        investment_entry.get()
        .replace(",", "")
        .replace("₹", "")
        .replace("Rs.", "")
        .replace("rs.", "")
        .strip()
    )
    if not raw_amount:
        copilot_summary.configure(
            text="Choose an amount to generate an AI-style allocation plan.",
            text_color="white",
        )
        return

    try:
        investment_amount = int(float(raw_amount))
    except ValueError:
        copilot_summary.configure(text="Enter a valid investment amount.", text_color="red")
        return

    if investment_amount not in PORTFOLIO_TEMPLATES:
        copilot_summary.configure(
            text="Enter exactly ₹10,000, ₹50,000, or ₹1,00,000.",
            text_color="red",
        )
        allocation_text.configure(text="")
        why_text.configure(text="")
        sector_text.configure(text="")
        rebalance_text.configure(text="")
        return

    plan = build_investment_plan(investment_amount)
    copilot_summary.configure(
        text=(
            f"AI Suggested Portfolio for {format_inr(investment_amount)}\n"
            f"Risk Profile: {plan['risk']} | Expected Return: {plan['expected_return']}\n"
            f"Expected Risk: {plan['risk_score']}/100"
        ),
        text_color="#00FFC6",
    )
    allocation_text.configure(text="\n".join(plan["allocation_rows"]))
    why_text.configure(text="\n".join(f"- {reason}" for reason in plan["why"]))
    sector_text.configure(text="\n".join(plan["sector_rows"]))
    rebalance_text.configure(text="\n".join(plan["rebalance_rows"]))


frame = ctk.CTkFrame(app, width=800, height=600, corner_radius=25, fg_color="#101820")
frame.pack(pady=30, padx=30, fill="both", expand=True)

tab_view = ctk.CTkTabview(frame, fg_color="#101820", segmented_button_fg_color="#182A3A")
tab_view.pack(padx=20, pady=20, fill="both", expand=True)
predictor_tab = tab_view.add("Stock Predictor")
copilot_tab = tab_view.add("AI Investment Copilot")

title_label = ctk.CTkLabel(
    predictor_tab, text="AI Stock Price Predictor",
    font=("Arial", 28, "bold"),
    text_color="#00FFC6"
)
title_label.pack(pady=30)

entry = ctk.CTkEntry(
    predictor_tab, placeholder_text="Enter Stock Symbol (e.g., AAPL)",
    width=400, height=50, font=("Arial", 16),
    border_width=1, corner_radius=10
)
entry.pack(pady=15)

train_button = ctk.CTkButton(
    predictor_tab, text="Train Model",
    command=lambda: Thread(target=fetch_and_train, args=(entry.get(),)).start(),
    width=300, height=50, font=("Arial", 16),
    fg_color="#00B894", hover_color="#00FFC6", corner_radius=10
)
train_button.pack(pady=15)

predict_button = ctk.CTkButton(
    predictor_tab, text="Predict Price",
    command=predict_price,
    width=300, height=50, font=("Arial", 16),
    fg_color="#0984E3", hover_color="#00FFC6", corner_radius=10
)
predict_button.pack(pady=15)

result_label = ctk.CTkLabel(
    predictor_tab, text="", font=("Arial", 18),
    text_color="white"
)
result_label.pack(pady=20)

copilot_title = ctk.CTkLabel(
    copilot_tab,
    text="AI Investment Copilot",
    font=("Arial", 28, "bold"),
    text_color="#00FFC6",
)
copilot_title.pack(pady=(22, 8))

copilot_subtitle = ctk.CTkLabel(
    copilot_tab,
    text="Portfolio allocation, risk, sector spread, and automatic rebalancing for Indian investors.",
    font=("Arial", 14),
    text_color="#C8D6E5",
)
copilot_subtitle.pack(pady=(0, 16))

amount_frame = ctk.CTkFrame(copilot_tab, fg_color="#132536", corner_radius=12)
amount_frame.pack(padx=20, pady=8, fill="x")
amount_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

investment_entry = ctk.CTkEntry(
    amount_frame,
    placeholder_text="Enter ₹10,000 / ₹50,000 / ₹1,00,000",
    width=220,
    height=42,
    font=("Arial", 15),
    border_width=1,
    corner_radius=8,
)
investment_entry.grid(row=0, column=0, padx=(14, 8), pady=14, sticky="ew")

generate_button = ctk.CTkButton(
    amount_frame,
    text="Generate Plan",
    command=update_copilot,
    width=128,
    height=42,
    font=("Arial", 14, "bold"),
    fg_color="#00B894",
    hover_color="#00FFC6",
    text_color="#08131D",
    corner_radius=8,
)
generate_button.grid(row=0, column=1, padx=8, pady=14, sticky="ew")

for button_index, quick_amount in enumerate((10000, 50000, 100000), start=2):
    amount_button = ctk.CTkButton(
        amount_frame,
        text=format_inr(quick_amount),
        command=lambda value=quick_amount: update_copilot(value),
        width=110,
        height=42,
        font=("Arial", 14, "bold"),
        fg_color="#0984E3",
        hover_color="#00B894",
        corner_radius=8,
    )
    amount_button.grid(
        row=0,
        column=button_index,
        padx=(8, 14 if button_index == 4 else 8),
        pady=14,
        sticky="ew",
    )

copilot_summary = ctk.CTkLabel(
    copilot_tab,
    text="Choose an amount to generate an AI-style allocation plan.",
    font=("Arial", 17, "bold"),
    text_color="white",
    justify="left",
)
copilot_summary.pack(padx=28, pady=14, anchor="w")

insight_grid = ctk.CTkFrame(copilot_tab, fg_color="transparent")
insight_grid.pack(padx=20, pady=8, fill="both", expand=True)
insight_grid.grid_columnconfigure((0, 1), weight=1)
insight_grid.grid_rowconfigure((0, 1), weight=1)


def create_copilot_panel(parent, title, row, column):
    panel = ctk.CTkFrame(parent, fg_color="#132536", corner_radius=12)
    panel.grid(row=row, column=column, padx=10, pady=10, sticky="nsew")
    heading = ctk.CTkLabel(
        panel,
        text=title,
        font=("Arial", 16, "bold"),
        text_color="#00FFC6",
    )
    heading.pack(padx=14, pady=(12, 6), anchor="w")
    body = ctk.CTkLabel(
        panel,
        text="",
        font=("Arial", 13),
        text_color="#EAF2F8",
        justify="left",
        wraplength=390,
    )
    body.pack(padx=14, pady=(0, 14), anchor="nw", fill="both", expand=True)
    return body


allocation_text = create_copilot_panel(insight_grid, "Suggested Allocation", 0, 0)
why_text = create_copilot_panel(insight_grid, "Why This Portfolio", 0, 1)
sector_text = create_copilot_panel(insight_grid, "Sector Diversification", 1, 0)
rebalance_text = create_copilot_panel(insight_grid, "Auto Rebalance Actions", 1, 1)

disclaimer_label = ctk.CTkLabel(
    copilot_tab,
    text="Educational model output only. Review goals, tax rules, and risk capacity before investing.",
    font=("Arial", 12),
    text_color="#95A5A6",
)
disclaimer_label.pack(pady=(0, 12))

predict_button.configure(state="disabled")
app.mainloop()
