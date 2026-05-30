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
app.geometry("900x700")
app.title("AI Stock Price Predictor")

scaler = MinMaxScaler(feature_range=(0, 1))
model = None
data = None
trained = False
DOWNLOAD_RETRY_DELAYS = (0, 15, 45)


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

frame = ctk.CTkFrame(app, width=800, height=600, corner_radius=25, fg_color="#101820")
frame.pack(pady=30, padx=30, fill="both", expand=True)

title_label = ctk.CTkLabel(
    frame, text="AI Stock Price Predictor",
    font=("Arial", 28, "bold"),
    text_color="#00FFC6"
)
title_label.pack(pady=30)

entry = ctk.CTkEntry(
    frame, placeholder_text="Enter Stock Symbol (e.g., AAPL)",
    width=400, height=50, font=("Arial", 16),
    border_width=1, corner_radius=10
)
entry.pack(pady=15)

train_button = ctk.CTkButton(
    frame, text="Train Model",
    command=lambda: Thread(target=fetch_and_train, args=(entry.get(),)).start(),
    width=300, height=50, font=("Arial", 16),
    fg_color="#00B894", hover_color="#00FFC6", corner_radius=10
)
train_button.pack(pady=15)

predict_button = ctk.CTkButton(
    frame, text="Predict Price",
    command=predict_price,
    width=300, height=50, font=("Arial", 16),
    fg_color="#0984E3", hover_color="#00FFC6", corner_radius=10
)
predict_button.pack(pady=15)

result_label = ctk.CTkLabel(
    frame, text="", font=("Arial", 18),
    text_color="white"
)
result_label.pack(pady=20)

predict_button.configure(state="disabled")
app.mainloop()
