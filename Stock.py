import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import customtkinter as ctk
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from threading import Thread

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("900x700")
app.title("AI Stock Price Predictor")

scaler = MinMaxScaler(feature_range=(0, 1))
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
data = None

def fetch_and_train(symbol):
    global data, model, scaler
    stock_symbol = symbol.upper()
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    if stock_data.empty:
        result_label.configure(text="Invalid Stock Symbol! Try Again.", text_color="red")
        return
    result_label.configure(text=f"Training Model for {stock_symbol}...", text_color="yellow")
    data = stock_data[['Close']].values
    scaled_data = scaler.fit_transform(data)
    X_train, y_train = [], []
    for i in range(60, len(scaled_data) - 1):
        X_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    model.fit(X_train, y_train)
    result_label.configure(text=f"Model Trained for {stock_symbol}", text_color="green")

def predict_price():
    if model is None or data is None:
        result_label.configure(text="Train the model first!", text_color="red")
        return
    last_60_days = data[-60:].reshape(-1, 1)
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = [last_60_days_scaled.flatten()]
    X_test = np.array(X_test)
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform([[predicted_price[0]]])[0][0]
    actual_price = data[-1][0]
    result_label.configure(
        text=f"Predicted Price: ${predicted_price:.2f}\nActual Price: ${actual_price:.2f}",
        text_color="cyan"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(data[-100:], label="Actual Price", color="green")
    plt.axhline(y=predicted_price, color='red', linestyle="dashed", label="Predicted Price")
    plt.xlabel("Days (Last 100 Days)")  # Label for X-axis
    plt.ylabel("Stock Price (in USD)")  # Label for Y-axis
    plt.legend()
    plt.title("Stock Price Prediction")
    plt.grid()
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

app.mainloop()