# 📈 AI Stock Price Prediction

## 🚀 Overview
This AI-powered stock price predictor fetches historical stock data and predicts the next closing price using **XGBoost**. The project includes a **modern GUI (CustomTkinter)** and a visualization chart showing the latest closing prices plus the predicted next-day close.

---
## ✨ Features
✅ **Real-time Stock Data** – Fetches stock prices using `yfinance`
✅ **AI-Powered Prediction** – Uses `XGBoost` for next-day close prediction
✅ **User-Friendly GUI** – Built with `CustomTkinter` for a clean interface
✅ **Graphical Visualization** – Shows recent close prices and the predicted next close

---
## 🛠️ Installation
### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/AI-Stock-Price-Predictor.git
cd AI-Stock-Price-Predictor
```

### 2️⃣ **Install Dependencies**
Make sure you have Python **3.10+** installed. Then run:
```sh
pip install -r requirements.txt
```

### 3️⃣ **Run the Application**
```sh
python Stock.py
```

---
## 📦 Dependencies
- `customtkinter`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `xgboost`
- `yfinance`

---
## 📊 How It Works
1. Enter a stock symbol (e.g. `AAPL`, `TSLA`).
2. Click **Train Model** to download historical data and train the predictor.
3. Click **Predict Price** to forecast the next closing price.
4. The app displays the predicted price and the latest known close price.

---
## 🤖 Technologies Used
- **Python**
- **XGBoost** – Machine Learning Model
- **yfinance** – Fetches historical stock data
- **CustomTkinter** – Modern GUI Framework
- **Matplotlib** – Plotting library
- **Pandas & NumPy** – Data handling

---
## 📌 Notes
- The app requires at least 70 days of historical data to train properly.
- Predictions are based on the latest 60 closing prices.

---
## 📜 License
This project is licensed under the **MIT License**.

---
### 🌟 Star the repo if you like it! ⭐

