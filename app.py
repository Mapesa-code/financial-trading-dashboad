import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from flask import request, jsonify
from drive_utils import upload_dataframe_to_drive

@app.route("/save_drive", methods=["POST"])
def save_drive():
    ticker = request.args.get("ticker", "AAPL")
    period = request.args.get("period", "1y")
    df = get_stock_data(ticker, period)
    filename = f"{ticker}_{period}_data.csv"
    url = upload_dataframe_to_drive(df, filename)
    return jsonify({"drive_url": url})

# Optional TensorFlow/Keras
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    HAS_TF = True
except Exception:
    HAS_TF = False

app = Flask(__name__)

def get_stock_data(ticker: str = "AAPL", period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    df = df.dropna()
    df["Return"] = df["Close"].pct_change()
    df["Target"] = df["Close"].shift(-1)
    df = df.dropna()
    return df

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[["Open", "High", "Low", "Close", "Volume", "Return"]].astype(float)
    y = df["Target"].astype(float)
    return train_test_split(X, y, test_size=0.2, shuffle=False)

def train_lightgbm(X_train, y_train, X_test, y_test):
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return model, float(rmse)

def train_keras(X_train, y_train, X_test, y_test):
    if not HAS_TF:
        return None, None
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    preds = model.predict(X_test, verbose=0).flatten()
    rmse = mean_squared_error(y_test, preds, squared=False)
    return model, float(rmse)

@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok", "tensorflow_available": HAS_TF})

@app.route("/predict", methods=["GET"])
def predict() -> Any:
    ticker = request.args.get("ticker", "AAPL")
    period = request.args.get("period", "1y")
    try:
        df = get_stock_data(ticker=ticker, period=period)
        X_train, X_test, y_train, y_test = prepare_data(df)

        lgb_model, lgb_rmse = train_lightgbm(X_train, y_train, X_test, y_test)
        lgb_pred = float(lgb_model.predict([X_test.iloc[-1]])[0])

        keras_pred_value = None
        keras_rmse_value = None
        if HAS_TF:
            keras_model, keras_rmse = train_keras(X_train, y_train, X_test, y_test)
            if keras_model is not None:
                keras_pred_value = float(keras_model.predict([X_test.iloc[-1]], verbose=0)[0][0])
                keras_rmse_value = float(keras_rmse) if keras_rmse is not None else None

        return jsonify({
            "ticker": ticker,
            "period": period,
            "latest_close": float(df["Close"].iloc[-1]),
            "lightgbm_prediction": lgb_pred,
            "lightgbm_rmse": float(lgb_rmse),
            "keras_prediction": keras_pred_value,
            "keras_rmse": keras_rmse_value
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)
