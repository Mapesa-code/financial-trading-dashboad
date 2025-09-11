# Stock ML Prediction & Data Fetching API

Deployable Flask API for stock price prediction using LightGBM and TensorFlow/Keras, with historical data fetching via Alpaca and Yahoo Finance.

---

## 🚀 Features

- **REST API endpoints:**
  - `/health`: Service health/status (shows if TensorFlow is available)
  - `/predict`: Predict next-day closing price for a stock ticker (LightGBM and optional Keras baseline)
  - `/save`: Example for saving data to external file storage (see below)
- **Fetch historical data:**
  - From Yahoo Finance (via `yfinance`)
  - From Alpaca Markets (via `alpaca-py`, see `scripts/fetch_alpaca_bars.py`)
- **ML Baselines:**
  - LightGBM regression
  - Keras (TensorFlow) regression (optional, install `tensorflow-cpu`)
- **External storage ready:** 
  - Example S3 utility included for permanent CSV storage (see [External File Storage](#external-file-storage))
- **Easy cloud deployment:** 
  - Render, Heroku, or any Gunicorn-compatible host
- **Configurable via environment variables** (`.env.example` provided)

---

## 🏗️ Project Structure

```
├── app.py                     # Main Flask API
├── scripts/
│   └── fetch_alpaca_bars.py   # Alpaca bar fetch utility
├── requirements.txt           # Python dependencies
├── .env.example               # Example environment variables
├── .gitignore                 # Git ignore rules
└── README.md                  # (This file)
```

---

## ⚡ Quickstart

### 1. Clone and Install

```bash
git clone https://github.com/Mapesa-code/your-repo.git
cd your-repo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Copy `.env.example` to `.env` and fill in your Alpaca keys and Flask config.

```env
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
FLASK_RUN_HOST=0.0.0.0
FLASK_RUN_PORT=5000
FLASK_DEBUG=1
```

### 3. Run Locally

```bash
flask run
# or for production:
gunicorn app:app --bind 0.0.0.0:5000
```

---

## 🌍 API Endpoints

### Health Check

`GET /health`

```json
{
  "status": "ok",
  "tensorflow_available": true
}
```

### Predict Next-Day Close

`GET /predict?ticker=AAPL&period=1y`

Returns predictions from both LightGBM and Keras (if available), plus RMSEs.

### Fetch Alpaca Bars

```bash
python scripts/fetch_alpaca_bars.py --symbols AAPL --hours 168 --timeframe Hour --output aapl_bars.csv
```

---

## 🗄️ External File Storage

For **permanent CSV or data storage**, use Amazon S3 or similar. Example using `boto3`:

```python
from storage_utils import upload_dataframe_to_s3

@app.route("/save", methods=["POST"])
def save_data():
    ticker = request.args.get("ticker", "AAPL")
    period = request.args.get("period", "1y")
    df = get_stock_data(ticker, period)
    key = f"{ticker}_{period}_data.csv"
    url = upload_dataframe_to_s3(df, key)
    return jsonify({"url": url})
```

See `storage_utils.py` for more details.

---

## 🛠️ Deployment (Render Example)

1. Push to GitHub
2. Add `render.yaml`:

```yaml
services:
  - type: web
    name: flask-stock-ml-app
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:5000
    envVars:
      - key: FLASK_RUN_HOST
        value: 0.0.0.0
      - key: FLASK_RUN_PORT
        value: 5000
      - key: FLASK_DEBUG
        value: 1
```

3. Add your Alpaca (and S3) keys in Render Dashboard.

---

## 📦 Requirements

- Python 3.10+
- Flask
- yfinance
- lightgbm
- scikit-learn
- pandas, numpy
- alpaca-py
- gunicorn
- (optional) tensorflow-cpu (`pip install tensorflow-cpu==2.17.*`)
- (optional) boto3 for S3

---

## 📚 References

- [Yahoo Finance API — yfinance](https://github.com/ranaroussi/yfinance)
- [Alpaca Data Docs](https://alpaca.markets/docs/api-references/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [Render](https://render.com/docs/deploy-python)

---

## 🤝 Contributing

PRs, issues, and suggestions welcome!  
See [CONTRIBUTING.md](CONTRIBUTING.md) (if available).

---

## 📝 License

MIT (see [LICENSE](LICENSE))
