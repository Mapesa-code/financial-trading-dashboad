import os
import argparse
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def fetch_and_save(symbols, hours_back: int, output: str | None, timeframe: str = "Hour") -> pd.DataFrame:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise SystemExit("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment.")
    client = StockHistoricalDataClient(api_key, secret_key)
    end = datetime.utcnow()
    start = end - timedelta(hours=hours_back)
    tf = TimeFrame.Hour if timeframe.lower() == "hour" else TimeFrame.Day
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start, end=end)
    bars = client.get_stock_bars(req)
    df = bars.df
    if output:
        df.to_csv(output, index=True)
        print(f"Saved {len(df)} rows to {output}")
    else:
        print(df.head())
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch bar data from Alpaca and optionally save to CSV.")
    parser.add_argument("--symbols", nargs="+", default=["AAPL"], help="List of symbols, default AAPL")
    parser.add_argument("--hours", type=int, default=24 * 7, help="How many hours back to fetch (default 168)")
    parser.add_argument("--timeframe", choices=["Hour", "Day"], default="Hour", help="Bar timeframe")
    parser.add_argument("--output", default="", help="CSV output path. If empty, just prints head().")
    args = parser.parse_args()
    output_path = args.output or None
    fetch_and_save(args.symbols, args.hours, output_path, timeframe=args.timeframe)

if __name__ == "__main__":
    main()
