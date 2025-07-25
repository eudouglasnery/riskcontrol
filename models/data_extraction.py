import yfinance as yf
import os
from datetime import datetime, timedelta


def extract_data(tickers: list, file_name="tickers_data.csv"):
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_path, "", file_name)

    end = datetime.today()
    start = end - timedelta(days=180)  # last 6 months

    tickers_data = yf.download(tickers, start=start, end=end)["Close"]
    tickers_data.dropna(inplace=True)
    tickers_data.to_csv(data_path)

    return tickers_data
