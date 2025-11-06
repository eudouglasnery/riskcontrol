import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DataExtraction:
    """Manage downloading and caching of ticker price data."""

    def __init__(self, tickers: list, months: int | None = None):
        """
        Initialize with:
        - tickers: list of ticker symbols to load
        - months: lookback window expressed in months; use None for full history
        """
        self.tickers_list = tickers
        self.period = months

    def extract_data(self):
        """
        Return DataFrame for requested tickers.
        Each ticker is cached individually via Streamlit to avoid redundant downloads.
        """
        start, end = self.define_start_end_date(self.period)

        frames = []
        for ticker in self.tickers_list:
            ticker_df = _cached_ticker_close_prices(ticker, start, end)
            if ticker_df.empty:
                continue
            frames.append(ticker_df)

        if not frames:
            return pd.DataFrame(columns=self.tickers_list)

        data = pd.concat(frames, axis=1).sort_index()
        missing = [ticker for ticker in self.tickers_list if ticker not in data.columns]
        for ticker in missing:
            data[ticker] = pd.Series(dtype=float)

        return data[self.tickers_list]

    @staticmethod
    def ticker_exists(ticker: str, lookback: str = "1mo") -> bool:
        """
        Check whether a ticker yields price data over a short lookback window.
        Returns True when at least one close price is available, False otherwise.
        """
        try:
            history = yf.Ticker(ticker).history(period=lookback, auto_adjust=False)
        except Exception:
            return False
        if history.empty or "Close" not in history.columns:
            return False
        return not history["Close"].dropna().empty

    @staticmethod
    def define_start_end_date(period: int | None = None):
        """
        Given a period in months, return (start, end) datetime pair.
        When period is None, fall back to requesting the maximum history available.
        """
        if period is None:
            return None, None

        end = datetime.today()
        start = end - relativedelta(months=period)
        return start, end


@st.cache_data(show_spinner=False)
def _cached_ticker_close_prices(ticker: str, start: datetime | None, end: datetime | None) -> pd.DataFrame:
    """
    Download (and cache) close prices for a single ticker.
    Streamlit caches the result per (ticker, start, end) combination so the
    application avoids repeated downloads for the same asset.
    """
    params = {
        "tickers": [ticker],
        "threads": False,
        "progress": False,
        "timeout": 30,
        "auto_adjust": False,
    }
    if start is not None and end is not None:
        params.update({"start": start, "end": end})
    else:
        params["period"] = "max"

    try:
        closes = yf.download(**params)["Close"]
    except KeyError:
        return pd.DataFrame(columns=[ticker])

    if closes.empty:
        return pd.DataFrame(columns=[ticker])

    if ticker in closes.columns:
        data = closes[[ticker]]
    else:
        # When only one ticker is requested, ensure the column name is consistent
        data = closes.rename(columns={col: ticker for col in closes.columns})
        data = data[[ticker]]

    data.dropna(how="all", inplace=True)
    data.index = pd.to_datetime(data.index)
    return data
