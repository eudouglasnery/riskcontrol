import yfinance as yf
import os
import time
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DataExtraction:
    """Manage downloading and caching of ticker price data."""
    def __init__(self, tickers: list, file_name: str = "tickers_data.csv", months: int | None = None):
        """
        Initialize with:
        - tickers: list of ticker symbols to load
        - file_name: CSV file to cache data
        - months: lookback window expressed in months; use None for full history
        """
        self.tickers_list = tickers
        self.file_name = file_name
        self.period = months

    def extract_data(self):
        """
        Return DataFrame for requested tickers.
        If cache CSV exists, update it with any missing tickers.
        Otherwise, download all data and save to CSV.
        """
        data_path = self.go_to_project_path(self.file_name)
        start, end = self.define_start_end_date(self.period)

        if os.path.exists(data_path):
            # CSV file exists: read it and download only missing tickers
            df = self.read_and_update_csv(self.tickers_list, start, end, data_path)
        else:
            # CSV file missing: download all tickers at once
            df = self.download_all(self.tickers_list, start, end, data_path)

        return df[self.tickers_list]

    @staticmethod
    def _download_prices(tickers: list, start, end):
        """Helper to call yfinance with either explicit dates or the maximum history."""
        params = {
            "tickers": tickers,
            "threads": False,
            "progress": False,
            "timeout": 30
        }
        if start is not None and end is not None:
            params.update({"start": start, "end": end})
        else:
            params["period"] = "max"

        return yf.download(**params)["Close"]

    @classmethod
    def read_and_update_csv(cls, tickers_list, start, end, data_path: str):
        """
        Read existing CSV into DataFrame.
        Identify any tickers not yet downloaded.
        Download missing tickers, merge into DataFrame, and overwrite CSV.
        """
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        missing = [ticker for ticker in tickers_list if ticker not in df.columns]
        if missing:
            new = cls._download_prices(missing, start, end)
            new.dropna(inplace=True)
            df = df.join(new, how="outer")
            df.to_csv(data_path)
        return df

    @staticmethod
    def go_to_project_path(file_name: str):
        """
        Compute the full path to the given file
        located in the project root directory.
        """
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_path, "", file_name)
        return data_path

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

    @classmethod
    def download_all(cls,
                     tickers_list: list,
                     start: datetime | None,
                     end: datetime | None,
                     data_path: str,
                     retries: int = 3,
                     wait: int = 5):
        """
        Download price data for all requested tickers.
        Save results to CSV and return the DataFrame.
        """
        for attempt in range(retries):
            try:
                df = cls._download_prices(tickers_list, start, end)
                df.dropna(how="all", inplace=True)
                df.to_csv(data_path)
                return df
            except Exception as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"Failed to download data after {retries} attempts: {e}") from e
                time.sleep(wait)
