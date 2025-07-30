import yfinance as yf
import os
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DataExtraction:
    """Manage downloading and caching of ticker price data."""
    def __init__(self, tickers: list, file_name: str = "tickers_data.csv", months: int = 6):
        """
        Initialize with:
        - tickers: list of ticker symbols to load
        - file_name: CSV file to cache data
        - period: lookback window in days for download
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
    def read_and_update_csv(tickers_list, start, end, data_path: str):
        """
        Read existing CSV into DataFrame.
        Identify any tickers not yet downloaded.
        Download missing tickers, merge into DataFrame, and overwrite CSV.
        """
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        missing = [ticker for ticker in tickers_list if ticker not in df.columns]
        if missing:
            new = yf.download(missing, start=start, end=end)["Close"]
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
    def define_start_end_date(period: int = 6):
        """
        Given a period in days, return (start, end) datetime pair,
        where 'end' is today and 'start' is 'period' days before.
        """
        end = datetime.today()
        start = end - relativedelta(months=period)
        return start, end

    @staticmethod
    def download_all(tickers_list: list, start, end, data_path):
        """
        Download price data for all requested tickers.
        Save results to CSV and return the DataFrame.
        """
        tickers_data = yf.download(tickers_list, start=start, end=end)["Close"]
        tickers_data.dropna(inplace=True)
        tickers_data.to_csv(data_path)
        return tickers_data
