import numpy as np
import pandas as pd
from scipy.stats import norm


class RiskIndicators:
    """
    Provides methods to calculate market risk indicators.
    """

    @staticmethod
    def calculate_annualized_volatility(returns: pd.DataFrame) -> pd.Series:
        """
        Calculates annualized volatility for each asset based on daily returns.
        Assumes 252 trading days per year.
        """
        return returns.std() * np.sqrt(252)

    @staticmethod
    def calculate_parametric_var(returns: pd.DataFrame, confidence_level: float = 0.95) -> pd.Series:
        """
        Returns parametric Value at Risk (VaR) at a given confidence level.
        Assumes returns are normally distributed.
        """
        z_score = norm.ppf(1 - confidence_level)
        return returns.mean() + returns.std() * z_score

    @staticmethod
    def calculate_correlation(returns: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the Pearson correlation matrix between assets.
        """
        return returns.corr()
