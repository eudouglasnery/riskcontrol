import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_annualized_volatility(returns: pd.DataFrame) -> pd.Series:
    """
    Calculates annualized volatility for each asset based on daily returns.
    Considering 252 business days.
    """
    return returns.std() * np.sqrt(252)


def calculate_parametric_var(returns: pd.DataFrame, confidence_level=0.95) -> pd.Series:
    """
    Return the parametric Value at Risk (VaR) at a given confidence level.
    Assumes normally distributed returns.
    """
    z_score = norm.ppf(1 - confidence_level)
    return returns.mean() + returns.std() * z_score


def calculate_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix between the assets.
    """
    return returns.corr()
