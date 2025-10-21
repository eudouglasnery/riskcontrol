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
    def calculate_historical_var(returns: pd.DataFrame, confidence_level: float = 0.95) -> pd.Series:
        """
        Historical VaR based on the empirical distribution of returns.
        """
        tail_probability = 1 - confidence_level
        return returns.quantile(tail_probability)

    @staticmethod
    def calculate_cvar(returns: pd.DataFrame, confidence_level: float = 0.95) -> pd.Series:
        """
        Conditional VaR (Expected Shortfall) computed as the average of losses
        that exceed the historical VaR threshold.
        """
        var_thresholds = RiskIndicators.calculate_historical_var(returns, confidence_level)

        def tail_mean(column: pd.Series) -> float:
            threshold = var_thresholds[column.name]
            tail = column[column <= threshold]
            if tail.empty:
                return threshold
            return tail.mean()

        cvar = returns.apply(tail_mean, axis=0)
        return cvar

    @staticmethod
    def calculate_correlation(returns: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the Pearson correlation matrix between assets.
        """
        return returns.corr()

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.DataFrame,
                               risk_free_rate: float = 0.0,
                               periods_per_year: int = 252) -> pd.Series:
        """
        Annualized Sharpe Ratio using excess returns over the supplied risk-free rate.
        - risk_free_rate is annual (e.g., 0.1 for 10%).
        - periods_per_year defaults to 252 for daily data.
        """
        if risk_free_rate:
            daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        else:
            daily_rf = 0.0

        excess_returns = returns - daily_rf
        mean_excess = excess_returns.mean() * periods_per_year
        volatility = returns.std() * np.sqrt(periods_per_year)

        sharpe = mean_excess / volatility
        sharpe.replace([np.inf, -np.inf], np.nan, inplace=True)
        return sharpe
