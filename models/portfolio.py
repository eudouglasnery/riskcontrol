import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy.optimize import minimize


class PortfolioAnalytics:
    """Portfolio analytics and optimization (no shorting by default)."""

    @staticmethod
    def annualized_inputs(returns: pd.DataFrame, periods: int = 252) -> Tuple[pd.Series, pd.DataFrame]:
        mu = returns.mean() * periods
        cov = returns.cov() * periods
        return mu, cov

    @staticmethod
    def normalize_weights(weights: pd.Series | np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        s = w.sum()
        if s == 0:
            return np.full_like(w, 1.0 / len(w))
        return w / s

    @staticmethod
    def portfolio_return(weights: np.ndarray, mu: pd.Series) -> float:
        return float(np.dot(weights, mu.values))

    @staticmethod
    def portfolio_volatility(weights: np.ndarray, cov: pd.DataFrame) -> float:
        w = weights.reshape(-1, 1)
        return float(np.sqrt((w.T @ cov.values @ w)[0, 0]))

    @staticmethod
    def portfolio_sharpe(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> float:
        ret = PortfolioAnalytics.portfolio_return(weights, mu)
        vol = PortfolioAnalytics.portfolio_volatility(weights, cov)
        if vol == 0:
            return 0.0
        return (ret - rf) / vol

    @staticmethod
    def optimize_max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0,
                            bounds: Optional[Tuple[float, float]] = (0.0, 1.0)) -> np.ndarray:
        n = len(mu)
        x0 = np.full(n, 1.0 / n)
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        bnds = tuple(bounds for _ in range(n)) if bounds else None

        def objective(w):
            # maximize Sharpe => minimize negative Sharpe
            s = PortfolioAnalytics.portfolio_sharpe(w, mu, cov, rf)
            return -s

        res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
        if not res.success:
            return PortfolioAnalytics.normalize_weights(x0)
        return PortfolioAnalytics.normalize_weights(res.x)

    @staticmethod
    def optimize_min_vol(mu: pd.Series, cov: pd.DataFrame,
                         bounds: Optional[Tuple[float, float]] = (0.0, 1.0)) -> np.ndarray:
        n = len(mu)
        x0 = np.full(n, 1.0 / n)
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        bnds = tuple(bounds for _ in range(n)) if bounds else None

        def objective(w):
            return PortfolioAnalytics.portfolio_volatility(w, cov)

        res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
        if not res.success:
            return PortfolioAnalytics.normalize_weights(x0)
        return PortfolioAnalytics.normalize_weights(res.x)

    @staticmethod
    def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, points: int = 50,
                           bounds: Optional[Tuple[float, float]] = (0.0, 1.0)) -> pd.DataFrame:
        """
        Compute a sampled efficient frontier (no shorting) by sweeping target returns
        and minimizing variance subject to sum(weights)=1 and target return equality.
        Returns a DataFrame with columns: 'Return', 'Volatility', and weights per asset.
        """
        n = len(mu)
        x0 = np.full(n, 1.0 / n)
        bnds = tuple(bounds for _ in range(n)) if bounds else None

        r_min, r_max = float(mu.min()), float(mu.max())
        # Expand slightly to avoid degenerate constraints when all mu are equal
        if abs(r_max - r_min) < 1e-9:
            r_min *= 0.99
            r_max *= 1.01
        targets = np.linspace(r_min, r_max, points)

        rows = []
        for tr in targets:
            cons = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w, tr=tr: np.dot(w, mu.values) - tr},
            )

            def objective(w):
                return PortfolioAnalytics.portfolio_volatility(w, cov)

            res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
            if not res.success:
                w = PortfolioAnalytics.normalize_weights(x0)
            else:
                w = PortfolioAnalytics.normalize_weights(res.x)

            ret = PortfolioAnalytics.portfolio_return(w, mu)
            vol = PortfolioAnalytics.portfolio_volatility(w, cov)
            row = {'Return': ret, 'Volatility': vol}
            # store weights per asset (column names from mu.index)
            row.update({asset: weight for asset, weight in zip(mu.index, w)})
            rows.append(row)

        df = pd.DataFrame(rows)
        return df.sort_values('Volatility').reset_index(drop=True)
