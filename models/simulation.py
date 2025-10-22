from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    wealth_paths: np.ndarray
    ages: np.ndarray
    percentiles: pd.DataFrame
    final_distribution: pd.Series
    probability_success: float
    target_wealth: float


class MonteCarloPlanner:
    """
    Monte Carlo engine to project wealth accumulation until retirement.
    All inputs are expressed in annual terms and treated in real (inflation-adjusted) values.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        weights: pd.Series | np.ndarray,
        inflation: float = 0.0,
    ) -> None:
        if isinstance(weights, pd.Series):
            aligned = weights.reindex(expected_returns.index)
            if aligned.isna().any():
                missing = aligned[aligned.isna()].index.tolist()
                raise ValueError(f"Weights missing for assets: {missing}")
            self.weights = aligned.to_numpy(dtype=float)
        else:
            weights_array = np.asarray(weights, dtype=float)
            if weights_array.shape[0] != len(expected_returns):
                raise ValueError("Weights size must match expected_returns length.")
            self.weights = weights_array

        total_weight = np.sum(self.weights)
        if np.isclose(total_weight, 0.0):
            raise ValueError("Weights must sum to a positive number.")
        self.weights = self.weights / total_weight

        self.expected_returns = expected_returns.astype(float)
        self.covariance = covariance.loc[expected_returns.index, expected_returns.index].astype(float)
        self.inflation = float(inflation)

    def simulate_wealth_paths(
        self,
        initial_wealth: float,
        annual_contribution: float,
        horizon_years: int,
        n_sims: int = 10_000,
        contribution_growth: float | None = None,
        rng_seed: int | None = None,
    ) -> np.ndarray:
        """
        Simulate wealth accumulation assuming yearly rebalancing to the target weights.
        Returns a matrix with shape (n_sims, horizon_years + 1).
        """
        if horizon_years <= 0:
            raise ValueError("horizon_years must be greater than zero.")
        if initial_wealth < 0:
            raise ValueError("initial_wealth must be non-negative.")
        if annual_contribution < 0:
            raise ValueError("annual_contribution must be non-negative.")
        if n_sims <= 0:
            raise ValueError("n_sims must be positive.")

        rng = np.random.default_rng(rng_seed)
        mu_vec = self.expected_returns.to_numpy()
        cov_matrix = self.covariance.to_numpy()
        n_assets = mu_vec.shape[0]

        # Draw annual asset returns for each simulation and year.
        asset_returns = rng.multivariate_normal(
            mean=mu_vec,
            cov=cov_matrix,
            size=(horizon_years, n_sims),
        )  # shape: (years, sims, assets)

        # Convert to portfolio returns via matrix product.
        portfolio_returns = np.tensordot(asset_returns, self.weights, axes=([2], [0]))  # (years, sims)
        portfolio_returns = np.clip(portfolio_returns, -0.99, None)  # prevent collapse below -100%

        if contribution_growth is None:
            contribution_growth = 0.0
        contribution_growth = float(contribution_growth)

        contrib_schedule = annual_contribution * (1 + contribution_growth) ** np.arange(horizon_years)

        inflation_adjustment = 1.0 + self.inflation
        adjusted_returns = ((1.0 + portfolio_returns) / inflation_adjustment) - 1.0

        wealth = np.zeros((n_sims, horizon_years + 1), dtype=float)
        wealth[:, 0] = initial_wealth

        for year in range(horizon_years):
            contrib = contrib_schedule[year]
            wealth[:, year] = np.maximum(wealth[:, year], 0.0)
            wealth[:, year + 1] = (wealth[:, year] + contrib) * (1.0 + adjusted_returns[year])

        wealth = np.maximum(wealth, 0.0)
        return wealth

    @staticmethod
    def wealth_percentiles(
        wealth_paths: np.ndarray,
        percentiles: Sequence[float] = (10, 25, 50, 75, 90),
        index: Iterable[int] | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with percentile trajectories for the simulated wealth paths.
        """
        pct_array = np.percentile(wealth_paths, percentiles, axis=0)
        pct_df = pd.DataFrame(pct_array.T, columns=[f"p{int(p)}" for p in percentiles])
        if index is not None:
            pct_df.index = list(index)
        return pct_df

    @staticmethod
    def final_distribution(
        wealth_paths: np.ndarray,
        percentiles: Sequence[float] = (5, 10, 25, 50, 75, 90, 95),
    ) -> pd.Series:
        final_values = wealth_paths[:, -1]
        pct_values = np.percentile(final_values, percentiles, axis=0)
        return pd.Series(pct_values, index=[f"p{int(p)}" for p in percentiles])

    @staticmethod
    def probability_of_target(wealth_paths: np.ndarray, target: float) -> float:
        final_values = wealth_paths[:, -1]
        if target <= 0:
            return 1.0
        probability = np.mean(final_values >= target)
        return float(probability)

    def run_simulation(
        self,
        *,
        current_age: int,
        retirement_age: int,
        initial_wealth: float,
        annual_contribution: float,
        desired_retirement_income: float,
        withdrawal_rate: float,
        inflation: float | None = None,
        n_sims: int = 10_000,
        contribution_growth: float | None = None,
        rng_seed: int | None = None,
    ) -> SimulationResult:
        """
        Helper wrapper to run the full simulation pipeline and capture summary statistics.
        """
        if retirement_age <= current_age:
            raise ValueError("retirement_age must be greater than current_age.")
        if withdrawal_rate <= 0:
            raise ValueError("withdrawal_rate must be positive.")

        horizon_years = retirement_age - current_age
        if inflation is not None:
            self.inflation = float(inflation)

        wealth_paths = self.simulate_wealth_paths(
            initial_wealth=initial_wealth,
            annual_contribution=annual_contribution,
            horizon_years=horizon_years,
            n_sims=n_sims,
            contribution_growth=contribution_growth,
            rng_seed=rng_seed,
        )

        ages = np.arange(current_age, retirement_age + 1)
        percentiles = self.wealth_percentiles(wealth_paths, index=ages)

        target_wealth = desired_retirement_income / withdrawal_rate if withdrawal_rate > 0 else 0.0
        probability = self.probability_of_target(wealth_paths, target_wealth)
        final_pct = self.final_distribution(wealth_paths)

        return SimulationResult(
            wealth_paths=wealth_paths,
            ages=ages,
            percentiles=percentiles,
            final_distribution=final_pct,
            probability_success=probability,
            target_wealth=target_wealth,
        )
