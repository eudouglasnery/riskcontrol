import unittest

import numpy as np
import pandas as pd

from models.simulation import MonteCarloPlanner


class MonteCarloPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected_returns = pd.Series(
            {"PETR4.SA": 0.12, "TAEE11.SA": 0.08}
        )
        cls.covariance = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.03]],
            index=cls.expected_returns.index,
            columns=cls.expected_returns.index,
        )
        cls.weights = pd.Series(
            {"PETR4.SA": 0.55, "TAEE11.SA": 0.45}
        )
        cls.planner = MonteCarloPlanner(
            expected_returns=cls.expected_returns,
            covariance=cls.covariance,
            weights=cls.weights,
            inflation=0.03,
        )

    def test_weights_are_normalized(self):
        self.assertAlmostEqual(float(self.planner.weights.sum()), 1.0)
        self.assertTrue(np.all(self.planner.weights >= 0.0))

    def test_simulate_wealth_paths_shape_and_bounds(self):
        wealth = self.planner.simulate_wealth_paths(
            initial_wealth=200_000.0,
            annual_contribution=30_000.0,
            horizon_years=10,
            n_sims=400,
            contribution_growth=0.02,
            rng_seed=123,
        )
        self.assertEqual(wealth.shape, (400, 11))
        self.assertTrue(np.all(wealth >= 0.0))

        mean_terminal = wealth[:, -1].mean()
        self.assertGreater(mean_terminal, 0.0)

    def test_wealth_percentiles(self):
        wealth = self.planner.simulate_wealth_paths(
            initial_wealth=100_000.0,
            annual_contribution=20_000.0,
            horizon_years=5,
            n_sims=200,
            rng_seed=321,
        )
        ages = range(35, 41)
        percentiles = MonteCarloPlanner.wealth_percentiles(wealth, index=ages)
        self.assertEqual(list(percentiles.columns), ["p10", "p25", "p50", "p75", "p90"])
        self.assertEqual(len(percentiles), 6)
        self.assertEqual(list(percentiles.index), list(ages))

    def test_final_distribution_and_probability(self):
        wealth = self.planner.simulate_wealth_paths(
            initial_wealth=80_000.0,
            annual_contribution=15_000.0,
            horizon_years=4,
            n_sims=300,
            rng_seed=99,
        )
        distribution = MonteCarloPlanner.final_distribution(wealth)
        self.assertEqual(list(distribution.index), ["p5", "p10", "p25", "p50", "p75", "p90", "p95"])
        self.assertTrue(np.all(distribution.values >= 0.0))

        probability = MonteCarloPlanner.probability_of_target(wealth, target=150_000.0)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_run_simulation_pipeline(self):
        result = self.planner.run_simulation(
            current_age=35,
            retirement_age=45,
            initial_wealth=150_000.0,
            annual_contribution=25_000.0,
            desired_retirement_income=120_000.0,
            withdrawal_rate=0.04,
            inflation=0.03,
            n_sims=500,
            contribution_growth=0.01,
            rng_seed=2024,
        )

        self.assertEqual(result.wealth_paths.shape, (500, 11))
        self.assertEqual(result.ages[0], 35)
        self.assertEqual(result.ages[-1], 45)
        self.assertEqual(result.target_wealth, 3_000_000.0)
        self.assertTrue(0.0 <= result.probability_success <= 1.0)
        self.assertEqual(list(result.final_distribution.index), ["p5", "p10", "p25", "p50", "p75", "p90", "p95"])
        self.assertEqual(list(result.percentiles.columns), ["p10", "p25", "p50", "p75", "p90"])

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            self.planner.simulate_wealth_paths(
                initial_wealth=100_000.0,
                annual_contribution=10_000.0,
                horizon_years=0,
            )
        with self.assertRaises(ValueError):
            self.planner.run_simulation(
                current_age=40,
                retirement_age=40,
                initial_wealth=100_000.0,
                annual_contribution=10_000.0,
                desired_retirement_income=80_000.0,
                withdrawal_rate=0.04,
            )
        with self.assertRaises(ValueError):
            self.planner.run_simulation(
                current_age=40,
                retirement_age=50,
                initial_wealth=100_000.0,
                annual_contribution=10_000.0,
                desired_retirement_income=80_000.0,
                withdrawal_rate=0.0,
            )


if __name__ == "__main__":
    unittest.main()
