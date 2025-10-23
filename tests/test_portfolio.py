import unittest

import numpy as np
import pandas as pd

from models.portfolio import PortfolioAnalytics


class PortfolioAnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        cls.returns = pd.DataFrame(
            {
                "PETR4.SA": [0.012, -0.018, 0.017, 0.001, 0.006, -0.011],
                "TAEE11.SA": [0.004, 0.006, -0.009, 0.011, -0.003, 0.004],
                "BBSE3.SA": [0.007, -0.003, 0.005, 0.002, -0.002, 0.003],
            },
            index=dates,
        )
        cls.mu, cls.cov = PortfolioAnalytics.annualized_inputs(cls.returns)
        cls.equal_weights = np.full(len(cls.returns.columns), 1 / len(cls.returns.columns))

    def test_annualized_inputs(self):
        expected_mu = self.returns.mean() * 252
        expected_cov = self.returns.cov() * 252
        pd.testing.assert_series_equal(self.mu, expected_mu, check_names=False)
        pd.testing.assert_frame_equal(self.cov, expected_cov)

    def test_normalize_weights(self):
        weights = np.array([0.2, 0.5, 0.3])
        result = PortfolioAnalytics.normalize_weights(weights)
        self.assertAlmostEqual(result.sum(), 1.0)
        np.testing.assert_array_almost_equal(result, np.array([0.2, 0.5, 0.3]))

        zero_weights = np.zeros(3)
        result_zero = PortfolioAnalytics.normalize_weights(zero_weights)
        self.assertAlmostEqual(result_zero.sum(), 1.0)
        self.assertTrue(np.allclose(result_zero, self.equal_weights))

    def test_portfolio_return_and_volatility(self):
        weights = np.array([0.5, 0.3, 0.2])
        expected_return = float(np.dot(weights, self.mu.values))
        variance_matrix = weights.reshape(1, -1) @ self.cov.values @ weights.reshape(-1, 1)
        expected_vol = float(np.sqrt(variance_matrix[0, 0]))
        result_return = PortfolioAnalytics.portfolio_return(weights, self.mu)
        result_vol = PortfolioAnalytics.portfolio_volatility(weights, self.cov)

        self.assertAlmostEqual(result_return, expected_return)
        self.assertAlmostEqual(result_vol, expected_vol)

    def test_optimize_max_sharpe(self):
        optimized = PortfolioAnalytics.optimize_max_sharpe(self.mu, self.cov)
        self.assertAlmostEqual(optimized.sum(), 1.0, places=6)
        self.assertTrue(np.all(optimized >= -1e-9))

        sharpe_equal = PortfolioAnalytics.portfolio_sharpe(
            self.equal_weights, self.mu, self.cov
        )
        sharpe_optimized = PortfolioAnalytics.portfolio_sharpe(
            optimized, self.mu, self.cov
        )
        self.assertGreaterEqual(sharpe_optimized, sharpe_equal - 1e-6)

    def test_optimize_min_vol(self):
        optimized = PortfolioAnalytics.optimize_min_vol(self.mu, self.cov)
        self.assertAlmostEqual(optimized.sum(), 1.0, places=6)
        self.assertTrue(np.all(optimized >= -1e-9))

        vol_equal = PortfolioAnalytics.portfolio_volatility(
            self.equal_weights, self.cov
        )
        vol_optimized = PortfolioAnalytics.portfolio_volatility(optimized, self.cov)
        self.assertLessEqual(vol_optimized, vol_equal + 1e-6)

    def test_efficient_frontier(self):
        frontier = PortfolioAnalytics.efficient_frontier(self.mu, self.cov, points=5)
        self.assertEqual(len(frontier), 5)
        self.assertIn("Return", frontier.columns)
        self.assertIn("Volatility", frontier.columns)
        np.testing.assert_array_almost_equal(frontier["Volatility"].values, np.sort(frontier["Volatility"].values))


if __name__ == "__main__":
    unittest.main()
