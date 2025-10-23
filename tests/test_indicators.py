import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

from models.indicators import RiskIndicators


class RiskIndicatorsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        cls.returns = pd.DataFrame(
            {
                "PETR4.SA": [0.01, -0.02, 0.015, 0.0, 0.005, -0.01],
                "TAEE11.SA": [0.005, 0.007, -0.01, 0.012, -0.004, 0.003],
                "BBSE3.SA": [0.008, -0.004, 0.006, 0.001, -0.002, 0.004],
            },
            index=dates,
        )

    def test_calculate_annualized_volatility(self):
        result = RiskIndicators.calculate_annualized_volatility(self.returns)
        expected = self.returns.std() * np.sqrt(252)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_calculate_parametric_var(self):
        for confidence in (0.95, 0.99):
            result = RiskIndicators.calculate_parametric_var(self.returns, confidence_level=confidence)
            z_score = norm.ppf(1 - confidence)
            expected = self.returns.mean() + self.returns.std() * z_score
            pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_calculate_historical_var(self):
        result = RiskIndicators.calculate_historical_var(self.returns)
        expected = self.returns.quantile(0.05)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_calculate_cvar(self):
        result = RiskIndicators.calculate_cvar(self.returns)
        thresholds = self.returns.quantile(0.05)

        def manual_cvar(column: pd.Series) -> float:
            threshold = thresholds[column.name]
            tail = column[column <= threshold]
            return tail.mean() if not tail.empty else threshold

        expected = self.returns.apply(manual_cvar, axis=0)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_calculate_sharpe_ratio(self):
        risk_free = 0.06  # 6% annual
        periods = 252
        result = RiskIndicators.calculate_sharpe_ratio(self.returns, risk_free_rate=risk_free)

        daily_rf = (1 + risk_free) ** (1 / periods) - 1
        excess = self.returns - daily_rf
        expected = (excess.mean() * periods) / (self.returns.std() * np.sqrt(periods))
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_calculate_correlation(self):
        result = RiskIndicators.calculate_correlation(self.returns)
        expected = self.returns.corr()
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
