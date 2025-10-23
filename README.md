# Market Risk Dashboard

## Project Objectives
- Deliver an interactive market risk dashboard focused on Brazilian equities and REITs.
- Automate price collection, cleaning, and incremental storage so the app stays fast after the first run.
- Offer risk analytics, portfolio optimisation, and Monte Carlo retirement planning in a single workflow.

## Tech Stack
- **Language**: Python 3.12
- **Core libraries**: Streamlit, Plotly, Pandas, NumPy, SciPy, yfinance, python-dateutil
- **Tooling**: Git for version control

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/your-user/riskcontrol.git
   cd riskcontrol
   ```
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## Key Features

**Data collection and cache**
- Downloads the last six months of prices from `yfinance`.
- Persists prices in `tickers_data.csv`; only missing tickers are fetched in subsequent runs.
- Ships with a default ticker list and a sidebar input to add any B3 symbol (suffix `.SA`).

**Return analytics**
- Interactive price history and daily return charts.
- Histogram of returns to assess the normality assumption.
- Drawdown curve to monitor peak-to-trough losses.
- Rolling annualised volatility for 21-day and 63-day windows.

**Risk indicators**
- Annualised volatility.
- Parametric VaR (95% and 99%).
- Historical VaR and CVaR / Expected Shortfall (95%).
- Sharpe ratio that honours the user-defined risk-free rate.
- Correlation heatmap to compare asset co-movements.

**Portfolio and optimisation**
- Manual weights with automatic normalisation plus an annual risk-free rate input.
- Portfolio summary with expected return, volatility, and Sharpe.
- Markowitz optimisation to maximise Sharpe or minimise volatility.
- Efficient frontier chart highlighting the optimal and minimum risk portfolios.

**Financial planning (Monte Carlo)**
- Detailed inputs: current and retirement ages, income, expenses, extra contributions, contribution growth, inflation, withdrawal rate, number of simulations, and RNG seed.
- Simulates annual real returns with rebalancing and inflows that can grow over time.
- Fan chart for wealth percentiles (p10, p25, p50, p75, p90) plus the final wealth distribution.
- Key outputs: probability of reaching the target wealth implied by the withdrawal rate, target level, median projection, and estimated annual contribution.

**Interactive interface**
- Tabs for Return Analysis, Risk, Correlation, Portfolio, and Financial Planning.
- Contextual guidance through expanders in each section.
- Plotly visualisations embedded directly in Streamlit.

## Repository Structure
- `app.py`: main Streamlit flow that wires together data, indicators, visuals, and simulations.
- `models/data_extraction.py`: handles downloads, rolling window definition, and cache persistence.
- `models/indicators.py`: volatility, VaR (parametric and historical), CVaR, Sharpe, and correlation calculations.
- `models/portfolio.py`: return/volatility helpers, weight normalisation, optimisation, and efficient frontier sampling.
- `models/visualizations.py`: interactive charts and tables (prices, returns, risk metrics, correlation, frontier, simulations).
- `models/simulation.py`: Monte Carlo engine for retirement planning scenarios.
- `tickers_data.csv`: cache file created after the first data pull.
