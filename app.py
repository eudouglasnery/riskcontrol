import pandas as pd
import streamlit as st

from models.data_extraction import DataExtraction
from models.indicators import RiskIndicators
from models.visualizations import DataVisualizations
from models.portfolio import PortfolioAnalytics
from models.simulation import MonteCarloPlanner


def format_currency(value: float) -> str:
    return f"R$ {value:,.2f}"


st.set_page_config(page_title="Market Risk Dashboard", layout="wide")
st.title("Market Risk Analysis")

# --- Default tickers cached in session ---
if 'tickers' not in st.session_state:
    st.session_state['tickers'] = [
        "PETR4.SA", "TAEE11.SA", "WEGE3.SA",
        "MGLU3.SA", "ITUB4.SA", "TAEE4.SA",
        "MXRF11.SA", "XPML11.SA"
    ]

# --- Sidebar: manage ticker universe ---
new_ticker = st.sidebar.text_input("Add new ticker (e.g., BBSE3.SA)")
if st.sidebar.button("Add"):
    symbol = new_ticker.strip().upper()
    if not symbol:
        st.sidebar.warning("Provide a ticker symbol before adding.")
    elif symbol in st.session_state['tickers']:
        st.sidebar.info(f"{symbol} is already in the list.")
    elif DataExtraction.ticker_exists(symbol):
        st.session_state['tickers'].append(symbol)
        st.sidebar.success(f"{symbol} added successfully.")
    else:
        st.sidebar.error(f"Ticker {symbol} not found. Please check the symbol and try again.")

st.sidebar.header("Asset Selection")
tickers = st.sidebar.multiselect(
    "Choose assets for analysis",
    options=st.session_state['tickers'],
    default=["PETR4.SA", "TAEE11.SA"]
)

if tickers:
    st.sidebar.header("Portfolio Parameters")
    rf_percent = st.sidebar.number_input(
        "Annual risk-free rate (%)",
        value=0.0,
        step=0.25,
        format="%.3f"
    )
    risk_free_rate = rf_percent / 100.0

    default_weight = 1.0 / len(tickers)
    manual_weights = {}
    with st.sidebar.expander("Portfolio weights (normalized automatically)", expanded=False):
        for ticker in tickers:
            manual_weights[ticker] = st.number_input(
                f"Weight {ticker}",
                min_value=0.0,
                max_value=1.0,
                value=float(default_weight),
                step=0.01,
                key=f"weight_{ticker}"
            )

    # --- Data load ---
    extraction = DataExtraction(tickers=tickers)
    price_data = extraction.extract_data()
    returns = price_data.pct_change().dropna()

    visualizations = DataVisualizations(price_data, returns)
    risk_indicators = RiskIndicators()

    volatility = risk_indicators.calculate_annualized_volatility(returns)
    var_95 = risk_indicators.calculate_parametric_var(returns)
    var_99 = risk_indicators.calculate_parametric_var(returns, confidence_level=0.99)
    historical_var_95 = risk_indicators.calculate_historical_var(returns)
    cvar_95 = risk_indicators.calculate_cvar(returns)
    sharpe_ratio = risk_indicators.calculate_sharpe_ratio(returns, risk_free_rate=risk_free_rate)
    correlation = risk_indicators.calculate_correlation(returns)

    mu, cov = PortfolioAnalytics.annualized_inputs(returns)
    weight_vector = PortfolioAnalytics.normalize_weights([manual_weights[t] for t in tickers])
    portfolio_return = PortfolioAnalytics.portfolio_return(weight_vector, mu.loc[tickers])
    portfolio_volatility = PortfolioAnalytics.portfolio_volatility(weight_vector, cov.loc[tickers, tickers])
    if portfolio_volatility == 0:
        portfolio_sharpe = 0.0
    else:
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

    max_sharpe_weights = PortfolioAnalytics.optimize_max_sharpe(
        mu.loc[tickers],
        cov.loc[tickers, tickers],
        rf=risk_free_rate
    )
    min_vol_weights = PortfolioAnalytics.optimize_min_vol(
        mu.loc[tickers],
        cov.loc[tickers, tickers]
    )

    max_sharpe_return = PortfolioAnalytics.portfolio_return(max_sharpe_weights, mu.loc[tickers])
    max_sharpe_volatility = PortfolioAnalytics.portfolio_volatility(max_sharpe_weights, cov.loc[tickers, tickers])
    max_sharpe_ratio = 0.0 if max_sharpe_volatility == 0 else (max_sharpe_return - risk_free_rate) / max_sharpe_volatility

    min_vol_return = PortfolioAnalytics.portfolio_return(min_vol_weights, mu.loc[tickers])
    min_vol_volatility = PortfolioAnalytics.portfolio_volatility(min_vol_weights, cov.loc[tickers, tickers])

    efficient_frontier = PortfolioAnalytics.efficient_frontier(
        mu.loc[tickers],
        cov.loc[tickers, tickers],
        points=50
    )

    max_sharpe_point = {
        "Return": max_sharpe_return,
        "Volatility": max_sharpe_volatility
    }
    min_vol_point = {
        "Return": min_vol_return,
        "Volatility": min_vol_volatility
    }

    tab_returns, tab_risk, tab_corr, tab_port, tab_plan = st.tabs([
        "Return Analysis",
        "Individual Risk Analysis",
        "Correlation",
        "Portfolio",
        "Financial Planning"
    ])

    with tab_returns:
        st.subheader("Return Analysis")
        with st.expander("What to watch?", expanded=False):
            st.markdown(
                "- **Historical prices** reveal the asset's trajectory over time.\n"
                "- **Daily returns** highlight short-term fluctuations.\n"
                "- **Return distribution** helps check the normality assumption used in the parametric VaR.\n"
                "- **Drawdown** measures declines relative to the most recent peak, useful for assessing prolonged loss risk."
            )
        visualizations.plot_price_series()
        col_a, col_b = st.columns(2)
        with col_a:
            visualizations.plot_daily_returns()
        with col_b:
            visualizations.plot_return_distribution()
        visualizations.plot_drawdown()

    with tab_risk:
        st.subheader("Individual Risk Analysis")
        with st.expander("How to interpret the indicators?", expanded=False):
            st.markdown(
                "- **Annualized volatility**: standard deviation of daily returns scaled to an annual basis.\n"
                "- **Parametric VaR**: worst expected loss assuming a normal distribution.\n"
                "- **Historical VaR**: worst loss observed in the historical data.\n"
                "- **CVaR / ES**: average loss when the VaR threshold is breached.\n"
                "- **Sharpe**: excess return over the risk-free rate divided by volatility."
            )
        visualizations.display_risk_indicators(
            volatility,
            var_95,
            var_99,
            historical_var_95,
            cvar_95,
            sharpe_ratio
        )
        st.caption("Rolling volatility (21 and 63 days) to monitor regime shifts.")
        col_vol_short, col_vol_long = st.columns(2)
        with col_vol_short:
            visualizations.plot_rolling_volatility(window=21)
        with col_vol_long:
            visualizations.plot_rolling_volatility(window=63)

    with tab_corr:
        st.subheader("Correlation")
        with st.expander("Why it matters?", expanded=False):
            st.markdown(
                "Correlation shows how assets move together. Low-correlation combinations tend to reduce overall portfolio risk."
            )
        visualizations.plot_correlation_heatmap(correlation)

    with tab_port:
        st.subheader("Portfolio Analysis and Optimization")
        with st.expander("How to interpret this section?", expanded=False):
            st.markdown(
                "- The provided **weights** are normalized automatically.\n"
                "- The **portfolio summary** shows annual return, risk, and Sharpe for the manual allocation.\n"
                "- The **optimized portfolios** use the Markowitz efficient frontier to maximize Sharpe or minimize risk.\n"
                "- The **efficient frontier** highlights every optimal combination for different return levels."
            )
        visualizations.display_portfolio_summary(
            weights=pd.Series(weight_vector, index=tickers),
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe=portfolio_sharpe
        )

        st.write("Optimized Portfolios")
        col_left, col_right = st.columns(2)
        with col_left:
            st.caption("Maximum Sharpe")
            st.metric("Sharpe", f"{max_sharpe_ratio:.2f}")
            st.metric("Annual return", f"{max_sharpe_return:.2%}")
            st.metric("Annual volatility", f"{max_sharpe_volatility:.2%}")
            st.dataframe(
                pd.Series(max_sharpe_weights, index=tickers, name="Weight")
                .to_frame()
                .style.format({"Weight": "{:.2%}"})
            )
        with col_right:
            st.caption("Minimum Volatility")
            st.metric("Annual return", f"{min_vol_return:.2%}")
            st.metric("Annual volatility", f"{min_vol_volatility:.2%}")
            st.dataframe(
                pd.Series(min_vol_weights, index=tickers, name="Weight")
                .to_frame()
                .style.format({"Weight": "{:.2%}"})
            )

        visualizations.plot_efficient_frontier(
            efficient_frontier,
            max_sharpe_point=max_sharpe_point,
            min_vol_point=min_vol_point
        )
        visualizations.plot_efficient_frontier_highlighted(
            efficient_frontier,
            max_sharpe_point=max_sharpe_point,
            min_vol_point=min_vol_point
        )
        efficient_frontier_only = efficient_frontier[efficient_frontier['Return'] >= min_vol_point['Return']].copy()
        if not efficient_frontier_only.empty:
            efficient_frontier_only = efficient_frontier_only.sort_values('Return', ascending=False).reset_index(drop=True)
            st.subheader("Efficient Frontier Portfolios")
            st.caption("Ordered from highest to lowest expected return.")
            weight_columns = [col for col in efficient_frontier_only.columns if col not in {'Return', 'Volatility'}]
            display_df = efficient_frontier_only[['Return', 'Volatility'] + weight_columns]
            formatters = {column: "{:.2%}" for column in ['Return', 'Volatility'] + weight_columns}
            st.dataframe(display_df.style.format(formatters))

    with tab_plan:
        st.subheader("Financial Planning - Monte Carlo")
        with st.expander("How to use this tab?", expanded=False):
            st.markdown(
                "- Enter your current situation (wealth, income, expenses) and goals.\n"
                "- The simulation runs multiple annual scenarios based on the selected portfolio.\n"
                "- Results are presented in real terms, adjusted by the stated inflation.\n"
                "- The success probability compares projected wealth with the goal implied by the withdrawal rate."
            )

        with st.form("financial_planner"):
            col_left, col_right = st.columns(2)

            with col_left:
                current_age = int(st.number_input("Current age", min_value=18, max_value=80, value=35, step=1))
                default_retirement_age = min(max(current_age + 25, current_age + 1), 100)
                retirement_age = int(
                    st.number_input(
                        "Planned retirement age",
                        min_value=current_age + 1,
                        max_value=100,
                        value=default_retirement_age,
                        step=1
                    )
                )
                initial_wealth = float(
                    st.number_input("Current wealth (BRL)", min_value=0.0, value=200000.0, step=10000.0, format="%.2f")
                )
                desired_income = float(
                    st.number_input(
                        "Desired annual retirement income (BRL)",
                        min_value=0.0,
                        value=120000.0,
                        step=5000.0,
                        format="%.2f"
                    )
                )
                contribution_growth_pct = float(
                    st.number_input(
                        "Annual contribution growth (%)",
                        min_value=0.0,
                        max_value=15.0,
                        value=0.0,
                        step=0.5,
                        format="%.2f"
                    )
                )

            with col_right:
                annual_income = float(
                    st.number_input("Current annual income (BRL)", min_value=0.0, value=180000.0, step=5000.0, format="%.2f")
                )
                annual_expenses = float(
                    st.number_input("Current annual expenses (BRL)", min_value=0.0, value=120000.0, step=5000.0, format="%.2f")
                )
                savings_rate_pct = float(
                    st.slider("Savings rate (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
                )
                extra_contribution = float(
                    st.number_input(
                        "Additional annual contribution (BRL)",
                        min_value=0.0,
                        value=0.0,
                        step=5000.0,
                        format="%.2f"
                    )
                )
                withdrawal_rate_pct = float(
                    st.number_input(
                        "Safe withdrawal rate (%)",
                        min_value=0.5,
                        max_value=10.0,
                        value=4.0,
                        step=0.1,
                        format="%.2f"
                    )
                )
                inflation_pct = float(
                    st.number_input(
                        "Expected annual inflation (%)",
                        min_value=0.0,
                        max_value=15.0,
                        value=3.0,
                        step=0.1,
                        format="%.2f"
                    )
                )
                num_simulations = int(
                    st.number_input("Number of simulations", min_value=1000, max_value=50000, value=10000, step=1000)
                )
                rng_seed_input = int(
                    st.number_input("Random seed (0 for random)", min_value=0, max_value=999_999, value=0, step=1)
                )

            savings_capacity = max(annual_income - annual_expenses, 0.0)
            annual_contribution = savings_capacity * (savings_rate_pct / 100.0) + extra_contribution
            st.caption(f"Estimated annual contribution: {format_currency(annual_contribution)}")

            submitted = st.form_submit_button("Run simulation")

        if submitted:
            errors: list[str] = []
            if retirement_age <= current_age:
                errors.append("Retirement age must be greater than current age.")
            if withdrawal_rate_pct <= 0.0:
                errors.append("Withdrawal rate must be positive.")
            if num_simulations <= 0:
                errors.append("Number of simulations must be positive.")

            if errors:
                for message in errors:
                    st.error(message)
            else:
                withdrawal_rate = withdrawal_rate_pct / 100.0
                inflation_rate = inflation_pct / 100.0
                contribution_growth = contribution_growth_pct / 100.0
                seed_value = None if rng_seed_input == 0 else rng_seed_input

                planner = MonteCarloPlanner(
                    expected_returns=mu.loc[tickers],
                    covariance=cov.loc[tickers, tickers],
                    weights=pd.Series(weight_vector, index=tickers),
                    inflation=inflation_rate
                )

                try:
                    result = planner.run_simulation(
                        current_age=current_age,
                        retirement_age=retirement_age,
                        initial_wealth=initial_wealth,
                        annual_contribution=annual_contribution,
                        desired_retirement_income=desired_income,
                        withdrawal_rate=withdrawal_rate,
                        inflation=inflation_rate,
                        n_sims=num_simulations,
                        contribution_growth=contribution_growth,
                        rng_seed=seed_value,
                    )
                except ValueError as exc:
                    st.error(f"Simulation error: {exc}")
                else:
                    final_values = pd.Series(result.wealth_paths[:, -1], name="Final wealth")
                    mean_final = float(final_values.mean())

                    st.success("Simulation completed successfully.")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    metrics_col1.metric("Probability of meeting the goal", f"{result.probability_success * 100:.1f}%")
                    metrics_col2.metric("Target wealth", format_currency(result.target_wealth))
                    metrics_col3.metric("Projected median wealth", format_currency(result.final_distribution["p50"]))

                    st.caption(f"Average final wealth: {format_currency(mean_final)}")

                    visualizations.plot_wealth_fan_chart(
                        result.percentiles,
                        title="Wealth projection (percentiles)",
                        yaxis_label="Real wealth (BRL)"
                    )

                    visualizations.plot_final_distribution(
                        final_values,
                        target=result.target_wealth,
                        title="Final wealth distribution"
                    )

                    st.write("Final wealth percentiles")
                    st.table(result.final_distribution.apply(format_currency).to_frame(name="Value"))
else:
    st.info("Select at least one ticker to begin the analysis.")
