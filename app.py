import streamlit as st
import pandas as pd

from models.data_extraction import DataExtraction
from models.indicators import RiskIndicators
from models.visualizations import DataVisualizations
from models.portfolio import PortfolioAnalytics

# --- Page config ---
st.set_page_config(page_title="Market Risk Dashboard", layout="wide")

# --- Title ---
st.title("📊 Market Risk Analysis")

# --- New ticker field ---
if 'tickers' not in st.session_state:
    st.session_state['tickers'] = [
        "PETR4.SA", "TAEE11.SA", "WEGE3.SA",
        "MGLU3.SA", "ITUB4.SA", "TAEE4.SA",
        "MXRF11.SA", "XPML11.SA"
    ]

novo = st.sidebar.text_input("New ticker \n\nExemple: BBSE3.SA")
if st.sidebar.button("Add"):
    t = novo.strip().upper()
    if t and t not in st.session_state['tickers']:
        st.session_state['tickers'].append(t)

# --- Sidebar ---
st.sidebar.header("Asset Selection")
tickers = st.sidebar.multiselect(
    "",
    options=st.session_state['tickers'],
    default=["PETR4.SA", "TAEE11.SA"]
)

if tickers:
    # --- Data load ---
    extraction = DataExtraction(tickers=tickers)
    data = extraction.extract_data()
    returns = data.pct_change().dropna()

    # --- Visualizations ---
    visualizations = DataVisualizations(data, returns)
    st.subheader("📈 Historical Prices")
    visualizations.plot_price_series()

    st.subheader("📈 Return & Volatility Analysis")
    visualizations.plot_daily_returns()
    visualizations.plot_rolling_volatility()
    visualizations.plot_rolling_volatility(window=63)

    # --- Risk indicators ---
    risk_indicators = RiskIndicators()
    volatility = risk_indicators.calculate_annualized_volatility(returns)
    var_95 = risk_indicators.calculate_parametric_var(returns)
    var_99 = risk_indicators.calculate_parametric_var(returns, confidence_level=0.99)
    historical_var_95 = risk_indicators.calculate_historical_var(returns)
    cvar_95 = risk_indicators.calculate_cvar(returns)
    sharpe_ratio = risk_indicators.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    correlation = risk_indicators.calculate_correlation(returns)

    visualizations.display_risk_indicators(
        volatility,
        var_95,
        var_99,
        historical_var_95,
        cvar_95,
        sharpe_ratio
    )

    st.subheader("🔗 Correlation Matrix")
    visualizations.plot_correlation_heatmap(correlation)

    # --- Portfolio Analysis ---
    st.subheader("📊 Portfolio Analysis")
    st.sidebar.header("Portfolio")
    rf_percent = st.sidebar.number_input("Risk-free (annual %)", value=0.0, step=0.25, format="%f")
    rf = rf_percent / 100.0

    # Weight inputs
    default_w = {t: 1.0 / len(tickers) for t in tickers}
    weights_input = {}
    with st.sidebar.expander("Weights (sum will be normalized)"):
        for t in tickers:
            weights_input[t] = st.number_input(f"{t}", value=float(default_w[t]), min_value=0.0, max_value=1.0, step=0.01)

    # Normalize weights and compute portfolio metrics
    w_vec = PortfolioAnalytics.normalize_weights([weights_input[t] for t in tickers])
    mu, cov = PortfolioAnalytics.annualized_inputs(returns)

    p_ret = PortfolioAnalytics.portfolio_return(w_vec, mu)
    p_vol = PortfolioAnalytics.portfolio_volatility(w_vec, cov)
    p_sharpe = 0.0 if p_vol == 0 else (p_ret - rf) / p_vol

    visualizations.display_portfolio_summary(
        weights=pd.Series(w_vec, index=tickers),
        expected_return=p_ret,
        volatility=p_vol,
        sharpe=p_sharpe
    )

    # Optimizations
    w_max_sharpe = PortfolioAnalytics.optimize_max_sharpe(mu.loc[tickers], cov.loc[tickers, tickers], rf=rf)
    w_min_vol = PortfolioAnalytics.optimize_min_vol(mu.loc[tickers], cov.loc[tickers, tickers])

    ms_ret = PortfolioAnalytics.portfolio_return(w_max_sharpe, mu.loc[tickers])
    ms_vol = PortfolioAnalytics.portfolio_volatility(w_max_sharpe, cov.loc[tickers, tickers])
    mv_ret = PortfolioAnalytics.portfolio_return(w_min_vol, mu.loc[tickers])
    mv_vol = PortfolioAnalytics.portfolio_volatility(w_min_vol, cov.loc[tickers, tickers])

    st.write("Optimized Portfolios")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Max Sharpe Weights")
        st.dataframe(pd.Series(w_max_sharpe, index=tickers, name="Weight").to_frame().T.style.format("{:.2%}"))
        st.metric("Max Sharpe", f"{((ms_ret - rf)/ms_vol):.2f}")
    with col2:
        st.caption("Min Vol Weights")
        st.dataframe(pd.Series(w_min_vol, index=tickers, name="Weight").to_frame().T.style.format("{:.2%}"))
        st.metric("Min Volatility", f"{mv_vol:.2%}")

    # Efficient Frontier
    ef = PortfolioAnalytics.efficient_frontier(mu.loc[tickers], cov.loc[tickers, tickers], points=30)
    max_sharpe_point = {"Return": ms_ret, "Volatility": ms_vol}
    min_vol_point = {"Return": mv_ret, "Volatility": mv_vol}
    visualizations.plot_efficient_frontier(ef, max_sharpe_point, min_vol_point)
else:
    st.info("Please select at least one ticker to begin.")
