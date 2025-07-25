import streamlit as st

from models.data_extraction import extract_data
from models.indicators import (
    calculate_annualized_volatility,
    calculate_parametric_var,
    calculate_correlation
)
from models.visualizations import (
    plot_price_series,
    plot_daily_returns,
    plot_correlation_heatmap,
    display_risk_indicators
)

# --- Page config ---
st.set_page_config(page_title="Market Risk Dashboard", layout="wide")

# --- Title ---
st.title("📊 Market Risk Analysis")

# --- Sidebar ---
st.sidebar.header("Asset Selection")
tickers = st.sidebar.multiselect(
    "",
    options=["PETR4.SA", "TAEE11.SA", "WEGE3.SA", "MGLU3.SA", "ITUB4.SA"],
    default=["PETR4.SA", "TAEE11.SA"]
)

# --- Data load ---
if tickers:
    data = extract_data(tickers)
    
    returns = data.pct_change().dropna()

    # --- Visualizations ---
    st.subheader("📈 Historical Prices")
    plot_price_series(data)

    st.subheader("📉 Daily Returns")
    plot_daily_returns(returns)

    # --- Risk indicators ---
    volatility = calculate_annualized_volatility(returns)
    var_95 = calculate_parametric_var(returns)
    correlation = calculate_correlation(returns)

    display_risk_indicators(volatility, var_95)

    st.subheader("🔗 Correlation Matrix")
    plot_correlation_heatmap(correlation)
else:
    st.info("Please select at least one ticker to begin.")
