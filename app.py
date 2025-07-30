import streamlit as st

from models.data_extraction import DataExtraction
from models.indicators import (
    calculate_annualized_volatility,
    calculate_parametric_var,
    calculate_correlation
)
from models.visualizations import DataVisualizations

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

novo = st.sidebar.text_input("Novo ticker")
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

# --- Data load ---
if tickers:
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
    volatility = calculate_annualized_volatility(returns)
    var_95 = calculate_parametric_var(returns)
    var_99 = calculate_parametric_var(returns, confidence_level=0.99)
    correlation = calculate_correlation(returns)

    visualizations.display_risk_indicators(volatility, var_95, var_99)

    st.subheader("🔗 Correlation Matrix")
    visualizations.plot_correlation_heatmap(correlation)
else:
    st.info("Please select at least one ticker to begin.")
