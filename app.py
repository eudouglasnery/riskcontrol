import streamlit as st

from models.data_extraction import DataExtraction
from models.indicators import RiskIndicators
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
    sharpe_ratio = risk_indicators.calculate_sharpe_ratio(returns, 0.1452)
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
else:
    st.info("Please select at least one ticker to begin.")
