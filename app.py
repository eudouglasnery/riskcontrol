import pandas as pd
import streamlit as st

from models.data_extraction import DataExtraction
from models.indicators import RiskIndicators
from models.visualizations import DataVisualizations
from models.portfolio import PortfolioAnalytics


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
new_ticker = st.sidebar.text_input("Adicionar novo ticker (ex: BBSE3.SA)")
if st.sidebar.button("Adicionar"):
    symbol = new_ticker.strip().upper()
    if symbol and symbol not in st.session_state['tickers']:
        st.session_state['tickers'].append(symbol)

st.sidebar.header("Selecao de Ativos")
tickers = st.sidebar.multiselect(
    "Escolha os ativos para analise",
    options=st.session_state['tickers'],
    default=["PETR4.SA", "TAEE11.SA"]
)

if tickers:
    st.sidebar.header("Parametros do Portfolio")
    rf_percent = st.sidebar.number_input(
        "Taxa livre de risco anual (%)",
        value=0.0,
        step=0.25,
        format="%.3f"
    )
    risk_free_rate = rf_percent / 100.0

    default_weight = 1.0 / len(tickers)
    manual_weights = {}
    with st.sidebar.expander("Pesos do portfolio (normalizados automaticamente)", expanded=False):
        for ticker in tickers:
            manual_weights[ticker] = st.number_input(
                f"Peso {ticker}",
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
        points=30
    )

    tab_returns, tab_risk, tab_corr, tab_port = st.tabs([
        "Analise de Retornos",
        "Analise de Risco Individual",
        "Correlacao",
        "Portfolio"
    ])

    with tab_returns:
        st.subheader("Analise de Retornos")
        with st.expander("O que observar?", expanded=False):
            st.markdown(
                "- **Precos historicos** mostram a trajetoria do ativo ao longo do tempo.\n"
                "- **Retornos diarios** evidenciam oscilacoes de curto prazo.\n"
                "- **Distribuicao de retornos** ajuda a checar a hipotese de normalidade usada no VaR parametrico.\n"
                "- **Drawdown** quantifica quedas em relacao ao ultimo pico, util para avaliar risco de perda prolongada."
            )
        visualizations.plot_price_series()
        col_a, col_b = st.columns(2)
        with col_a:
            visualizations.plot_daily_returns()
        with col_b:
            visualizations.plot_return_distribution()
        visualizations.plot_drawdown()

    with tab_risk:
        st.subheader("Analise de Risco Individual")
        with st.expander("Como interpretar os indicadores?", expanded=False):
            st.markdown(
                "- **Volatilidade anualizada**: desvio padrao dos retornos diarios convertido para base anual.\n"
                "- **VaR parametrico**: pior perda esperada assumindo distribuicao normal.\n"
                "- **VaR historico**: pior perda baseada nos dados observados.\n"
                "- **CVaR / ES**: perda media quando o VaR e ultrapassado.\n"
                "- **Sharpe**: retorno excedente sobre a taxa livre de risco dividido pela volatilidade."
            )
        visualizations.display_risk_indicators(
            volatility,
            var_95,
            var_99,
            historical_var_95,
            cvar_95,
            sharpe_ratio
        )
        st.caption("Volatilidade movel (21 e 63 dias) para acompanhar mudancas de regime.")
        col_vol_short, col_vol_long = st.columns(2)
        with col_vol_short:
            visualizations.plot_rolling_volatility(window=21)
        with col_vol_long:
            visualizations.plot_rolling_volatility(window=63)

    with tab_corr:
        st.subheader("Correlacao")
        with st.expander("Por que importa?", expanded=False):
            st.markdown(
                "Correlacao mostra a co-movimentacao entre ativos. Combinacoes com baixa correlacao tendem a reduzir o risco total do portfolio."
            )
        visualizations.plot_correlation_heatmap(correlation)

    with tab_port:
        st.subheader("Analise e Otimizacao de Portfolio")
        with st.expander("Como interpretar esta secao?", expanded=False):
            st.markdown(
                "- Os **pesos informados** sao normalizados automaticamente.\n"
                "- O **resumo da carteira** mostra retorno, risco e Sharpe anuais da alocacao manual.\n"
                "- Os **portfolios otimizados** usam a fronteira eficiente de Markowitz para maximizar Sharpe ou minimizar risco.\n"
                "- A **fronteira eficiente** destaca todas as combinacoes otimas para diferentes niveis de retorno."
            )
        visualizations.display_portfolio_summary(
            weights=pd.Series(weight_vector, index=tickers),
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe=portfolio_sharpe
        )

        st.write("Portfolios Otimizados")
        col_left, col_right = st.columns(2)
        with col_left:
            st.caption("Maximo Sharpe")
            st.metric("Sharpe", f"{max_sharpe_ratio:.2f}")
            st.metric("Retorno anual", f"{max_sharpe_return:.2%}")
            st.metric("Volatilidade anual", f"{max_sharpe_volatility:.2%}")
            st.dataframe(
                pd.Series(max_sharpe_weights, index=tickers, name="Peso")
                .to_frame()
                .style.format({"Peso": "{:.2%}"})
            )
        with col_right:
            st.caption("Minima Volatilidade")
            st.metric("Retorno anual", f"{min_vol_return:.2%}")
            st.metric("Volatilidade anual", f"{min_vol_volatility:.2%}")
            st.dataframe(
                pd.Series(min_vol_weights, index=tickers, name="Peso")
                .to_frame()
                .style.format({"Peso": "{:.2%}"})
            )

        visualizations.plot_efficient_frontier(
            efficient_frontier,
            max_sharpe_point={"Return": max_sharpe_return, "Volatility": max_sharpe_volatility},
            min_vol_point={"Return": min_vol_return, "Volatility": min_vol_volatility}
        )
else:
    st.info("Selecione pelo menos um ticker para iniciar a analise.")
