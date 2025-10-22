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

    tab_returns, tab_risk, tab_corr, tab_port, tab_plan = st.tabs([
        "Analise de Retornos",
        "Analise de Risco Individual",
        "Correlacao",
        "Portfolio",
        "Planejamento Financeiro"
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

    with tab_plan:
        st.subheader("Planejamento Financeiro - Monte Carlo")
        with st.expander("Como usar esta aba?", expanded=False):
            st.markdown(
                "- Informe sua situacao atual (patrimonio, renda, despesas) e metas.\n"
                "- A simulacao roda multiplos cenarios anuais com base no portfolio selecionado.\n"
                "- Resultados sao apresentados em valores reais, ajustados pela inflacao informada.\n"
                "- A probabilidade de sucesso compara o patrimonio projetado com a meta baseada na taxa de retirada."
            )

        with st.form("financial_planner"):
            col_left, col_right = st.columns(2)

            with col_left:
                current_age = int(st.number_input("Idade atual", min_value=18, max_value=80, value=35, step=1))
                default_retirement_age = min(max(current_age + 25, current_age + 1), 100)
                retirement_age = int(
                    st.number_input(
                        "Idade planejada para aposentadoria",
                        min_value=current_age + 1,
                        max_value=100,
                        value=default_retirement_age,
                        step=1
                    )
                )
                initial_wealth = float(
                    st.number_input("Patrimonio atual (R$)", min_value=0.0, value=200000.0, step=10000.0, format="%.2f")
                )
                desired_income = float(
                    st.number_input(
                        "Renda anual desejada na aposentadoria (R$)",
                        min_value=0.0,
                        value=120000.0,
                        step=5000.0,
                        format="%.2f"
                    )
                )
                contribution_growth_pct = float(
                    st.number_input(
                        "Crescimento anual das contribuicoes (%)",
                        min_value=0.0,
                        max_value=15.0,
                        value=0.0,
                        step=0.5,
                        format="%.2f"
                    )
                )

            with col_right:
                annual_income = float(
                    st.number_input("Renda anual atual (R$)", min_value=0.0, value=180000.0, step=5000.0, format="%.2f")
                )
                annual_expenses = float(
                    st.number_input("Despesas anuais atuais (R$)", min_value=0.0, value=120000.0, step=5000.0, format="%.2f")
                )
                savings_rate_pct = float(
                    st.slider("Percentual da sobra destinado a poupanca (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
                )
                extra_contribution = float(
                    st.number_input(
                        "Contribuicao anual adicional (R$)",
                        min_value=0.0,
                        value=0.0,
                        step=5000.0,
                        format="%.2f"
                    )
                )
                withdrawal_rate_pct = float(
                    st.number_input(
                        "Taxa de retirada segura (%)",
                        min_value=0.5,
                        max_value=10.0,
                        value=4.0,
                        step=0.1,
                        format="%.2f"
                    )
                )
                inflation_pct = float(
                    st.number_input(
                        "Inflacao anual esperada (%)",
                        min_value=0.0,
                        max_value=15.0,
                        value=3.0,
                        step=0.1,
                        format="%.2f"
                    )
                )
                num_simulations = int(
                    st.number_input("Numero de simulacoes", min_value=1000, max_value=50000, value=10000, step=1000)
                )
                rng_seed_input = int(
                    st.number_input("Semente aleatoria (0 para aleatorio)", min_value=0, max_value=999_999, value=0, step=1)
                )

            savings_capacity = max(annual_income - annual_expenses, 0.0)
            annual_contribution = savings_capacity * (savings_rate_pct / 100.0) + extra_contribution
            st.caption(f"Aporte anual estimado: {format_currency(annual_contribution)}")

            submitted = st.form_submit_button("Rodar simulacao")

        if submitted:
            errors: list[str] = []
            if retirement_age <= current_age:
                errors.append("A idade de aposentadoria deve ser maior que a idade atual.")
            if withdrawal_rate_pct <= 0.0:
                errors.append("A taxa de retirada deve ser positiva.")
            if num_simulations <= 0:
                errors.append("Numero de simulacoes deve ser positivo.")

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
                    st.error(f"Erro na simulacao: {exc}")
                else:
                    final_values = pd.Series(result.wealth_paths[:, -1], name="Final wealth")
                    mean_final = float(final_values.mean())

                    st.success("Simulacao concluida com sucesso.")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    metrics_col1.metric("Probabilidade de atingir a meta", f"{result.probability_success * 100:.1f}%")
                    metrics_col2.metric("Meta de patrimonio", format_currency(result.target_wealth))
                    metrics_col3.metric("Patrimonio mediano projetado", format_currency(result.final_distribution["p50"]))

                    st.caption(f"Media do patrimonio final: {format_currency(mean_final)}")

                    visualizations.plot_wealth_fan_chart(
                        result.percentiles,
                        title="Projecao do patrimonio (percentis)",
                        yaxis_label="Patrimonio real (R$)"
                    )

                    visualizations.plot_final_distribution(
                        final_values,
                        target=result.target_wealth,
                        title="Distribuicao do patrimonio final"
                    )

                    st.write("Percentis do patrimonio final")
                    st.table(result.final_distribution.apply(format_currency).to_frame(name="Valor"))
else:
    st.info("Selecione pelo menos um ticker para iniciar a analise.")
