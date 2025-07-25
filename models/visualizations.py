import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


def plot_price_series(prices: pd.DataFrame):
    fig = go.Figure()
    for column in prices.columns:
        fig.add_trace(go.Scatter(x=prices.index, y=prices[column], mode='lines', name=column))

    fig.update_layout(title='Price Series',
                      xaxis_title='Date',
                      yaxis_title='Adjusted Price',
                      template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)


def plot_daily_returns(returns: pd.DataFrame):
    fig = px.line(returns, title='Daily Returns')
    fig.update_layout(xaxis_title='Date', yaxis_title='Return', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_heatmap(correlation: pd.DataFrame):
    fig = px.imshow(correlation,
                    text_auto=True,
                    title='Correlation Matrix',
                    color_continuous_scale='RdBu',
                    zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)


def display_risk_indicators(volatility: pd.Series, var: pd.Series):
    """
    Displays the risk indicators (volatility and VaR) in a table.
    """
    risk_df = pd.DataFrame({
        'Annualized Volatility': volatility,
        'Parametric VaR (95%)': var
    })
    st.subheader("Risk Indicators")
    st.dataframe(risk_df.style.format("{:.2%}"))
