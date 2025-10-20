import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


class DataVisualizations:
    def __init__(self, data: pd.DataFrame, returns: pd.DataFrame):
        self.data = data
        self.returns = returns

    def plot_price_series(self):
        prices = self.data
        fig = go.Figure()
        for column in prices.columns:
            fig.add_trace(go.Scatter(x=prices.index, y=prices[column], mode='lines', name=column))

        fig.update_layout(xaxis_title='Date',
                          yaxis_title='Adjusted Price',
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    def plot_daily_returns(self):
        fig = px.line(self.returns, title='Daily Returns')
        fig.update_layout(xaxis_title='Date', yaxis_title='Return', template='plotly_white')
        fig.update_yaxes(tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

    def plot_rolling_volatility(self, window: int = 21):
        """
        Plots annualized volatility in a moving window.
        - window: number of days in the window (21 ≈ 1 trading month).
        """
        rolling_vol = self.returns.rolling(window).std() * (252 ** 0.5)
        fig = px.line(rolling_vol,
                      title=f'Rolling {window}-Day Annualized Volatility')
        fig.update_yaxes(tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_correlation_heatmap(correlation: pd.DataFrame):
        fig = px.imshow(correlation,
                        text_auto=True,
                        color_continuous_scale='RdBu',
                        zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def display_risk_indicators(volatility: pd.Series,
                                parametric_var_95: pd.Series,
                                parametric_var_99: pd.Series,
                                historical_var_95: pd.Series,
                                cvar_95: pd.Series,
                                sharpe_ratio: pd.Series):
        """
        Displays the risk indicators (volatility and VaR) in a table.
        """
        risk_df = pd.DataFrame({
            'Annualized Volatility': volatility,
            'Parametric VaR (95%)': parametric_var_95,
            'Parametric VaR (99%)': parametric_var_99,
            'Historical VaR (95%)': historical_var_95,
            'CVaR / ES (95%)': cvar_95,
            'Sharpe Ratio': sharpe_ratio
        })
        st.subheader("Risk Indicators")
        formatters = {
            'Annualized Volatility': "{:.2%}",
            'Parametric VaR (95%)': "{:.2%}",
            'Parametric VaR (99%)': "{:.2%}",
            'Historical VaR (95%)': "{:.2%}",
            'CVaR / ES (95%)': "{:.2%}",
            'Sharpe Ratio': "{:.2f}"
        }
        st.dataframe(risk_df.style.format(formatters))
