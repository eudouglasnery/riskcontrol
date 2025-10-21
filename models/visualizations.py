import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


class DataVisualizations:
    def __init__(self, data: pd.DataFrame, returns: pd.DataFrame):
        self.data = data
        self.returns = returns

    def plot_price_series(self):
        fig = go.Figure()
        for column in self.data.columns:
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data[column], mode='lines', name=column))

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Adjusted Price',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_daily_returns(self):
        fig = px.line(
            self.returns,
            title='Daily Returns',
            template='plotly_white'
        )
        fig.update_layout(xaxis_title='Date', yaxis_title='Return')
        fig.update_yaxes(tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

    def plot_return_distribution(self, bins: int = 50):
        """
        Plot histogram of daily returns to inspect distribution shape.
        """
        returns_long = self.returns.melt(var_name='Asset', value_name='Return').dropna()
        fig = px.histogram(
            returns_long,
            x='Return',
            color='Asset',
            nbins=bins,
            barmode='overlay',
            opacity=0.6,
            title='Distribution of Daily Returns',
            template='plotly_white'
        )
        fig.update_xaxes(tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

    def plot_rolling_volatility(self, window: int = 21):
        """
        Plot annualized volatility in a moving window.
        - window: number of days in the window (21 ~ 1 trading month).
        """
        rolling_vol = self.returns.rolling(window).std() * (252 ** 0.5)
        fig = px.line(
            rolling_vol,
            title=f'Rolling {window}-Day Annualized Volatility',
            template='plotly_white'
        )
        fig.update_yaxes(tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

    def plot_drawdown(self):
        """
        Plot drawdown (peak-to-trough decline) for each asset.
        """
        drawdown = self.data / self.data.cummax() - 1
        fig = px.line(
            drawdown,
            title='Historical Drawdown',
            template='plotly_white'
        )
        fig.update_layout(yaxis_title='Drawdown')
        fig.update_yaxes(tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_correlation_heatmap(correlation: pd.DataFrame):
        fig = px.imshow(
            correlation,
            text_auto=True,
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            title='Correlation Matrix',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def display_risk_indicators(volatility: pd.Series,
                                parametric_var_95: pd.Series,
                                parametric_var_99: pd.Series,
                                historical_var_95: pd.Series,
                                cvar_95: pd.Series,
                                sharpe_ratio: pd.Series):
        """
        Display the risk indicators (volatility and VaR) in a table.
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

    @staticmethod
    def display_portfolio_summary(weights: pd.Series,
                                  expected_return: float,
                                  volatility: float,
                                  sharpe: float):
        st.subheader("Portfolio Summary")
        cols = st.columns(3)
        cols[0].metric("Expected Return (annual)", f"{expected_return:.2%}")
        cols[1].metric("Volatility (annual)", f"{volatility:.2%}")
        cols[2].metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.write("Weights")
        weights_df = weights.rename("Weight").to_frame()
        st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))

    @staticmethod
    def plot_efficient_frontier(frontier: pd.DataFrame,
                                max_sharpe_point: dict | None = None,
                                min_vol_point: dict | None = None):
        fig = px.line(
            frontier,
            x='Volatility',
            y='Return',
            title='Efficient Frontier',
            template='plotly_white'
        )
        fig.update_traces(mode='lines+markers')

        if max_sharpe_point is not None:
            fig.add_trace(go.Scatter(
                x=[max_sharpe_point['Volatility']],
                y=[max_sharpe_point['Return']],
                mode='markers',
                marker=dict(size=10, color='green'),
                name='Max Sharpe'
            ))
        if min_vol_point is not None:
            fig.add_trace(go.Scatter(
                x=[min_vol_point['Volatility']],
                y=[min_vol_point['Return']],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Min Vol'
            ))
        st.plotly_chart(fig, use_container_width=True)
