# 📊 Market Risk Analysis – Accenture RiskControl

## 🌟 Project Objective

Build a simple market-risk analysis pipeline for Brazilian assets, offering an interactive dashboard that calculates and displays key risk indicators.

## 🛠️ Tools Used

* **Python** 3.12
* **Libraries**

  * [`yfinance`](https://pypi.org/project/yfinance/) – Financial data collection
  * `pandas`, `numpy`, `scipy`, `python-dateutil` – Data handling and statistics
  * `plotly`, `streamlit` – Interactive visualization
* **Other**: Git (version control)

## ▶️ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-user/riskcontrol.git
   cd riskcontrol
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the dashboard:

   ```bash
   streamlit run app.py
   ```

## ⚙️ Key Features

| Functionality              | Description                                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Download & cache**       | Daily prices for the last 6 months are pulled via `yfinance` and cached in `tickers_data.csv`. Subsequent runs fetch only missing tickers. |
| **Manual ticker addition** | A sidebar field lets you add any B3 symbol (e.g., `BBSE3.SA`) on the fly.                                                                  |
| **Risk indicators**        | • Annualized volatility  • Parametric VaR 95 % **and** 99 %  • Correlation matrix                                                          |
| **Rolling volatility**     | Charts for 21-day (≈ 1 month) and 63-day (≈ 3 months) windows.                                                                             |
| **Interactive dashboard**  | All charts and tables are dynamic (Plotly + Streamlit).                                                                                    |

## 📈 Calculation Explained

### 1. Annualized Volatility

Computed from the standard deviation of daily returns, scaled to 252 business days:

```python
vol = returns.std() * np.sqrt(252)
```

### 2. Parametric VaR (95 % and 99 %)

Assuming normally distributed returns, it estimates the maximum expected loss at the 95 % and 99 % confidence levels:

```python
z_score = norm.ppf(1 - confiance)
var = returns.mean() + returns.std() * z_score
```

### 3. Correlation

Calculated via the Pearson correlation matrix of the assets:

```python
correlation_matrix = returns.corr()
```

## 📊 Visualizations

The dashboard displays:

* **Historical price series**
* **Daily returns**
* **Rolling volatility 21 d & 63 d**
* **Risk‐indicator table**: Vol, VaR 95 %, VaR 99 %
* **Correlation matrix**
