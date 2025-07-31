# 📊 Market Risk Analysis – Accenture RiskControl

## 🌟 Objetivo do Projeto

Construir um pipeline simples de análise de risco de mercado para ativos brasileiros, oferecendo um dashboard interativo que calcula e exibe indicadores de risco.

## 🛠️ Ferramentas Utilizadas

* **Python** 3.12
* **Bibliotecas**:

  * [`yfinance`](https://pypi.org/project/yfinance/) – Coleta de dados financeiros
  * `pandas`, `numpy`, `scipy`, `python‑dateutil` – Tratamento de dados e estatística
  * `plotly`, `streamlit` – Visualização interativa
* **Outras**: Git (controle de versão)

## ▶️ Como Executar

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/riskcontrol.git
   cd riskcontrol
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Execute o dashboard:

   ```bash
   streamlit run app.py
   ```

## ⚙️ Funcionalidades Principais

| Funcionalidade               | Descrição                                                                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Download & cache**         | Preços diários dos últimos 6 meses são baixados via `yfinance` e cacheados em `tickers_data.csv`. Downloads posteriores só buscam novos tickers. |
| **Adição manual de tickers** | Campo na sidebar permite incluir qualquer código B3 (ex.: `BBSE3.SA`) em tempo real.                                                             |
| **Indicadores de risco**     | • Volatilidade anualizada  • VaR paramétrico 95 % **e** 99 %  • Matriz de correlação                                                             |
| **Rolling volatility**       | Gráficos para janelas de 21 dias (≈1 mês) e 63 dias (≈3 meses).                                                                                  |
| **Dashboard interativo**     | Todos os gráficos e tabelas são dinâmicos (Plotly + Streamlit).  

## 📈 Explicação dos Cálculos

### 1. Volatilidade Anualizada

Calculada com base no desvio padrão dos retornos diários e ajustada para 252 dias úteis:

```python
vol = returns.std() * np.sqrt(252)
```

### 2. VaR Paramétrico (95 % e 99 %)

Baseado na suposição de retornos normalmente distribuídos. Calcula a perda máxima esperada com 95% e 99% de confiança:

```python
z_score = norm.ppf(1 - confiance)  # confidence = 0.95 ou 0.99
var = returns.mean() + returns.std() * z_score
```

### 3. Correlação

Calculada com a matriz de correlação de Pearson entre os ativos:

```python
correlation_matrix = returns.corr()
```

## 📊 Visualizações

O dashboard exibe:

* **Série histórica de preços**
* **Retornos diários** 
* **Rolling volatility 21 d & 63 d** 
* **Tabela de indicadores**: Vol, VaR 95 %, VaR 99 %
* **Matriz de correlação** 
