# 📊 Market Risk Analysis – Accenture RiskControl

## 🌟 Objetivo do Projeto

Este projeto consiste em construir um pipeline simples de análise de risco de mercado, utilizando dados de ativos financeiros brasileiros. O objetivo é calcular e visualizar indicadores de risco como **volatilidade anualizada**, **VaR paramétrico a 95%** e **correlação entre ativos**, com visualização interativa em um dashboard.

## 🛠️ Ferramentas Utilizadas

* **Linguagem**: Python
* **Bibliotecas**:

  * [`yfinance`](https://pypi.org/project/yfinance/) – Coleta de dados financeiros
  * `pandas`, `numpy`, `scipy` – Manipulação de dados e cálculos estatísticos
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

## 📈 Explicação dos Cálculos

Os dados são obtidos para os últimos 6 meses via `yfinance` com base nos tickers selecionados pelo usuário. A seguir, são calculados os seguintes indicadores:

### 1. Volatilidade Anualizada

Calculada com base no desvio padrão dos retornos diários e ajustada para 252 dias úteis:

```python
vol = returns.std() * np.sqrt(252)
```

### 2. VaR Paramétrico a 95%

Baseado na suposição de retornos normalmente distribuídos. Calcula a perda máxima esperada com 95% de confiança:

```python
z_score = norm.ppf(1 - 0.95)
var = returns.mean() + returns.std() * z_score
```

### 3. Correlação

Calculada com a matriz de correlação de Pearson entre os ativos:

```python
correlation_matrix = returns.corr()
```

## 📊 Visualizações

O dashboard exibe:

* Série histórica de preços
* Retornos diários
* Tabela de indicadores de risco (volatilidade e VaR)
* Matriz de correlação

As visualizações são interativas e renderizadas com Plotly, via interface Streamlit.
