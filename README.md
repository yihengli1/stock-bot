# IBKR Stock Bot
A Python trading bot that combines automated DCF valuation, mean–variance optimization, and AI‑driven cash‑flow forecasts to generate and execute a dynamic, risk‑targeted portfolio through Interactive Brokers.

## Features

### DCF Valuation
- Uses 5‑year explicit free‑cash‑flow projections (GPT‑driven growth paths + terminal growth)
- Estimates WACC from market data (beta, debt/equity, FRED 10‑yr yield)
- Screens NASDAQ tickers for undervaluation

### Mean–Variance Optimization
- Constructs the maximum‑Sharpe “risky” sleeve via optimization (CVXPY)
- Dynamically blends in the risk‑free asset to hit a VIX‑based volatility target

### Automated Order Execution
- Connects to IB TWS/Gateway (paper or real) via ib_insync
- Fetches prices, computes allocations, places market orders
- Built‑in scheduler (schedule) to run the full rebalance workflow daily at a configured time
  
### AI Forecasting
- Pulls recent news headlines via NewsAPI
- Uses OpenAI (gpt‑4o‑mini) to forecast FCF growth rates & Terminal Growth

## 🚀 Quickstart

### 1. Clone & create venv

Clone repo locally
```
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### Configure environment
Copy .env.example to .env and fill in your keys:
```
OPENAI_API_KEY=sk‑...
FRED_API_KEY=...
NEWSAPI_KEY=...
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
```

Note*: You need an IB TWS/Gateway running and your account must:
- Disable API read only access
- Enable ActiveX and Socket Client
- Ensure sock port is the same 

### Adjust constants
In bot.py, you can tweak:
```
NASDAQ_TICKERS (your tickers)
TOTAL_CAPITAL (your portfolio)
RISK_FREE_TICKER (default "IEF")
EXECUTION_TIME (e.g. "09:30")
TIMEZONE (e.g. "America/Vancouver")
```

#### Run Program

python bot.py
