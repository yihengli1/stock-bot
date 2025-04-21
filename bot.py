import numpy as np
import pandas as pd
import cvxpy as cp
from ib_insync import IB, Stock, util, Order
import requests
import os
from dotenv import load_dotenv
import yfinance as yf

# Loading Environment Variables
load_dotenv()

ib = IB()

# Setup IB Connection
# 7496: REAL | 7497: PAPER
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)
print("Connected:", ib.isConnected())

# TICKERS
nasdaq_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# RFA
risk_free_ticker = 'IEF'

tickers = nasdaq_tickers + [risk_free_ticker]


def fetchRiskFreeRate():
    api_key = os.getenv("FRED_API_KEY")
    series_id = 'DGS10'  # 10-Year Treasury Constant Maturity Rate
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=desc&limit=1'

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        latest_observation = data['observations'][0]
        date = latest_observation['date']
        value = latest_observation['value']
        print(f"Date: {date}, 10-Year Treasury Yield/Risk Free Rate: {value}%")
        return value
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return

# VIX-Based Dynamic Volatility Targeting


def get_vix_level():
    vix = yf.download('^VIX', period='5d', interval='1d')
    if vix.empty:
        raise Exception("Could not fetch VIX data.")
    return vix['Close'].iloc[-1]


def get_target_volatility(vix_level):
    if vix_level < 15:
        return 0.20
    elif vix_level < 25:
        return 0.12
    else:
        return 0.07

# Fetch historical price data (close prices) for a symbol.


def fetch_historical_data(symbol, duration='1 Y', barSize='1 day'):

    contract = Stock(symbol, 'SMART', 'USD')
    # Historical data: use endDateTime='' to get latest available data.
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=barSize,
        whatToShow='MIDPOINT',
        useRTH=True,
        formatDate=1
    )
    # Convert the returned bar data into a pandas DataFrame.
    df = util.df(bars)
    df.set_index('date', inplace=True)
    return df['close']

# For each ticker, fetch historical close prices, compute daily returns


def get_return_stats(tickers):
    returns_data = {}

    for ticker in tickers:
        try:
            close_prices = fetch_historical_data(ticker)
            # Calculate daily returns
            returns = close_prices.pct_change().dropna()
            returns_data[ticker] = returns
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")
            continue

    # Combine returns into a DataFrame
    returns_df = pd.DataFrame(returns_data)
    # Expected annualized returns approximated from daily returns (approx 252 trading days/year)
    exp_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    return exp_returns, cov_matrix

# Valuation Functions (Relative Valuation and DCF)


# Relative Valuation
def relative_valuation(symbol):
    return 1

# DCF Valuation


def dcf_valuation(symbol):
    return 1


# Composite valuation
def composite_valuation(symbol, weight_relative=0.5, weight_dcf=0.5):
    rel_val = relative_valuation(symbol)
    dcf_val = dcf_valuation(symbol)
    composite_score = weight_relative * rel_val + weight_dcf * dcf_val
    return composite_score

# Efficient Frontiers!!


def optimize_on_cml(exp_returns, cov_matrix, risk_free_rate, min_weight=0.0, max_weight=0.5):
    """
    Maximize Sharpe Ratio: Place portfolio on the Capital Market Line.
    Returns: (risky_asset_weights, weight_in_risk_free)
    """
    n = len(exp_returns)
    w = cp.Variable(n)

    excess_returns = exp_returns.values - \
        float(risk_free_rate) / 100  # Convert % to decimal
    portfolio_return = excess_returns @ w
    portfolio_risk = cp.sqrt(cp.quad_form(w, cov_matrix.values))

    # Objective: maximize Sharpe ratio = excess return / volatility
    # Or equivalently, maximize (return / risk)
    objective = cp.Maximize(portfolio_return / portfolio_risk)

    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None:
        raise Exception("Optimization failed")

    risky_weights = pd.Series(w.value, index=exp_returns.index)
    return risky_weights


def optimize_max_sharpe(risky_returns, cov_matrix):
    n = len(risky_returns)
    w = cp.Variable(n)
    excess_returns = risky_returns.values
    port_return = excess_returns @ w
    port_risk = cp.sqrt(cp.quad_form(w, cov_matrix.values))
    objective = cp.Maximize(port_return / port_risk)
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if w.value is None:
        raise Exception("Optimization failed")
    return pd.Series(w.value, index=risky_returns.index)


def blend_with_risk_free(risky_weights, risky_returns, cov_matrix, rf_return, target_vol=0.12):
    risky_vol = np.sqrt(risky_weights.values @
                        cov_matrix.values @ risky_weights.values)
    leverage = target_vol / risky_vol
    final_risky_weights = risky_weights * leverage
    rf_weight = 1 - leverage
    combined = final_risky_weights.append(
        pd.Series({risk_free_ticker: rf_weight}))
    return combined


# Selection
def select_stocks(tickers, valuation_threshold=1.0):
    return tickers
    # selected = []
    # valuation_scores = {}
    # for ticker in tickers:
    #     score = composite_valuation(ticker)
    #     valuation_scores[ticker] = score
    #     if score < valuation_threshold:
    #         selected.append(ticker)
    # print("Valuation scores:", valuation_scores)
    # return selected


# Placing orders
def place_orders(weights, total_investment=100):
    for ticker, weight in weights.items():
        # Compute dollar allocation and then approximate share quantity (you may want to retrieve live prices)
        allocation = total_investment * weight
        # For a realistic scenario, get the latest market price:
        contract = Stock(ticker, 'SMART', 'USD')
        market_data = ib.reqMktData(contract, '', False, False)
        ib.sleep(2)  # allow a brief pause to collect data
        try:
            # Use last price from market data
            price = float(market_data.last)
        except Exception:
            price = 0
        if price <= 0:
            print(f"Skipping {ticker} due to invalid price data")
            continue
        shares = int(allocation // price)
        if shares <= 0:
            print(f"Not enough allocation for {ticker} to buy any shares.")
            continue
        order = Order()
        order.action = "BUY"
        order.orderType = "MKT"
        order.totalQuantity = shares
        trade = ib.placeOrder(contract, order)
        print(
            f"Placed order for {shares} shares of {ticker} at approx ${price:.2f}")


def main():
    # VIX
    print("Fetching VIX and setting target volatility...")
    try:
        vix = get_vix_level()
        print(f"VIX: {vix:.2f}")
        target_vol = get_target_volatility(vix)
        print(f"Target volatility: {target_vol:.2%}")
    except Exception as e:
        print("Failed to fetch VIX:", e)
        target_vol = 0.12

    # Stock selection based on valuation
    selected_stocks = select_stocks(nasdaq_tickers, valuation_threshold=1.0)
    print("Selected stocks after valuation filter:", selected_stocks)

    if not selected_stocks:
        print("No stocks qualify based on the valuation criteria.")
        return

    # Obtain return statistics (expected returns and covariances) for selected stocks
    exp_returns, cov_matrix = get_return_stats(selected_stocks)
    print("Expected Annualized Returns:\n", exp_returns)
    print("Covariance Matrix:\n", cov_matrix)

    fetchRiskFreeRate()

    rf_rate = float(fetchRiskFreeRate())
    opt_weights = optimize_on_cml(exp_returns, cov_matrix, rf_rate)
    print("Optimized Risky Asset Weights:\n", opt_weights)

    # Risk-free asset weight is 1 - sum(risky_weights)
    rf_weight = 1 - opt_weights.sum()
    print(f"Allocate {rf_weight:.2%} to the risk-free asset (e.g., Treasury)")

    # # Place trade orders based on optimized weights
    # place_orders(opt_weights)


if __name__ == "__main__":
    main()
    ib.disconnect()
    print("Disconnected from IB.")
