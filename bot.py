import numpy as np
import pandas as pd
import cvxpy as cp
import datetime
from ib_insync import IB, Stock, util, Order

# Create an IB instance
ib = IB()

# Setup IB Connection

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)
print("Connected:", ib.isConnected())

# TICKERS
nasdaq_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']\


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

# -----------------------------------------------------------
# 5. Valuation Functions (Relative Valuation and DCF)
# -----------------------------------------------------------

# Relative Valuation


def relative_valuation(symbol):

    return

# DCF Valuation


def dcf_valuation(symbol):

    return


# Composite valuation
def composite_valuation(symbol, weight_relative=0.5, weight_dcf=0.5):
    rel_val = relative_valuation(symbol)
    dcf_val = dcf_valuation(symbol)
    composite_score = weight_relative * rel_val + weight_dcf * dcf_val
    return composite_score

# Efficient Frontiers!!


def optimize_portfolio(exp_returns, cov_matrix, min_weight=0.0, max_weight=0.5):
    """
    Use a mean-variance optimization (Markowitz model) to compute optimal weights.
    Constraint: the sum of weights equals 1 and each weight lies within [min_weight, max_weight].
    """
    n = len(exp_returns)
    w = cp.Variable(n)  # portfolio weights vector

    # For illustration, we maximize risk-adjusted return
    # We choose a risk aversion parameter lambda (this can be tuned)
    risk_aversion = 0.5

    portfolio_return = exp_returns.values @ w
    portfolio_risk = cp.quad_form(w, cov_matrix.values)
    # The objective is to maximize (return - risk_aversion * risk); equivalently, minimize negative of that:
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)

    # Define constraints: weights sum to 1; each weight between min and max
    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None:
        raise Exception("Optimization did not converge.")

    # Create a series for optimal weights
    opt_weights = pd.Series(w.value, index=exp_returns.index)
    return opt_weights

# Selection
def select_stocks(tickers, valuation_threshold=1.0):
    selected = []
    valuation_scores = {}
    for ticker in tickers:
        score = composite_valuation(ticker)
        valuation_scores[ticker] = score
        if score < valuation_threshold:
            selected.append(ticker)
    print("Valuation scores:", valuation_scores)
    return selected



# Placing orders
def place_orders(weights, total_investment=0.5):
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

    # Run portfolio optimization on selected stocks
    try:
        opt_weights = optimize_portfolio(
            exp_returns, cov_matrix, min_weight=0.0, max_weight=0.5)
        print("Optimized weights:\n", opt_weights)
    except Exception as e:
        print("Portfolio optimization failed:", e)
        return

    # lace trade orders based on optimized weights
    place_orders(opt_weights)


if __name__ == "__main__":
    main()
    ib.disconnect()
    print("Disconnected from IB.")
