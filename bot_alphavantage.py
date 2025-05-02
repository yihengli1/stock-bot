import ast
import os
import time
import datetime
import math
import requests
import schedule
import numpy as np
import pandas as pd
import cvxpy as cp
from dotenv import load_dotenv
from ib_insync import IB, Stock, util, Order
import openai
from newsapi import NewsApiClient
from typing import List, Tuple
import re

# Loading Environment Variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
fred_key = os.getenv("FRED_API_KEY")
news_key = os.getenv("NEWSAPI_KEY")
# You'll need to add this to your .env file
alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
newsapi = NewsApiClient(api_key=news_key)

TIMEZONE = "America/Vancouver"
EXECUTION_TIME = "09:30"                        # local tz
TOTAL_CAPITAL = 100_000                        # USD
NASDAQ_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
RISK_FREE_TICKER = "IEF"

OPENAI_MODEL = "gpt-4o-mini"
FIVE_YEAR_PERIOD = "5y"


class AlphaVantageStock:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.base_url = "https://www.alphavantage.co/query"
        self._info = None
        self._income_stmt = None
        self._balance_sheet = None
        self._cashflow = None

    def _make_request(self, function: str):
        params = {
            "function": function,
            "symbol": self.symbol,
            "apikey": alpha_vantage_key
        }
        response = requests.get(self.base_url, params=params)
        data = response.json()
        return data

    @property
    def info(self):
        if self._info is None:
            # Get company overview
            params = {
                "function": "OVERVIEW",
                "symbol": self.symbol,
                "apikey": alpha_vantage_key
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()

            # Convert string numbers to float/int
            for key in ['MarketCapitalization', 'EBITDA', 'SharesOutstanding', 'Beta']:
                if key in data:
                    try:
                        data[key] = float(data[key])
                    except (ValueError, TypeError):
                        pass

            self._info = data
        return self._info

    @property
    def income_stmt(self):
        if self._income_stmt is None:
            data = self._make_request("INCOME_STATEMENT")

            if 'annualReports' in data:
                df = pd.DataFrame(data['annualReports'])
                if not df.empty:
                    df.set_index('fiscalDateEnding', inplace=True)
                    # Convert string values to float
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                self._income_stmt = df
            else:
                print(
                    f"[WARN] No annual reports in income statement for {self.symbol}")
                self._income_stmt = pd.DataFrame()

        return self._income_stmt

    @property
    def balance_sheet(self):
        if self._balance_sheet is None:
            data = self._make_request("BALANCE_SHEET")

            if 'annualReports' in data:
                df = pd.DataFrame(data['annualReports'])
                if not df.empty:
                    df.set_index('fiscalDateEnding', inplace=True)
                    # Convert string values to float
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                self._balance_sheet = df
            else:
                print(
                    f"[WARN] No annual reports in balance sheet for {self.symbol}")
                self._balance_sheet = pd.DataFrame()

        return self._balance_sheet

    @property
    def cashflow(self):
        if self._cashflow is None:
            data = self._make_request("CASH_FLOW")

            print(data)

            if 'annualReports' in data:
                df = pd.DataFrame(data['annualReports'])
                if not df.empty:
                    df.set_index('fiscalDateEnding', inplace=True)
                    # Convert string values to float
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                self._cashflow = df
            else:
                print(
                    f"[WARN] No annual reports in cash flow for {self.symbol}")
                self._cashflow = pd.DataFrame()

        return self._cashflow

    def get_historical_data(self, period="1y"):
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": self.symbol,
            "outputsize": "full" if period == "5y" else "compact",
            "apikey": alpha_vantage_key
        }
        response = requests.get(self.base_url, params=params)
        data = response.json()

        if "Time Series (Daily)" in data:
            df = pd.DataFrame(data["Time Series (Daily)"]).T
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            })
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        return pd.DataFrame()


def fetch_risk_free_rate():
    """Most‑recent 10‑Y Treasury constant‑maturity yield (%)"""
    url = ("https://api.stlouisfed.org/fred/series/observations?"
           f"series_id=DGS10&api_key={fred_key}&file_type=json&sort_order=desc&limit=1")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    obs = r.json()["observations"][0]
    print(f"[FRED] 10‑Y Yield {obs['date']}: {obs['value']} %")
    return float(obs["value"])


def get_vix_level():
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": "^VIX",
        "apikey": alpha_vantage_key
    }
    response = requests.get("https://www.alphavantage.co/query", params=params)
    data = response.json()
    return float(data["Global Quote"]["05. price"])


def target_vol_from_vix(vix):
    if vix < 15:
        return 0.20
    if vix < 25:
        return 0.12
    return 0.07


def fetch_prices_ib(ib: IB, symbol, duration="1 Y"):
    contract = Stock(symbol, "SMART", "USD")
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr=duration,
        barSizeSetting="1 day", whatToShow="MIDPOINT",
        useRTH=True, formatDate=1)
    df = util.df(bars)
    df.set_index("date", inplace=True)
    return df["close"]


def returns_and_cov(ib: IB, tickers):
    rets = {}
    for t in tickers:
        try:
            p = fetch_prices_ib(ib, t)
            rets[t] = p.pct_change().dropna()
        except Exception as e:
            print(f"[WARN] {t}: {e}")
    df = pd.DataFrame(rets)
    return df.mean()*252, df.cov()*252


def news_headlines(symbol, lookback_days=30, n=10):
    from_date = (datetime.date.today() -
                 datetime.timedelta(days=lookback_days)).isoformat()
    q = f"{symbol} stock"
    articles = newsapi.get_everything(q=q, language="en",
                                      from_param=from_date, sort_by="relevancy", page_size=n)
    return [a["title"] for a in articles["articles"]]


def gpt_growth_path(symbol: str, headlines: List[str]) -> List[float]:
    prompt = (
        f"Recent news headlines for {symbol}:\n"
        + "\n".join(f"- {h}" for h in headlines)
        + "\nGive a conservative forecast of the company's free-cash-flow growth "
          "rate for *each* of the next five years. "
          "Return a Python list of five percentages, most imminent year first."
    )
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL, temperature=0.3,
            messages=[{"role": "user", "content": prompt}])

        raw = resp.choices[0].message.content.strip()

        # pull out the first [...] block
        m = re.search(r"\[.*?\]", raw, re.S)
        if not m:
            raise ValueError("No list of growth rates found")
        lst = ast.literal_eval(m.group(0))

        # tolerate either strings ("2.0%") or bare numbers (2.0)
        growths = []
        for x in lst:
            if isinstance(x, str):
                growths.append(float(x.strip("% ")) / 100)
            else:
                growths.append(float(x) / 100)

        if len(growths) != 5:
            raise ValueError("Need five growth rates")
    except Exception as e:
        print("gpt_growth_path Exception:", e)
        growths = [0.03] * 5

    return growths


def gpt_terminal_growth(symbol: str, headlines: List[str]) -> float:
    prompt = (
        f"Below are several recent news headlines about {symbol}:\n" + "\n".join(f"- {h}" for h in headlines) +
        "\nBased on this information, what would be a *sustainable long‑run* annual growth rate (in %) for the company's free cash flow beyond year 5? "
        "Reply with **one number only**."
    )
    try:
        resp = openai.chat.completions.create(model=OPENAI_MODEL, temperature=0.3,
                                              messages=[{"role": "user", "content": prompt}])
        val = float(resp.choices[0].message.content.strip().strip("% "))/100
    except Exception:
        print("gpt_terminal_growth Exception")
        val = 0.025
    return val


def grab_row(df: pd.DataFrame, aliases: list[str]) -> float:
    if df.empty:
        raise KeyError("DataFrame empty")
    for alias in aliases:
        matching_cols = [
            col for col in df.columns if alias.lower() in col.lower()]
        if matching_cols:
            val = df[matching_cols[0]].iloc[0]
            if not pd.isna(val):
                return float(val)
    raise KeyError(f"None of the aliases {aliases} found")


def latest_fcf(symbol: str, ticker: AlphaVantageStock) -> Tuple[float, str]:
    """Build trailing FCFF with generous fallbacks."""

    fin = ticker.income_stmt
    bs = ticker.balance_sheet
    cf = ticker.cashflow

    # Try cash flow based FCF first
    try:
        op = grab_row(cf, ["operatingCashflow", "operatingCashFlow",
                           "operatingCashFlowFromContinuingOperations"])
        capx = grab_row(cf, ["capitalExpenditures", "capitalExpenditure",
                             "capitalExpendituresDiscontinuedOperations"])
        fcf = op - abs(capx)
        if fcf > 0:
            return fcf, "CF fallback"
    except KeyError:
        pass

    # If cash flow method fails, try income statement based FCF
    if not fin.empty:
        try:
            EBIT = grab_row(
                fin, ["ebit", "operatingIncome", "operatingIncomeLoss"])
            tax = grab_row(
                fin, ["incomeTaxExpense", "incomeTaxExpenseBenefit"])
            pretax = grab_row(
                fin, ["incomeBeforeTax", "incomeBeforeIncomeTaxes"])
            τ = abs(tax / pretax) if pretax else 0.25

            try:
                DnA = grab_row(
                    cf, ["depreciation", "depreciationAndAmortization"])
            except KeyError:
                DnA = 0.0

            try:
                CapEx = abs(
                    grab_row(cf, ["capitalExpenditures", "capitalExpenditure"]))
            except KeyError:
                CapEx = 0.0

            if not bs.empty:
                try:
                    current_assets = grab_row(
                        bs, ["totalCurrentAssets", "currentAssets"])
                    current_liabilities = grab_row(
                        bs, ["totalCurrentLiabilities", "currentLiabilities"])
                    nwc = current_assets - current_liabilities
                    ΔNWC = 0  # Simplified for Alpha Vantage data structure
                except KeyError:
                    ΔNWC = 0.0
            else:
                ΔNWC = 0.0

            fcff = EBIT * (1 - τ) + DnA - CapEx - ΔNWC
            if fcff > 0:
                return fcff, "EBIT formula (FCFF)"
        except KeyError as e:
            print(f"[WARN] Missing key in financial data: {e}")

    return 0.0, "no data"


def dcf_valuation(symbol: str) -> float:
    heads = news_headlines(symbol)
    growths = gpt_growth_path(symbol, heads)
    term_g = gpt_terminal_growth(symbol, heads)

    print("Growth", growths)
    print("Terminal Growth", term_g)

    tkr = AlphaVantageStock(symbol)

    fcf0, method = latest_fcf(symbol, tkr)
    print("Starting FCF", fcf0)
    print("Method", method)

    disc_rate = wacc(symbol, tkr)

    pv, fcf = 0.0, fcf0
    for t, g in enumerate(growths, 1):
        fcf *= (1 + g)
        pv += fcf * ((1/((1 + disc_rate)**t)))

    tv = fcf * (1 + term_g)/(disc_rate - term_g)
    pv += tv * ((1/((1 + disc_rate)**len(growths))))

    shares = float(tkr.info.get("SharesOutstanding", 0))
    if shares == 0:
        raise ValueError("Shares outstanding data not available")

    intrinsic = pv/shares
    print(f"[DCF] {symbol}: ${intrinsic:,.2f}  (WACC {disc_rate:.2%}, term g {(term_g*100):.2f} %, FCF via {method})")
    return intrinsic


def wacc(symbol: str, ticker: AlphaVantageStock) -> float:
    info = ticker.info
    fin = ticker.income_stmt
    bs = ticker.balance_sheet

    report = bs["annualReports"][0]
    print("Report", report)

    beta = float(info.get("Beta", 1.0))
    exp_ret = 0.1  # Hardcoded expected return

    rf = fetch_risk_free_rate()
    erp = exp_ret - (rf/100)

    print("expected return", exp_ret)
    print("risk-free rate", rf)
    print("risk premium", erp)

    short_term = float(report.get("shortTermDebt"))
    long_term = float(report.get("longTermDebt"))
    cap_leases = float(report.get("capitalLeaseObligations"))

    total_debt = short_term + long_term + cap_leases

    print("totalDebt", total_debt)

    # Equity & Debt totals
    equity = float(info.get("MarketCapitalization", 0))

    if (equity + total_debt) == 0:
        raise ValueError("No cap‑structure data")

    w_e = equity/(equity+total_debt)
    w_d = 1 - w_e

    print("equity percentage", w_e)
    print("debt percentage", w_d)

    # Cost of debt = interest expense / total debt
    try:
        int_exp = abs(fin["interestExpense"].iloc[0])
        r_d = int_exp/total_debt if total_debt else 0.04
    except:
        r_d = 0.04

    print("cost of debt", r_d)

    # Tax rate
    try:
        tax_exp = fin["incomeTaxExpense"].iloc[0]
        ebt = fin["incomeBeforeTax"].iloc[0]
        tax_rate = abs(tax_exp / ebt) if ebt else 0.25
    except:
        tax_rate = 0.25

    print("tax rate", tax_rate)

    r_e = (rf/100) + beta*erp

    print("Equity rate", r_e)
    wacc_val = w_e*r_e + w_d*r_d*(1 - tax_rate)
    return wacc_val


def dcf_filter(ib: IB, tickers):
    selected = []
    for t in tickers:
        try:
            price = fetch_prices_ib(ib, t, duration="3 D").iloc[-1]
            val = dcf_valuation(t)
            if val >= price:
                selected.append(t)
        except Exception as e:
            print(f"[DCF‑FAIL] {t}: {e}")
    return selected


def optimise_max_sharpe(mu, cov):
    n = len(mu)
    w = cp.Variable(n)
    sharpe = (mu.values @ w) / cp.sqrt(cp.quad_form(w, cov.values))
    prob = cp.Problem(cp.Maximize(sharpe), [cp.sum(w) == 1, w >= 0])
    prob.solve()
    if w.value is None:
        raise RuntimeError("MVO failed")
    return pd.Series(w.value, index=mu.index)


def blend_with_risk_free(w_risky, cov, target_vol):
    sig = math.sqrt(w_risky.values @ cov.values @ w_risky.values)
    lever = target_vol/sig
    w_final = w_risky*lever
    w_final[RISK_FREE_TICKER] = 1-lever
    return w_final


def place_orders(ib: IB, weights, capital):
    for tick, w in weights.items():
        if abs(w) < 1e-4:
            continue
        alloc = capital*w
        contract = Stock(tick, "SMART", "USD")
        md = ib.reqMktData(contract, "", False, False)
        ib.sleep(2)
        price = float(md.last or md.close or 0)
        if price <= 0:
            print(f"[SKIP] {tick} price unavailable")
            continue
        qty = int(alloc//price)
        if qty == 0:
            continue
        order = Order(action="BUY", orderType="MKT", totalQuantity=qty)
        ib.placeOrder(contract, order)
        print(f"BUY {qty} {tick} ≈ ${price:.2f}")


def rebalance():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)
    print(f"\n=== REBALANCE {datetime.datetime.now()} ===")

    candidates = dcf_filter(ib, NASDAQ_TICKERS)
    if not candidates:
        print("No undervalued stocks today.")
        ib.disconnect()
        return
    risky = candidates
    tickers = risky + [RISK_FREE_TICKER]

    mu, cov = returns_and_cov(ib, tickers)
    rf_ret = mu.pop(RISK_FREE_TICKER)
    cov_risky = cov.loc[risky, risky]

    w_risky = optimise_max_sharpe(mu, cov_risky)

    vix = get_vix_level()
    targ = target_vol_from_vix(vix)
    w_port = blend_with_risk_free(w_risky, cov_risky, targ)

    print(f"VIX {vix:.2f} → target σ {targ:.0%}")
    print("Weights:\n", w_port.round(4))

    # Execute (UNCOMMENT WHEN READY)
    # place_orders(ib, w_port, TOTAL_CAPITAL)

    ib.disconnect()
    print("Done.\n")


if __name__ == "__main__":
    print(dcf_valuation("AAPL"))
    # rebalance()
