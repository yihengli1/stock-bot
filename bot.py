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
import yfinance as yf
import openai
from newsapi import NewsApiClient
from typing import List, Tuple
import re
from curl_cffi import requests


# Loading Environment Variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
fred_key = os.getenv("FRED_API_KEY")
news_key = os.getenv("NEWSAPI_KEY")
newsapi = NewsApiClient(api_key=news_key)

TIMEZONE = "America/Vancouver"
EXECUTION_TIME = "09:30"                        # local tz
TOTAL_CAPITAL = 100_000                        # USD
NASDAQ_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
RISK_FREE_TICKER = "IEF"

OPENAI_MODEL = "gpt-4o-mini"
FIVE_YEAR_PERIOD = "5y"


def fetch_risk_free_rate():
    """Most‑recent 10‑Y Treasury constant‑maturity yield (%)"""
    url = ("https://api.stlouisfed.org/fred/series/observations?"
           f"series_id=DGS10&api_key={fred_key}&file_type=json&sort_order=desc&limit=1")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    obs = r.json()["observations"][0]
    print(f"[FRED] 10‑Y Yield {obs['date']}: {obs['value']} %")
    return float(obs["value"])


def get_vix_level():
    vix = yf.download("^VIX", period="5d", interval="1d")
    return float(vix["Close"].iloc[-1])


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

        raw = resp.choices[0].message.content.strip(
        )                                 # — diagnostic

        # pull out the first [...] block
        m = re.search(r"\[.*?\]", raw, re.S)
        if not m:
            raise ValueError("No list of growth rates found")
        lst = ast.literal_eval(m.group(0))         # safe eval -> list

        # tolerate either strings ("2.0%") or bare numbers (2.0)
        growths = []
        for x in lst:
            if isinstance(x, str):
                growths.append(float(x.strip("% ")) / 100)
            else:                                    # int / float
                growths.append(float(x) / 100)

        # sanity-check length
        if len(growths) != 5:
            raise ValueError("Need five growth rates")
    except Exception as e:
        print("gpt_growth_path Exception:", e)
        growths = [0.03] * 5                         # fallback flat 3 %

    return growths


def gpt_terminal_growth(symbol: str, headlines: List[str]) -> float:
    """Infer a long‑run terminal FCF growth rate from recent qualitative news."""
    prompt = (
        f"Below are several recent news headlines about {symbol}:\n" + "\n".join(f"- {h}" for h in headlines) +
        "\nBased on this information, what would be a *sustainable long‑run* annual growth rate (in %) for the company's free cash flow beyond year 5? "
        "Reply with **one number only**."
    )
    try:
        resp = openai.chat.completions.create(model=OPENAI_MODEL, temperature=0.3,
                                              messages=[{"role": "user", "content": prompt}])
        val = float(resp.choices[0].message.content.strip().strip("% "))/100
    except Exception:
        print("gpt_terminal_growth Exception")
        val = 0.025   # fallback 2.5 %
    return val


def grab_row(df: pd.DataFrame, aliases: list[str]) -> float:
    """Return first non‑NaN value for any alias, case‑insensitive."""
    if df.empty:
        raise KeyError("DataFrame empty")
    idx_l = {i.lower(): i for i in df.index}
    for a in aliases:
        key = a.lower()
        if key in idx_l and not df.loc[idx_l[key]].isna().all():
            return float(df.loc[idx_l[key]].dropna().iloc[0])
    # also try substring match
    for i in df.index:
        for a in aliases:
            if a.lower() in i.lower() and not df.loc[i].isna().all():
                return float(df.loc[i].dropna().iloc[0])
    raise KeyError(f"None of the aliases {aliases} found")


def cost_of_debt_and_tax(tkr: yf.Ticker) -> tuple[float, float]:
    is_df = tkr.income_stmt if hasattr(tkr, "income_stmt") else tkr.financials
    bs_df = tkr.balance_sheet
    try:
        int_exp = abs(grab_row(is_df, ["Interest Expense"]))
        tot_debt = grab_row(bs_df, ["Total Debt"])
        cod = int_exp / tot_debt if tot_debt else 0.0
    except KeyError:
        cod = 0.03      # fallback 3 %
    # effective tax rate
    try:
        tax = grab_row(is_df, ["Income Tax Expense"])
        ebt = grab_row(is_df, ["Ebit", "Pretax Income"])
        tax_rate = tax / ebt if ebt else 0.25
    except KeyError:
        tax_rate = 0.25
    return cod, tax_rate


def dcf_valuation(symbol: str) -> float:
    """Intrinsic value per share (USD) via 5‑year explicit FCF + GPT terminal growth."""
    heads = news_headlines(symbol)
    growths = gpt_growth_path(symbol, heads)
    # ← NEW GPT‑driven terminal growth
    term_g = gpt_terminal_growth(symbol, heads)

    print("Growth", growths)
    print("Terminal Growth", term_g)

    # impersonate browser TLS, (patched yfinance api for some reason)
    session = requests.Session(impersonate="chrome")
    tkr = yf.Ticker(symbol, session=session)

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

    shares = tkr.info["sharesOutstanding"]
    intrinsic = pv/shares
    print(f"[DCF] {symbol}: ${intrinsic:,.2f}  (WACC {disc_rate:.2%}, term g {(term_g*100):.2f} %, FCF via {method})")
    return intrinsic


def latest_fcf(symbol: str, ticker: yf.Ticker) -> Tuple[float, str]:
    """Build trailing FCFF with generous fallbacks – never crash if a tax row is missing."""

    # Try the new income_stmt API first, then financials
    if hasattr(ticker, "income_stmt") and not ticker.income_stmt.empty:
        fin = ticker.income_stmt
    else:
        # explicit fetch
        ticker.get_financials()
        fin = ticker.financials

    if fin.empty:
        print(
            f"[WARN] No P&L data for {symbol}, falling back to CF-based FCF.")
        # Try cashflow directly
        cf = ticker.cashflow
        try:
            op = grab_row(cf, ["Total Cash From Operating Activities"])
            capx = grab_row(cf, ["Capital Expenditures"])
            return op - capx, "CF fallback"
        except KeyError:
            return 0.0, "no data"

    bs = ticker.balance_sheet
    cf = ticker.cashflow

    # existing EBIT-based FCFF
    EBIT = grab_row(fin, ["Ebit", "EBIT", "Operating Income"])

    # tax rows are notoriously inconsistent; treat missing as zero (=> τ = 0.25 fallback)
    try:
        tax = grab_row(fin, [
            "Income Tax Expense", "Provision for Income Taxes", "Income Taxes Paid"
        ])
        pretax = grab_row(fin, [
            "Pretax Income", "Ebt", "Income Before Tax", "Income Before Taxes"
        ])
        τ = abs(tax / pretax) if pretax else 0.25
    except KeyError:
        τ = 0.25  # flat fallback

    DnA = grab_row(cf, [
        "Depreciation", "Depreciation & Amortization", "Depreciation & depletion"
    ])

    CapEx = -grab_row(cf, [
        "Capital Expenditure", "Capital Expenditures", "Capital Expenditures (purchase of plant, property, and equipment)"
    ])  # Yahoo values are negative

    # ΔNWC with safety for single‑period balance sheets
    if not bs.empty and "Total Current Assets" in bs.index and "Total Current Liabilities" in bs.index:
        nwc = bs.loc["Total Current Assets"] - \
            bs.loc["Total Current Liabilities"]
        ΔNWC = nwc.iloc[0] - (nwc.iloc[1] if len(nwc) > 1 else nwc.iloc[0])
    else:
        ΔNWC = 0.0

    fcff = EBIT * (1 - τ) + DnA - CapEx - ΔNWC
    return float(fcff), "EBIT formula (FCFF)"


def wacc(symbol: str, tkr: yf.ticker) -> float:
    info = tkr.info

    beta = info.get("beta", 1.0)
    exp_ret = 0.1  # Hardcoded expected return

    rf = fetch_risk_free_rate()

    print("expected return", exp_ret)

    erp = exp_ret - (rf/100)

    print("risk-free rate", rf)
    print("risk premium", erp)

    # Equity & Debt totals
    equity = info.get("marketCap", 0)
    debt = info.get("totalDebt", 0)
    if (equity + debt) == 0:
        raise ValueError("No cap‑structure data")

    w_e = equity/(equity+debt)
    w_d = 1 - w_e

    print("equity percentage", w_e)
    print("debt percentage", w_d)

    # Cost of debt = interest expense / total debt
    fin = tkr.financials
    try:
        int_exp = abs(
            fin.loc[[i for i in fin.index if "interest expense" in i.lower()][0]].dropna().iloc[0])
        r_d = int_exp/debt
    except Exception:
        r_d = 0.04  # 4 % fallback

    print("cost of debt", r_d)

    # Tax rate heuristic (same as earlier)
    try:
        tax_exp = fin.loc[[
            i for i in fin.index if "tax" in i.lower()][0]].dropna().iloc[0]
        ebt = fin.loc[[i for i in fin.index if "ebt" in i.lower(
        ) or "pretax" in i.lower()][0]].dropna().iloc[0]
        tax_rate = abs(tax_exp / ebt) if ebt else 0.25
    except Exception:
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


# MEAN‑VAR OPTIMISATION

# Efficient Frontiers!!
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

# ORDER EXECUTION


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


# REBALANCE EVERY DAY yay
def rebalance():

    ib = IB()

    # 7496: REAL | 7497: PAPER
    ib.connect('127.0.0.1', 7497, clientId=1)
    print(f"\n=== REBALANCE {datetime.datetime.now()} ===")

    # Stock universe via DCF
    candidates = dcf_filter(ib, NASDAQ_TICKERS)
    if not candidates:
        print("No undervalued stocks today.")
        ib.disconnect()
        return
    risky = candidates
    tickers = risky + [RISK_FREE_TICKER]

    # Expected returns & cov
    mu, cov = returns_and_cov(ib, tickers)
    rf_ret = mu.pop(RISK_FREE_TICKER)
    cov_risky = cov.loc[risky, risky]

    # Max‑Sharpe risky portfolio
    w_risky = optimise_max_sharpe(mu, cov_risky)

    # Blend with risk‑free via VIX
    vix = get_vix_level()
    targ = target_vol_from_vix(vix)
    w_port = blend_with_risk_free(w_risky, cov_risky, targ)

    print(f"VIX {vix:.2f} → target σ {targ:.0%}")
    print("Weights:\n", w_port.round(4))

    # Execute (UNCOMMENT WHEN READY)
    # place_orders(ib, w_port, TOTAL_CAPITAL)

    ib.disconnect()
    print("Done.\n")


# SCHEDULER
# if __name__ == "__main__":
#     schedule.every().day.at(EXECUTION_TIME).do(rebalance)
#     print(
#         f"Scheduler started – will run daily at {EXECUTION_TIME} ({TIMEZONE})")
#     while True:
#         schedule.run_pending()
#         time.sleep(60)


# TESTING PURPOSES
if __name__ == "__main__":
    print(dcf_valuation("AAPL"))
    # rebalance()
