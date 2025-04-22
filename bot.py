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


def estimate_growth_via_gpt(symbol: str, headlines: list[str]) -> float:
    """
    Ask GPT for a conservative 5‑year FCF growth rate (in %).
    Returns a float; falls back to 3.0 on any error / bad parse.
    """
    prompt = (
        f"Given these recent news headlines about {symbol}:\n" +
        "\n".join(f"- {h}" for h in headlines) +
        "\nProvide a conservative average annual free‑cash‑flow growth rate (%) "
        "you expect for the next 5 years. Reply with one number only."
    )

    try:
        resp = openai.chat.completions.create(            # ← NEW endpoint path
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        content = resp.choices[0].message.content.strip()
        return float(content.strip("% \n"))
    except Exception as e:
        print(f"[GPT‑fallback] {symbol}: {e}")
        return 3.0        # ← safe default


def grab_row(df: pd.DataFrame, aliases: list[str]) -> float:
    """
    Return the first non‑NaN value in df for any alias in *aliases*.
    Raises KeyError if none of the aliases are present.
    """
    for a in aliases:
        if a in df.index and not df.loc[a].isna().all():
            return float(df.loc[a].dropna().iloc[0])
    raise KeyError(f"None of the aliases {aliases} found in DataFrame")


def dcf_valuation(symbol: str, years: int = 5, terminal_growth: float = 2.5) -> float:
    """
    One‑step FCFF DCF.
    • FCF₀  =  Operating Cash Flow  –  |CapEx|
      (or use Yahoo’s ‘Free Cash Flow’ line when available)
    • 5‑year explicit growth  →  GPT‑estimated
    • WACC  =  r_f  +  β·ERP   (very simplified)
    Returns intrinsic value per share (USD).
    """
    yf_tkr = yf.Ticker(symbol)
    cf = yf_tkr.cashflow
    if cf.empty:
        raise ValueError("No cash‑flow data from Yahoo")

    try:
        # Try to use Yahoo’s ready‑made FCF line first
        fcf0 = grab_row(cf, ["Free Cash Flow"])
    except KeyError:
        ocf = grab_row(cf, ["Operating Cash Flow",
                            "Total Cash From Operating Activities"])
        capex = grab_row(cf, ["Capital Expenditure",
                              "Capital Expenditures",
                              "Net PPE Purchase And Sale"])
        fcf0 = ocf + capex                 # capex is negative in Yahoo tables

    # ── growth rate via GPT on recent news ───────────────────
    heads = news_headlines(symbol)
    g = estimate_growth_via_gpt(symbol, heads) / 100        # decimal

    print("GRWOTH: ", g)

    # ── discount rate (WACC) ──────────────────────────────────
    beta = yf_tkr.info.get("beta", 1.0)
    rf = fetch_risk_free_rate() / 100                        # decimal
    # equity‑risk premium assumption
    erp = 0.05
    wacc = rf + beta * erp
    print("wacc", wacc)

    disc = 1 / (1 + wacc)

    # ── explicit FCF projection ───────────────────────────────
    pv, fcf = 0.0, fcf0

    print("free cf", fcf0)
    for t in range(1, years + 1):
        fcf *= (1 + g)
        pv += fcf * (disc ** t)

    print("free cf", fcf)

    # ── terminal value ────────────────────────────────────────
    tg = terminal_growth / 100
    tv = fcf * (1 + tg) / (wacc - tg)

    print("terminal value", tv)
    pv += tv * (disc ** years)

    shares = yf_tkr.info["sharesOutstanding"]
    intrinsic = pv / shares
    print(f"[DCF] {symbol}: intrinsic ≈ ${intrinsic:,.2f}  "
          f"(FCF₀ {fcf0/1e9:,.1f} B, g {g*100:.1f} %, β {beta:.2f})")
    return intrinsic


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

    # 1️⃣ Stock universe via DCF
    candidates = dcf_filter(ib, NASDAQ_TICKERS)
    if not candidates:
        print("No undervalued stocks today.")
        ib.disconnect()
        return
    risky = candidates
    tickers = risky + [RISK_FREE_TICKER]

    # 2️⃣ Expected returns & cov
    mu, cov = returns_and_cov(ib, tickers)
    rf_ret = mu.pop(RISK_FREE_TICKER)
    cov_risky = cov.loc[risky, risky]

    # 3️⃣ Max‑Sharpe risky portfolio
    w_risky = optimise_max_sharpe(mu, cov_risky)

    # 4️⃣ Blend with risk‑free via VIX
    vix = get_vix_level()
    targ = target_vol_from_vix(vix)
    w_port = blend_with_risk_free(w_risky, cov_risky, targ)

    print(f"VIX {vix:.2f} → target σ {targ:.0%}")
    print("Weights:\n", w_port.round(4))

    # 5️⃣ Execute (UNCOMMENT WHEN READY)
    # place_orders(ib, w_port, TOTAL_CAPITAL)

    ib.disconnect()
    fetch_risk_free_rate()
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
