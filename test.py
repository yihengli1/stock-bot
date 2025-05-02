import yfinance as yf

if __name__ == "__main__":
    ticker = yf.Ticker("AAPL")

    # Get basic info
    print("Basic Info:")
    print(ticker.info)

    # Get historical data
    print("\nHistorical Data:")
    hist = ticker.history(period="1mo")
    print(hist)

    # Annual income statement
    df_annual_is = ticker.financials
    print("\nAnnual Financials:")
    print(df_annual_is)

    # Quarterly income statement
    df_qtr_is = ticker.quarterly_financials
    print("\nQuarterly Financials:")
    print(df_qtr_is)

    # Get balance sheet
    df_bs = ticker.balance_sheet
    print("\nBalance Sheet:")
    print(df_bs)

    # Get cash flow
    df_cf = ticker.cashflow
    print("\nCash Flow:")
    print(df_cf)
