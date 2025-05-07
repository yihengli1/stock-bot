from curl_cffi import requests
import yfinance as yf

session = requests.Session(impersonate="chrome")
ticker = yf.Ticker('MSFT', session=session)

print(ticker.info)
print(ticker.calendar)
