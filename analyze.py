import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

tickers = ["SPY", "EEM", "TLT", "GLD"]
weights = pd.Series([0.4, 0.2, 0.2, 0.2], index=tickers)

prices = yf.download(tickers, period="5y", auto_adjust=True)["Close"]
returns = prices[tickers].pct_change().dropna()
port = returns @ weights

print(port.describe())

port.hist(bins=50)
plt.title("Tagesrenditen des Portfolios")
plt.savefig("output/histogramm.png")
plt.show()