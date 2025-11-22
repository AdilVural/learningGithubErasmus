#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# --- Load data ---
spx = pd.read_csv('../data/spx.csv', parse_dates=['date'])

# --- Analysis of closing price (not very interesting) ---
plt.figure(figsize=(12, 6))
plt.plot(spx['date'], spx['close'], color='blue')
plt.title('S&P500 closing price from 2010-01-01 until 2018-06-29')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Train ARIMA model on closing prices
order = (1, 1, 1)  # (p,d,q)
arima_model = ARIMA(spx['close'], order=order).fit()
d = order[1]  # Number of differences

# Get in-sample fitted values starting after differencing
fitted_values = arima_model.predict(start=d, end=len(spx)-1)

# Prepend NaNs for first d values to match original series length
fitted_full = np.concatenate([np.full(d, np.nan), fitted_values])

df_arima = pd.DataFrame({
    'date': spx['date'],
    'real': spx['close'],
    'fitted': fitted_full
})

# Plot real vs fitted
plt.figure(figsize=(12, 6))
plt.plot(df_arima['date'], df_arima['real'], color='red', label='Real')
plt.plot(df_arima['date'], df_arima['fitted'], color='blue', label='Fitted')
plt.title('S&P500 closing price (red) and fitted values (blue)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# AR(1) with drift (random walk)
random_walk_model = ARIMA(spx['close'], order=(1,0,0), trend='c').fit()

# --- Analysis of daily stock returns (more interesting) ---
spx['return'] = spx['close'].pct_change() * 100  # percentage returns

df_returns = pd.DataFrame({
    'date': spx['date'],
    'return': spx['return']
})

# Plot daily returns
plt.figure(figsize=(12, 6))
plt.plot(df_returns['date'], df_returns['return'], color='blue')
plt.title('Daily returns S&P 500')
plt.xlabel('Date')
plt.ylabel('Return in %')
plt.show()

# ARIMA model for daily returns
arima_returns_model = ARIMA(df_returns['return'].dropna(), order=(1, 0, 1)).fit()
fitted_returns = arima_returns_model.fittedvalues
df_returns['fitted'] = np.nan
df_returns.loc[df_returns['return'].notna(), 'fitted'] = fitted_returns

# Analyze returns in May and June 2018
returns_june = df_returns[df_returns['date'] >= pd.to_datetime('2018-05-01')]

plt.figure(figsize=(12, 6))
plt.plot(returns_june['date'], returns_june['return'], color='red', label='Real')
plt.plot(returns_june['date'], returns_june['fitted'], color='blue', label='Fitted')
plt.title('Daily returns S&P 500 (red) and fitted values (blue)')
plt.xlabel('Date')
plt.ylabel('Return in %')
plt.legend()
plt.show()

# %%
