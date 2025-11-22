#%%
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

wine_sales = pd.read_csv('../data/wine_sales.csv', parse_dates=['date'])
plt.figure(figsize=(12,8))
plt.title('Historical wine sales in Australia since 1980')
plt.plot( wine_sales['date'], wine_sales['sales'])
plt.show()

###--- Training the model ---###

n_obs = len(wine_sales['sales'])
n_holdout = 12

# Convert sales into a time series with monthly frequency
wine_sales.set_index('date', inplace=True)
wine_ts = wine_sales['sales'].asfreq('MS')  # Monthly start frequency

# Train-test split
train = wine_ts.iloc[:-n_holdout]
test = wine_ts.iloc[-n_holdout:]

# Fit SARIMA model manually
# Using SARIMA(1,1,1)(1,1,1,12) as a reasonable starting point for seasonal monthly data

full_arima = SARIMAX(wine_ts, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()

# Extract fitted values for train and test
train_res = pd.DataFrame({
    'date': train.index,
    'sales': train.values,
    'arima': full_arima.fittedvalues.iloc[:len(train)]
})

test_res = pd.DataFrame({
    'date': test.index,
    'sales': test.values,
    'arima': full_arima.fittedvalues.iloc[len(train):]
})


###--- Evaluate model performance ---###

# Plot in-sample fit
plt.figure(figsize=(12, 6))
plt.plot(train_res['date'], train_res['sales'], color='red', marker='o', label='Real Sales')
plt.plot(train_res['date'], train_res['arima'], color='blue', marker='o', label='Fitted Sales')
plt.title('Real sales (red) vs fitted sales (blue)')
plt.legend()
plt.show()

# Plot out-of-sample predictions
plt.figure(figsize=(12, 6))
plt.plot(test_res['date'], test_res['sales'], color='red', marker='o', label='Real Sales')
plt.plot(test_res['date'], test_res['arima'], color='blue', marker='o', label='Predicted Sales')
plt.title('Real sales (red) vs predicted sales (blue)')
plt.legend()
plt.show()

wine_ts = pd.Series(
    data=wine_sales["sales"].values,
    index=pd.date_range(start="1980-01-01", periods=len(wine_sales), freq="M")
)



# %%
