#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns

# Load data 
bikes = pd.read_csv('../data/bikes.csv', parse_dates=['date'])

# --- Preliminary analysis --- #
plt.figure(figsize=(10,4))
plt.plot(bikes['date'], bikes['count'])
plt.title('Bike rentals January 2011 - December 2012')
plt.show()

# Identify and remove outliers based on IQR method
Q1 = bikes['count'].quantile(0.25)
Q3 = bikes['count'].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (bikes['count'] < (Q1 - 1.5 * IQR)) | (bikes['count'] > (Q3 + 1.5 * IQR))
bikes = bikes[~outlier_condition]

# --- Model Training --- #
count = bikes['count'].values
n_obs = len(count)
n_holdout = 100
train_index = np.arange(n_obs - n_holdout)

train = count[train_index]
test = count[n_obs - n_holdout:]

# Non-seasonal ARIMA
fit_arima = ARIMA(train, order=(5, 1, 5)).fit()

# Autoregressive (AR) model
fit_ar = ARIMA(train, order=(7, 0, 0)).fit()

# Moving Average (MA) model
fit_ma = ARIMA(train, order=(0, 0, 7)).fit()

# --- Forecast 100 days ahead --- #
pred_arima = fit_arima.forecast(steps=n_holdout)
pred_ar = fit_ar.forecast(steps=n_holdout)
pred_ma = fit_ma.forecast(steps=n_holdout)

# --- 1-step ahead rolling forecasts --- #
model_arima = ARIMA(count, order=(5, 1, 5)).fit()
arima_1_ahead = model_arima.fittedvalues[-n_holdout:]

model_ar = ARIMA(count, order=(7, 0, 0)).fit()
ar_1_ahead = model_ar.fittedvalues[-n_holdout:]

model_ma = ARIMA(count, order=(0, 0, 7)).fit()
ma_1_ahead = model_ma.fittedvalues[-n_holdout:]

real_count = test


# Select the dates corresponding to the holdout/test set
test_dates = bikes['date'].iloc[-n_holdout:].reset_index(drop=True)


# --- Combine everything into a single DataFrame --- #
df_1_ahead = pd.DataFrame({
    "date": test_dates,  
    "real": test,
    "ARIMA_1step": arima_1_ahead,
    "AR_1step": ar_1_ahead,
    "MA_1step": ma_1_ahead
})

# Melt the DataFrame for seaborn plotting
df_plot = df_1_ahead.melt(id_vars="date", var_name="model", value_name="count")

plt.figure(figsize=(14,7))
sns.lineplot(
    data=df_plot,
    x="date",
    y="count",
    hue="model",
    marker="o"
)

plt.title("One-step ahead forecasts vs Real counts")
plt.xlabel("Date")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Model / Forecast type")
plt.tight_layout()
plt.show()

# RMSE calculations
rmse_arima = np.sqrt(np.mean((arima_1_ahead - real_count)**2))
rmse_ar = np.sqrt(np.mean((ar_1_ahead - real_count)**2))
rmse_ma = np.sqrt(np.mean((ma_1_ahead - real_count)**2))

print('RMSE ARIMA:', rmse_arima)
print('RMSE AR:', rmse_ar)
print('RMSE MA:', rmse_ma)

# %%
