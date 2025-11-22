
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera, shapiro
from scipy import stats


sns.set(style="whitegrid")

# --- Load data ---
quakes = pd.read_excel('../data/earthquakes.xlsx', header=None)
quakes.columns = ['year', 'nr_quakes']

# create lagged first difference
quakes['lag_difference'] = quakes['nr_quakes'].shift(1) - quakes['nr_quakes'].shift(2)

# --- Preliminary visualization ---
plt.figure(figsize=(10, 5))
plt.plot(quakes['year'], quakes['nr_quakes'], color='blue', linewidth=1)
plt.xlabel('Year')
plt.ylabel('Number of earthquakes')
plt.title('Number of earthquakes 1900-2005')
plt.show()

# --- Split estimation and hold-out samples ---
estimation = quakes[quakes['year'] <= 1980]
hold_out = quakes[quakes['year'] > 1980]

# Plot estimation sample
plt.figure(figsize=(10, 5))
plt.plot(estimation['year'], estimation['nr_quakes'], color='blue', linewidth=1)
plt.xlabel('Year')
plt.ylabel('Number of earthquakes')
plt.title('Number of earthquakes 1900-1980; estimation sample')
plt.show()

# Plot hold-out sample
plt.figure(figsize=(10, 5))
plt.plot(hold_out['year'], hold_out['nr_quakes'], color='blue', linewidth=1)
plt.xlabel('Year')
plt.ylabel('Number of earthquakes')
plt.title('Number of earthquakes 1981-2005; hold-out sample')
plt.show()

# --- ACF and PACF ---
max_lag = 12
plot_acf(estimation['nr_quakes'], lags=max_lag)
plt.show()
plot_pacf(estimation['nr_quakes'], lags=max_lag)
plt.show()

# Ljung-Box test
Q_stats = []
p_vals = []
for i in range(1, max_lag+1):
    lb_test = acorr_ljungbox(estimation['nr_quakes'], lags=[i], return_df=True)
    Q_stats.append(lb_test['lb_stat'].values[0])
    p_vals.append(lb_test['lb_pvalue'].values[0])
    


# Compute ACF and PACF values manually
acf_vals = [estimation['nr_quakes'].autocorr(lag=i) for i in range(1, max_lag+1)]
pacf_vals = pacf(estimation['nr_quakes'], nlags=max_lag)[1:]  # skip lag 0

quakes_stats = pd.DataFrame({
    'acf': acf_vals,
    'pacf': pacf_vals,
    'Q_stat': Q_stats,
    'p_val': p_vals
})


# --- Function to compute p-values and t-statistics ---
def compute_pval(arima_res):
    coef = arima_res.params
    se = arima_res.bse
    t_stat = coef / se
    pval = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(arima_res.resid)-1))
    return pd.DataFrame({'t_stat': t_stat, 'p_val': pval})

# --- Fit AR(1) model ---
ar1 = ARIMA(estimation['nr_quakes'], order=(1,0,0)).fit()
compute_pval(ar1)


resid_ar1 = pd.DataFrame({'year': estimation['year'], 'residuals': ar1.resid})
resid_std = np.sqrt(ar1.mse)

plt.figure(figsize=(10,5))
plt.plot(resid_ar1['year'], resid_ar1['residuals'], color='blue')
plt.axhline(y=-resid_std, linestyle='--')
plt.axhline(y=resid_std, linestyle='--')
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.title('Residuals estimation sample; AR(1) model')
plt.show()

plot_acf(resid_ar1['residuals'])

# Histogram of residuals
plt.figure(figsize=(8,5))
sns.histplot(resid_ar1['residuals'], bins=20, color='skyblue', kde=False)
plt.show()

# Normality tests
jarque_res = jarque_bera(ar1.resid)
shapiro_test = shapiro(ar1.resid)
print("Jarque-Bera:", jarque_res)
print("Shapiro-Wilk:", shapiro_test)

fit_ar1 = pd.DataFrame({
    'year': estimation['year'],
    'real': estimation['nr_quakes'],
    'fit': ar1.fittedvalues
})

plt.figure(figsize=(10,5))
plt.plot(fit_ar1['year'], fit_ar1['real'], color='red', label='Real')
plt.plot(fit_ar1['year'], fit_ar1['fit'], color='blue', label='Fitted (AR1)')
plt.xlabel('Year')
plt.ylabel('Number of Earthquakes')
plt.title('Fitted vs Real Values (Estimation Sample; AR(1) model)')
plt.legend()
plt.show()


# Now we try to improve the model by adding the lag of the first-difference: 'y_{t-1} - y_{t-2}'.
# Since this model cannot be constructed with the 'order = c(p, d, q)' argument, we pass the lag
# of the first difference as a variable/regressor.
ar_diff = ARIMA(estimation['nr_quakes'].iloc[2:], order=(1,0,0), exog=estimation[['lag_difference']].iloc[2:]).fit()
compute_pval(ar_diff)
print(ar_diff.summary()) #lag not significant

# Maybe we can improve the model by adding 2 moving average terms?
arma12 = ARIMA(estimation['nr_quakes'], order=(1,0,2)).fit()
compute_pval(arma12)
# print(arma12.summary())

# Or maybe we can improve the model by adding a second lag of y
ar2 = ARIMA(estimation['nr_quakes'], order=(2,0,0)).fit()
compute_pval(ar2)

# Let's try an arma(1,1) model:
arma11 = ARIMA(estimation['nr_quakes'], order=(1,0,1)).fit()
compute_pval(arma11)

# --- Recursive 1-step ahead predictions after 1980 ---
ar1_full = ARIMA(quakes['nr_quakes'], order=(1,0,0)).fit()
pred_onestep = ar1_full.fittedvalues[quakes['year'] > 1980]

onestep_df = quakes.copy()
onestep_df['predictions'] = np.nan
onestep_df.loc[quakes['year'] > 1980, 'predictions'] = pred_onestep.values

# Plot predictions
plt.figure(figsize=(10,5))
plt.plot(onestep_df['year'], onestep_df['predictions'], color='blue', label='Predictions')
plt.plot(onestep_df['year'], onestep_df['nr_quakes'], color='red', label='Real')
plt.axvline(x=1980, linestyle='--')
plt.xlabel('Year')
plt.ylabel('Number of Earthquakes')
plt.title('One step ahead predictions (blue) vs real (red) number of earthquakes; AR(1) model')
plt.legend()
plt.show()

# Add 95% confidence intervals
sigma = np.std(ar1_full.resid, ddof=0)
onestep_df['upper'] = onestep_df['predictions'] + 2*sigma
onestep_df['lower'] = onestep_df['predictions'] - 2*sigma

plt.figure(figsize=(10,5))
plt.plot(onestep_df['year'], onestep_df['predictions'], color='blue', label='Predictions')
plt.plot(onestep_df['year'], onestep_df['nr_quakes'], color='red', label='Real')
plt.fill_between(onestep_df['year'], onestep_df['lower'], onestep_df['upper'], alpha=0.2)
plt.axvline(x=1980, linestyle='--')
plt.xlabel('Year')
plt.ylabel('Number of Earthquakes')
plt.title('One step ahead predictions with 95% C.I.; AR(1) model')
plt.legend()
plt.show()

# --- Forecast 15 years into the future ---
ar1_new = ARIMA(quakes['nr_quakes'], order=(1,0,0)).fit()
forecast_15yrs = ar1_new.get_forecast(steps=15)
forecast_mean = forecast_15yrs.predicted_mean
forecast_ci = forecast_15yrs.conf_int(alpha=0.05)

forecast_df = pd.DataFrame({
    'year': range(1900, 2021),
    'nr_quakes': list(quakes['nr_quakes']) + list(forecast_mean),
    'upper': [np.nan]*len(quakes) + list(forecast_ci['upper nr_quakes']),
    'lower': [np.nan]*len(quakes) + list(forecast_ci['lower nr_quakes'])
})

plt.figure(figsize=(10,5))
plt.plot(forecast_df['year'], forecast_df['nr_quakes'], color='red', label='Actual & Forecast')
plt.fill_between(forecast_df['year'], forecast_df['lower'], forecast_df['upper'], alpha=0.2)
plt.axvline(x=2005, linestyle='--')
plt.xlabel('Year')
plt.ylabel('Number of Earthquakes')
plt.title('Forecast of the number of earthquakes after 2005 with 95% C.I.; AR(1) model')
plt.legend()
plt.show()

# %%
