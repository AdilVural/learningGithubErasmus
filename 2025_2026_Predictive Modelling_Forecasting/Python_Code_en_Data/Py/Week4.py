#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey
from scipy.stats import jarque_bera
import statsmodels.api as sm
from scipy.stats import chi2

# Load data
#INSERT DATA HERE
earthquakes = XX
earthquakes.columns = ['year', 'nr']

# Plot full data
earthquakes.plot(x='year', y='nr')
plt.show()

# Split estimation and hold-out samples
estimation_sample = earthquakes[earthquakes['year'] <= 1980]
hold_out_sample = earthquakes[earthquakes['year'] > 1980]

# Plot subsets
estimation_sample.plot(x='year', y='nr')
plt.show()
hold_out_sample.plot(x='year', y='nr')
plt.show()

# ACF and PACF
max_lag = 12
acf_vals = acf(estimation_sample['nr'], nlags=max_lag)[0:]
pacf_vals = pacf(estimation_sample['nr'], nlags=max_lag)[0:]
acf_pacf = pd.DataFrame({'lag': np.arange(0, max_lag+1), 'acf': acf_vals, 'pacf': pacf_vals})
N = len(estimation_sample['nr'])
two_SE = 2 / np.sqrt(N)
lags = np.arange(max_lag+1)

plt.figure(figsize=(10,6))
plt.stem(lags, acf_vals)
plt.axhline(two_SE, color='red')
plt.axhline(-two_SE, color='red')
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.title("ACF")
plt.show()

plt.figure(figsize=(10,6))
plt.stem(lags[1:], pacf_vals[1:])
plt.axhline(two_SE, color='red')
plt.axhline(-two_SE, color='red')
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.title("PACF")
plt.show()


# AR(1)
ar1 = AutoReg(estimation_sample['nr'], lags=1, old_names=False).fit()
print(ar1.summary())

# Residuals
actual = estimation_sample['nr'].iloc[1:].reset_index(drop=True)
fitted = ar1.fittedvalues
residual = ar1.resid
actuals_predict = pd.DataFrame({'actual': actual, 'fitted': fitted, 'residual': residual})
plt.figure(figsize=(10,6))
plt.plot(actuals_predict['actual'], color='red')
plt.plot(actuals_predict['fitted'], color='green')
plt.plot(actuals_predict['residual'], color='blue')
plt.show()

# Ljung-Box test
print(acorr_ljungbox(residual, lags=[1], return_df=True))

# ARMA(2,2) auxiliary regression
p = 2
r = 2

# Fit ARMA(p, r)
arma_p_r = ARIMA(estimation_sample['nr'], order=(p, 0, r)).fit()

# Add residuals to the dataset
example_data = estimation_sample.copy()
example_data['residual'] = arma_p_r.resid

# Build auxiliary regression dataset:
aux_reg_data = example_data[['residual', 'nr']].copy()

# Add lags of nr
for i in range(1, p + 1):
    aux_reg_data[f'lag_nr_{i}'] = aux_reg_data['nr'].shift(i)

# Add lags of residual
for j in range(1, r + 1):
    aux_reg_data[f'lag_resid_{j}'] = aux_reg_data['residual'].shift(j)

# Drop NA rows (aligns all data exactly like R)
aux_clean = aux_reg_data.dropna().copy()

# Endogenous variable = residual
Y = aux_clean['residual']

# Exogenous variables = all except residual and nr
X = aux_clean.drop(columns=['residual', 'nr'])

# Add constant
X = sm.add_constant(X)

# Fit auxiliary regression
aux_reg = sm.OLS(Y, X).fit()

# Compute R-squared and LM test statistic
R_squared = aux_reg.rsquared
test_stat = len(estimation_sample) * R_squared
critical_value = chi2.ppf(0.95, df=r)

# Output (matching R)
print("R-squared:", R_squared)
print("Test statistic (n * R²):", test_stat)
print("Critical value (chi-square):", critical_value)

if test_stat < critical_value:
    print("Failed to reject H0: The AR(p) model is adequate")
elif test_stat > critical_value:
    print("H0: The AR(p) model is adequate is being rejected")
else:
    print("Something unexpected happened.")


################################## Breusch-Godfrey test ####################################
# Prepare data
est = estimation_sample.copy()
est['lag_nr'] = est['nr'].shift(1)

# Drop NA for regression
est_clean = est.dropna()

# Fit the regression nr ~ lag(nr)
X = sm.add_constant(est_clean['lag_nr'])
y = est_clean['nr']

model = sm.OLS(y, X).fit()

# Perform Breusch-Godfrey test, order = 2
bg_test = acorr_breusch_godfrey(model, nlags=2)

# bg_test returns:
# (LM statistic, LM p-value, F statistic, F p-value)

lm_stat = bg_test[0]
lm_pvalue = bg_test[1]
f_stat = bg_test[2]
f_pvalue = bg_test[3]

print("Breusch-Godfrey Test")
print("--------------------------------")
print(f"LM statistic: {lm_stat}")
print(f"LM p-value:   {lm_pvalue}")
print(f"F statistic:  {f_stat}")
print(f"F p-value:    {f_pvalue}")



# ----------------------------------------------------------
# 2. Build auxiliary regression dataset to get coefficients
# ----------------------------------------------------------
# Get residuals
resid = model.resid
aux = pd.DataFrame({
    'residual': resid,
    'lag_nr': est_clean['lag_nr'],
    'lag_resid_1': resid.shift(1),
    'lag_resid_2': resid.shift(2)
})

aux_clean = aux.dropna()

Y_aux = aux_clean['residual']
X_aux = aux_clean[['lag_nr', 'lag_resid_1', 'lag_resid_2']]
X_aux = sm.add_constant(X_aux)

# ----------------------------------------------------------
# 3. Run the auxiliary regression
# ----------------------------------------------------------
aux_reg = sm.OLS(Y_aux, X_aux).fit()
print(aux_reg.summary())
print(aux_reg.params)

####################################
# Normality
plt.figure(figsize=(10,6))
plt.hist(residual, bins=14)
plt.show()
print(residual.describe())
print("Skew", residual.skew())
print("Kurt", residual.kurt()+3)
print(jarque_bera(residual))
####################################


# AR(2)
ar2 = AutoReg(estimation_sample['nr'], lags=2, old_names=False).fit()
print(ar2.summary())
# ARMA(1,1)
arma_11 = ARIMA(estimation_sample['nr'], order=(1,0,1)).fit()
print(arma_11.summary())

# Recursive forecasts AR(1)
# Combine last row of estimation_sample with hold_out_sample
combined_sample = pd.concat([estimation_sample.tail(1), hold_out_sample])
combined_sample.index = pd.date_range(start='1980', periods=len(combined_sample), freq='YE')


# Recursive 1-step-ahead forecasts
forecasts = []
combined_sample = pd.concat([estimation_sample, hold_out_sample])
combined_values = combined_sample['nr'].values

# Start from the end of estimation sample
start_idx = len(estimation_sample)

for t in range(start_idx, len(combined_sample)):
    # Fit AR(1) on all available data up to t-1
    model = ARIMA(combined_values[:t], order=(1,0,0)).fit()
    # Forecast 1-step ahead
    forecasts.append(model.forecast(steps=1)[0])

# Create DataFrame for plotting
forecasts_ar_1 = pd.DataFrame({
    't': np.arange(1981, 1981 + len(forecasts)),
    'forecasts': forecasts,
    'two_SE': 2 * np.std(estimation_sample['nr'])
})

# Plot forecasts with ±2 SE
plt.figure(figsize=(12,6))
plt.plot(forecasts_ar_1['t'], forecasts_ar_1['forecasts'], color='blue', label='Forecast')
plt.plot(forecasts_ar_1['t'], forecasts_ar_1['forecasts'] + forecasts_ar_1['two_SE'], color='red', linestyle='--', label='+2 SE')
plt.plot(forecasts_ar_1['t'], forecasts_ar_1['forecasts'] - forecasts_ar_1['two_SE'], color='red', linestyle='--', label='-2 SE')
plt.xlabel('Year')
plt.ylabel('Forecasted Value')
plt.title('Recursive 1-step-ahead AR(1) Forecasts')
plt.legend()
plt.show()


# Recursive ARMA(1,1)
# Initialize forecast DataFrame
n_est = len(estimation_sample)
n_hold_out = len(hold_out_sample)

one_step_ahead_forecasts = pd.DataFrame({
    't': np.arange(n_est, n_est + n_hold_out),
    'forecast': np.nan,
    'error': np.nan,
    'actual': hold_out_sample['nr'].values
})

# Extract ARMA coefficients
phi = arma_11.params.get('ar.L1', 0)       # AR(1) coefficient
theta = arma_11.params.get('ma.L1', 0)     # MA(1) coefficient
c = arma_11.params.get('const', 0)         # constant term

# Last residual and last observed value from estimation sample
last_residual = arma_11.resid.iloc[-1]
last_y = estimation_sample['nr'].iloc[-1]

# Recursive 1-step-ahead forecasts
for idx in one_step_ahead_forecasts.index:
    if idx == one_step_ahead_forecasts.index[0]:
        lag_y = last_y
        lag_error = last_residual
    else:
        lag_y = one_step_ahead_forecasts.loc[idx-1, 'forecast']
        lag_error = one_step_ahead_forecasts.loc[idx-1, 'error']
    
    # ARMA(1,1) forecast: forecast = c + phi*lag_y + theta*lag_error
    prediction_i = c + phi * lag_y + theta * lag_error

    # Store forecast and error
    one_step_ahead_forecasts.loc[idx, 'forecast'] = prediction_i
    one_step_ahead_forecasts.loc[idx, 'error'] = prediction_i - one_step_ahead_forecasts.loc[idx, 'actual']

# Prepare DataFrame for plotting
two_SE = 2 * np.std(estimation_sample['nr'])
forecasts_arma = pd.DataFrame({
    't': np.arange(1981, 1981 + n_hold_out),
    'forecasts': one_step_ahead_forecasts['forecast'].values,
    'two_SE': [two_SE]*n_hold_out
})

# Plot
plt.figure(figsize=(12,6))
plt.plot(forecasts_arma['t'], forecasts_arma['forecasts'], color='blue', label='Forecast')
plt.plot(forecasts_arma['t'], forecasts_arma['forecasts'] + forecasts_arma['two_SE'], color='red', linestyle='--', label='+2 SE')
plt.plot(forecasts_arma['t'], forecasts_arma['forecasts'] - forecasts_arma['two_SE'], color='red', linestyle='--', label='-2 SE')
plt.xlabel('Year')
plt.ylabel('Forecasted Value')
plt.title('Recursive 1-step-ahead ARMA(1,1) Forecasts')
plt.legend()
plt.show()



# Full ARMA(1,1) refit
arma_11_all = ARIMA(earthquakes['nr'], order=(1,0,1)).fit()
coefs_full = arma_11_all.params
coefs_full
lag_y = earthquakes['nr'].iloc[-1]
lag_err = arma_11_all.resid.iloc[-1]

future_preds = []
for t in range(2006,2021):
    pred = coefs_full[0] + coefs_full[1]*lag_y + coefs_full[2]*lag_err
    future_preds.append({'t': t, 'forecast': pred})
    lag_y = pred
    lag_err = 0

future_df = pd.DataFrame(future_preds)

plt.figure(figsize=(10,5))
plt.plot(future_df['t'], future_df['forecast'], marker='o', color='blue', label='Forecast')
plt.xlabel('Year')
plt.ylabel('Predicted Value')
plt.title('ARMA(1,1) Forecasts: 2006-2020')
plt.grid(True)
plt.legend()
plt.show()
