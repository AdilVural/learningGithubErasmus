#%%
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import datetime
from statsmodels.tsa.stattools import acf

sns.set(style="whitegrid")

# ------------------ A trend ------------------

# Load GDP data
gdp_data = pd.read_excel("../data/LA real GDP per capita.xlsx")

# Remove first year and turn into long format
gdp_data = gdp_data[gdp_data['YEAR'] != 1950]
gdp_data['CHILE'] = pd.to_numeric(gdp_data['CHILE'])
gdp_long = gdp_data.melt(id_vars='YEAR', var_name='country', value_name='GDP')

# Plot GDP trend
plt.figure(figsize=(10,5))
sns.lineplot(data=gdp_long, x='YEAR', y='GDP', hue='country')
plt.title("GDP per capita over time")
plt.show()

# ------------------ Industrial production ------------------

industrial_production = pd.read_excel("../data/US industrial production.xlsx", skiprows=1)
industrial_production = industrial_production.rename(columns={
    'Datastream Code': 'date',
    'USIPTOT.G': 'usiptot_g',
    'USIPTOT.H': 'usiptot_h'
})

def parse_quarter_date(q):
    year = int(q[-4:])
    if 'Q1' in q: return datetime.date(year,1,1)
    if 'Q2' in q: return datetime.date(year,4,1)
    if 'Q3' in q: return datetime.date(year,7,1)
    if 'Q4' in q: return datetime.date(year,10,1)
    return pd.NaT

industrial_production['date'] = industrial_production['date'].apply(parse_quarter_date)

plt.figure(figsize=(10,5))
plt.plot(industrial_production['date'], industrial_production['usiptot_g'])
plt.title("US Industrial Production")


# ------------------ NL stock of motorcycles ------------------

stock_motorcycles = pd.read_excel("../data/NL stock of motorcycles.xlsx")
stock_motorcycles = stock_motorcycles.rename(columns={'YEAR':'year', 'MOTORSTOCK':'stock'})

plt.figure(figsize=(10,5))
plt.plot(stock_motorcycles['year'], stock_motorcycles['stock'])
plt.title("Stock of Motorcycles in NL")
plt.show()

# ------------------ Stationary AR(1) no trend ------------------

mu = 10
phi_1 = 0.8
Time = 200

e = np.random.normal(0,1,(Time,4))
x = np.full((Time,4), mu)

for i in range(1, Time):
    x[i,0] = phi_1*(x[i-1,0]-mu)+e[i,0]+mu
    x[i,1] = phi_1*(x[i-1,1]-mu)+e[i,1]+mu
    x[i,2] = phi_1*(x[i-1,2]-mu)+e[i,2]+mu
    x[i,3] = phi_1*(x[i-1,3]-mu)+e[i,3]+mu

fig, axs = plt.subplots(2,2, figsize=(12,8))
for j, ax in enumerate(axs.flatten()):
    ax.plot(range(1,Time+1), x[:,j])
plt.show()

# ------------------ Stationary AR(1) with trend ------------------

delta = 0.1
x_trend = np.full((Time,4), mu)

for i in range(1, Time):
    for j in range(4):
        x_trend[i,j] = phi_1*(x_trend[i-1,j]-mu-delta*(i-1)) + e[i,j] + mu + delta*i

fig, axs = plt.subplots(2,2, figsize=(12,8))
for j, ax in enumerate(axs.flatten()):
    ax.plot(range(1,Time+1), x_trend[:,j])
plt.show()

# ------------------ Trend-stationarity ------------------

gdp_data_trend = gdp_data.copy()
gdp_data_trend['log_argentina'] = np.log(gdp_data_trend['ARGENTINA'])
gdp_data_trend['t'] = np.arange(1, len(gdp_data_trend)+1)

model = ols("log_argentina ~ t", data=gdp_data_trend).fit()
gdp_data_trend['fit'] = model.fittedvalues
gdp_data_trend['residual'] = gdp_data_trend['log_argentina'] - gdp_data_trend['fit']

plt.figure(figsize=(10,5))
plt.plot(gdp_data_trend['t'], gdp_data_trend['log_argentina'], color='red', label='Actual')
plt.plot(gdp_data_trend['t'], gdp_data_trend['fit'], color='green', label='Fit')
plt.legend()
plt.show()

# ------------------ Industrial production reversion ------------------

industrial_production['log_usip'] = np.log(industrial_production['usiptot_g'])
industrial_production['t'] = np.arange(1, len(industrial_production)+1)
model_ip = ols("log_usip ~ t", data=industrial_production).fit()
industrial_production['fit'] = model_ip.fittedvalues
industrial_production['residual'] = industrial_production['log_usip'] - industrial_production['fit']

plt.figure(figsize=(10,5))
plt.plot(industrial_production['t'], industrial_production['log_usip'], color='red', label='Actual')
plt.plot(industrial_production['t'], industrial_production['fit'], color='green', label='Fit')
plt.legend()
plt.show()

# ---------------- Trend in a stationary AR(1) ---------------- #

delta = 0.1
phi_1 = 0.99
mu = 10

# Create time series
Time = 200
time = np.arange(1, Time + 1)

# Generate random disturbances
e = pd.DataFrame({
    "e_1": np.random.normal(0, 1, Time),
    "e_2": np.random.normal(0, 1, Time),
    "e_3": np.random.normal(0, 1, Time),
    "e_4": np.random.normal(0, 1, Time)
})

# Initialize series
x1_ts = np.full(Time, mu)
x2_ts = np.full(Time, mu)
x3_ts = np.full(Time, mu)
x4_ts = np.full(Time, mu)

# Generate AR(1) series with trend
for i in range(1, Time):
    x1_ts[i] = phi_1 * (x1_ts[i-1] - mu - delta*(i-1)) + e.e_1[i] + mu + delta*i
    x2_ts[i] = phi_1 * (x2_ts[i-1] - mu - delta*(i-1)) + e.e_2[i] + mu + delta*i
    x3_ts[i] = phi_1 * (x3_ts[i-1] - mu - delta*(i-1)) + e.e_3[i] + mu + delta*i
    x4_ts[i] = phi_1 * (x4_ts[i-1] - mu - delta*(i-1)) + e.e_4[i] + mu + delta*i

# Put into DataFrame
series_ts = pd.DataFrame({
    "time": time,
    "x1_ts": x1_ts,
    "x2_ts": x2_ts,
    "x3_ts": x3_ts,
    "x4_ts": x4_ts
})

# Fit regressions
def fit_series(y):
    X = sm.add_constant(series_ts["time"])
    model = sm.OLS(y, X).fit()
    return model

model_x1 = fit_series(series_ts["x1_ts"])
model_x2 = fit_series(series_ts["x2_ts"])
model_x3 = fit_series(series_ts["x3_ts"])
model_x4 = fit_series(series_ts["x4_ts"])

# Add fitted values and residuals
series_ts["fit_x1"] = model_x1.fittedvalues
series_ts["residual_x1"] = series_ts["x1_ts"] - series_ts["fit_x1"]

series_ts["fit_x2"] = model_x2.fittedvalues
series_ts["residual_x2"] = series_ts["x2_ts"] - series_ts["fit_x2"]

series_ts["fit_x3"] = model_x3.fittedvalues
series_ts["residual_x3"] = series_ts["x3_ts"] - series_ts["fit_x3"]

series_ts["fit_x4"] = model_x4.fittedvalues
series_ts["residual_x4"] = series_ts["x4_ts"] - series_ts["fit_x4"]

# Plot for the first series (x1_ts)
plt.figure(figsize=(10,5))
plt.plot(series_ts["time"], series_ts["fit_x1"], color="green", label="Fit")
plt.plot(series_ts["time"], series_ts["x1_ts"], color="red", label="Actual")
plt.plot(series_ts["time"], series_ts["residual_x1"], color="blue", label="Residual")
plt.legend()
plt.title("Trend in a stationary AR(1) – Series 1")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

# Autocorrelations ------------------------------------------------------------------------------------------------
# Note that this only hodls for phi_1 = 1
delta = 0.1
y_0 = 10

def exp_y_t(t):
    return y_0 + delta * t

# Generate y_t (t = 1 to 45)
t_values = np.arange(1, 46)
y_t = exp_y_t(t_values)

n = len(y_t)

# Compute ACF values (match R: no FFT)
acf_values = acf(y_t, nlags=20, fft=False)

# R's confidence interval
ci = 1.96 / np.sqrt(n)

lags = np.arange(len(acf_values))

plt.figure(figsize=(8, 4))
plt.stem(lags, acf_values)

# Add R-style CI
plt.axhline(ci, color='red', linestyle='--', label=f"R CI = ±{ci:.3f}")
plt.axhline(-ci, color='red', linestyle='--')

# Zero line
plt.axhline(0, color='black', linewidth=1)

plt.title("ACF with R-style 95% Confidence Intervals")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.legend()
plt.show()


# ------------------ Stochastic trend AR(1) ------------------
delta = 0.1
phi_1 = 1
mu = np.array([-2, -4, 10, 6])

Time = 200
time = np.arange(1, Time + 1)

# --------------------------------
# Random disturbances
# --------------------------------

np.random.seed(123)  
e = pd.DataFrame({
    "e_1": np.random.normal(0, 1, Time),
    "e_2": np.random.normal(0, 1, Time),
    "e_3": np.random.normal(0, 1, Time),
    "e_4": np.random.normal(0, 1, Time),
})

# --------------------------------
# Generate stochastic-trend AR(1)
# --------------------------------

x_st = np.zeros((Time, 4))
x_st[0, :] = mu  

for i in range(1, Time):
    x_st[i, 0] = phi_1 * (x_st[i-1, 0] - mu[0] - delta*(i-1)) + e.iloc[i, 0] + mu[0] + delta*i
    x_st[i, 1] = phi_1 * (x_st[i-1, 1] - mu[1] - delta*(i-1)) + e.iloc[i, 1] + mu[1] + delta*i
    x_st[i, 2] = phi_1 * (x_st[i-1, 2] - mu[2] - delta*(i-1)) + e.iloc[i, 2] + mu[2] + delta*i
    x_st[i, 3] = phi_1 * (x_st[i-1, 3] - mu[3] - delta*(i-1)) + e.iloc[i, 3] + mu[3] + delta*i

# --------------------------------
# Build DataFrame
# --------------------------------

series_st = pd.DataFrame({
    "time": time,
    "x1_st": x_st[:, 0],
    "x2_st": x_st[:, 1],
    "x3_st": x_st[:, 2],
    "x4_st": x_st[:, 3],
})

# --------------------------------
# Linear regression fits 
# --------------------------------

def fit_lm(y, x):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model.params  # intercept and slope

b1 = fit_lm(series_st["x1_st"], series_st["time"])
b2 = fit_lm(series_st["x2_st"], series_st["time"])
b3 = fit_lm(series_st["x3_st"], series_st["time"])
b4 = fit_lm(series_st["x4_st"], series_st["time"])

# Add fitted and residuals columns
series_st["x1_fit"] = b1[0] + b1[1] * series_st["time"]
series_st["x2_fit"] = b2[0] + b2[1] * series_st["time"]
series_st["x3_fit"] = b3[0] + b3[1] * series_st["time"]
series_st["x4_fit"] = b4[0] + b4[1] * series_st["time"]

series_st["x1_residual"] = series_st["x1_st"] - series_st["x1_fit"]
series_st["x2_residual"] = series_st["x2_st"] - series_st["x2_fit"]
series_st["x3_residual"] = series_st["x3_st"] - series_st["x3_fit"]
series_st["x4_residual"] = series_st["x4_st"] - series_st["x4_fit"]

# --------------------------------
# Plot 
# --------------------------------

plt.figure(figsize=(10, 5))
plt.plot(series_st["time"], series_st["x1_fit"], label="x1_fit", color="green")
plt.plot(series_st["time"], series_st["x1_st"], label="x1_st", color="red")
plt.plot(series_st["time"], series_st["x1_residual"], label="x1_residual", color="blue")

plt.legend()
plt.xlabel("time")
plt.title("Stochastic Trend AR(1) — x1")
plt.show()


# ------------------ US industrial production logs ------------------

industrial_production['log_usip_sa'] = 4*100*np.log(industrial_production['usiptot_g']/industrial_production['usiptot_g'].shift(4))
industrial_production['first_difference_log'] = industrial_production['log_usip_sa'] - industrial_production['log_usip_sa'].shift(1)

industrial_production_lm = ols("first_difference_log ~ log_usip_sa.shift(1) + t + first_difference_log.shift(1) + first_difference_log.shift(2)", 
                                data=industrial_production).fit()
print(industrial_production_lm.summary())

# ------------------ Dow Jones returns ------------------

dow_jones = pd.read_excel("../data/US Dow Jones Industrials index.xlsx", skiprows=1)
dow_jones = dow_jones.rename(columns={'Datastream':'date','DJINDUS':'index'})
dow_jones['date'] = pd.to_datetime(dow_jones['date'])
dow_jones = dow_jones[(dow_jones['date'] >= "1989-12-29") & (dow_jones['date'] <= "2005-12-30")]
dow_jones.loc[:, "index"] = np.log(dow_jones["index"]) * 100

dow_jones.loc[:,'first_difference'] = dow_jones['index'].shift(1) - dow_jones['index']
dow_jones.loc[:,'lag_index'] = dow_jones['index'].shift(1)

dow_jones_lm = ols("first_difference ~ lag_index", data=dow_jones).fit()
print(dow_jones_lm.summary())

# %%
