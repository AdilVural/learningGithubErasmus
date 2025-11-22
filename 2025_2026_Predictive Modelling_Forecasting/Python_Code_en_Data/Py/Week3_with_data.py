#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA

# Load data
gdp_data = pd.read_excel("../data/LA real GDP per capita.xlsx")

# Remove first year and create t = 1, 2, 3, ..., T
gdp_data = gdp_data[gdp_data['YEAR'] != 1950].copy()
gdp_data['t'] = np.arange(1, len(gdp_data)+1)

# Get growth
gdp_data['growth_brazil'] = gdp_data['BRAZIL'].pct_change() * 100

# Create line plot of growth
plt.figure(figsize=(10,6))
plt.plot(gdp_data['t'], gdp_data['growth_brazil'])
plt.ylim(-12, 12)
plt.xlabel('t')
plt.ylabel('Growth Brazil (%)')
plt.show()

# Autocorrelations
growth_brazil_nonan = gdp_data['growth_brazil'].dropna()
acf_growth_brazil = acf(growth_brazil_nonan, fft=False) 

# Calculate 2SE
two_SE = 2 / np.sqrt(len(growth_brazil_nonan))

# Combine into dataframe
acf_data = pd.DataFrame({
    'acf': acf_growth_brazil,
    'i': np.arange(len(acf_growth_brazil))
})
acf_data = acf_data[acf_data['i'] != 0]
acf_data['two_SE_plus'] = two_SE
acf_data['two_SE_minus'] = -two_SE

# Create plot
plt.figure(figsize=(10,6))
plt.stem(acf_data['i'], acf_data['acf'])
#plt.plot(acf_data['i'], acf_data['acf'], color='blue') #makes one series/line if preferred
plt.plot(acf_data['i'], acf_data['two_SE_plus'], color='green')
plt.plot(acf_data['i'], acf_data['two_SE_minus'], color='red')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.show()

# Partial autocorrelation
pacf_growth_brazil = pacf(growth_brazil_nonan)
pacf_data = pd.DataFrame({
    'pacf': pacf_growth_brazil[1:],
    'i': np.arange(1, len(pacf_growth_brazil))
})
pacf_data['two_SE_plus'] = two_SE
pacf_data['two_SE_minus'] = -two_SE

plt.figure(figsize=(10,6))
plt.stem(pacf_data['i'], pacf_data['pacf'])
#plt.plot(pacf_data['i'], pacf_data['acf'], color='blue') #makes one series/line if preferred
plt.plot(pacf_data['i'], pacf_data['two_SE_plus'], color='green')
plt.plot(pacf_data['i'], pacf_data['two_SE_minus'], color='red')
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.show()

# Estimation results AR(2)
ar_2_data = gdp_data.copy()
ar_2_data['lag_1'] = gdp_data['growth_brazil'].shift(1)
ar_2_data['lag_2'] = gdp_data['growth_brazil'].shift(2)

# Using ARIMA
ar_2 = ARIMA(gdp_data['growth_brazil'], order=(2,0,0)).fit()

# Airline revenues
revenue_per_passenger = pd.read_excel("../data/AEA revenue passenger kilometres.xlsx")
revenue_per_passenger = revenue_per_passenger.rename(columns={'Month':'month','AEA RPK TO':'revenue'})
revenue_per_passenger['month'] = pd.to_datetime(revenue_per_passenger['month'])

# Monthly growth
revenue_per_passenger['growth'] = revenue_per_passenger['revenue'].pct_change() * 100
revenue_per_passenger['log_growth'] = 100 * np.log(revenue_per_passenger['revenue']/revenue_per_passenger['revenue'].shift(1))

# ACF
acf_revenue = acf(revenue_per_passenger['growth'].dropna(), fft=False)

# Annual growth
revenue_per_passenger['annual_growth'] = revenue_per_passenger['revenue'].pct_change(12) * 100
revenue_per_passenger['log_annual_growth'] = 100 * np.log(revenue_per_passenger['revenue']/revenue_per_passenger['revenue'].shift(12))

acf_annual_growth = acf(revenue_per_passenger['annual_growth'].dropna(), fft=False)

# Plot revenue
plt.figure(figsize=(10,6))
plt.plot(revenue_per_passenger['month'], revenue_per_passenger['revenue'])
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.show()

# Double differences
revenue_per_passenger['first_difference'] = np.log(revenue_per_passenger['revenue']) - np.log(revenue_per_passenger['revenue'].shift(1))
revenue_per_passenger['second_difference'] = revenue_per_passenger['first_difference'] - revenue_per_passenger['first_difference'].shift(12)

# Drop NaN values for ACF
second_diff_nonan = revenue_per_passenger['second_difference'].dropna()

# ACF values (biased = matches R)
acf_vals = acf(second_diff_nonan, fft=False)

# R-style constant SE
N = len(second_diff_nonan)
two_SE = 2 / np.sqrt(N)

lags = np.arange(len(acf_vals))

plt.figure(figsize=(10,6))
plt.stem(lags, acf_vals)
plt.axhline(two_SE, color='red')
plt.axhline(-two_SE, color='red')
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.title("ACF with R-style constant CI")
plt.show()

# %%
