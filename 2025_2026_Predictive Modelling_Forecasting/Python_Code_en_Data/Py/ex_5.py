#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# ----------------------------
# Lecture 5 Exercise 1
# ----------------------------

# Create time series
T = 300
trend = np.arange(T)
dt = 0.1 * trend

e = np.random.normal(0, 1, T)

x = np.zeros(T)
y = np.zeros(T)

for i in range(1, T):
    x[i] = 0.1*trend[i] + 0.5*(x[i-1] - 0.1*trend[i-1]) + e[i]
    y[i] = y[i-1] + 0.1 + e[i]

# Put into DataFrame
series1 = pd.DataFrame({
    'time': np.arange(1, T+1),
    'x': x,
    'y': y,
    'dt': dt
})

# Plot
plt.figure(figsize=(10,6))
plt.plot(series1['time'], series1['x'], color='red', label='x')
plt.plot(series1['time'], series1['y'], color='blue', label='y')
plt.plot(series1['time'], series1['dt'], color='green', label='dt')
plt.xlabel('time')
plt.ylabel('x, y, dt')
plt.legend()
plt.title('Lecture 5 Exercise 1')
plt.show()

# ----------------------------
# Lecture 5 Exercise 2
# ----------------------------

# Create time series with shock
add = np.zeros(T)
add[149:] = 5  
time = np.arange(1, T+1)

e = np.random.normal(0, 1, T)

x = np.zeros(T)
y = np.zeros(T)

for i in range(1, T):
    x[i] = add[i] + 0.8*(x[i-1] - add[i]) + e[i]
    y[i] = 0.8*y[i-1] + e[i]

# Put into DataFrame
series2 = pd.DataFrame({
    'time': time,
    'x': x,
    'y': y
})

# Plot
plt.figure(figsize=(10,6))
plt.plot(series2['time'], series2['x'], color='red', label='x')
plt.plot(series2['time'], series2['y'], color='blue', label='y')
plt.xlabel('time')
plt.ylabel('x, y')
plt.legend()
plt.title('Lecture 5 Exercise 2')
plt.show()

# ----------------------------
# Augmented Dickey-Fuller test
# ----------------------------

lag_order = 1

test_x = adfuller(x, maxlag=lag_order, autolag=None)
test_y = adfuller(y, maxlag=lag_order, autolag=None)

print("ADF test for x:")
print(f"ADF Statistic: {test_x[0]}")
print(f"p-value: {test_x[1]}\n")

print("ADF test for y:")
print(f"ADF Statistic: {test_y[0]}")
print(f"p-value: {test_y[1]}")


# %%
