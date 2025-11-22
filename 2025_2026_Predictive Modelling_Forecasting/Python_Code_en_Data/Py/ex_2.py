#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

# Exercise 1 ---------------------------------------------------------------

# Create time series
T = 200
add = 20  # shock
time = np.arange(1, T+1)

# generate a sequence of random disturbances
e = np.random.normal(0, 1, T)

# add a shock at t = 100
e[99] += add  # Python is zero-indexed

x1 = np.zeros(T)
x2 = np.zeros(T)
x3 = np.zeros(T)

# generate time series
for i in range(1, T):
    x1[i] = 0.5 * x1[i-1] + e[i]
    x2[i] = 0.9 * x2[i-1] + e[i]
    x3[i] = -1 * x3[i-1] + e[i]

# put time series in a DataFrame
series = pd.DataFrame({
    'time': time,
    'x1': x1,
    'x2': x2,
    'x3': x3
})

# create plot
plt.figure(figsize=(10, 6))
plt.plot(series['time'], series['x1'], label='x1')
plt.plot(series['time'], series['x2'], label='x2')
plt.plot(series['time'], series['x3'], label='x3')
plt.xlabel('time')
plt.ylabel('x1, x2, x3')
plt.legend()
plt.show()


# Exercise 2 ---------------------------------------------------------------

# Create time series
T = 200
time = np.arange(1, T+1)

e = np.random.normal(0, 1, T)

y1 = np.zeros(T)
y2 = np.zeros(T)
y3 = np.zeros(T)

for i in range(1, T):
    y1[i] = 0.5 * y1[i-1] + e[i]
    y2[i] = 0.9 * y2[i-1] + e[i]
    y3[i] = y3[i-1] + e[i]

# plot ACFs
acf_y1 = acf(y1, fft=False)
acf_y2 = acf(y2, fft=False)
acf_y3 = acf(y3, fft=False)

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plot_acf(y1, ax=plt.gca())
plt.title('ACF of y1')

plt.subplot(2, 2, 2)
plot_acf(y2, ax=plt.gca())
plt.title('ACF of y2')

plt.subplot(2, 2, 3)
plot_acf(y3, ax=plt.gca())
plt.title('ACF of y3')

plt.tight_layout()
plt.show()

# %%
