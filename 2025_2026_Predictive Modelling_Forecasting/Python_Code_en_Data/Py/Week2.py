#%%
# --------------------------------------------------------------
# Week 2 
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA

# --------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------

#INSERT DATA HERE
df = pd.read_excel(XX)
df = df.rename(columns={"Month": "month", "AEA RPK TO": "revenue"})
df["month"] = pd.to_datetime(df["month"])
df["log_revenue"] = np.log(df["revenue"])

# --------------------------------------------------------------
# 2. Autocorrelation of log-levels
# --------------------------------------------------------------

autocorr_values = acf(df["log_revenue"], fft=False)
two_SE = 2 / np.sqrt(len(df))

plot_df = pd.DataFrame({
    "acf": autocorr_values,
    "time": np.arange(1, len(autocorr_values)+1),
    "two_SE": two_SE
})

plt.figure()
plt.plot(plot_df["time"], plot_df["acf"], )
plt.axhline(two_SE, color="red")
plt.ylabel("ACF")
plt.xlabel("Lag")
plt.title("log monthly revenue-passenger kilometers")
plt.show()

# --------------------------------------------------------------
# 3. Own ACF function (manual)
# --------------------------------------------------------------

def own_acf(time_series, k_series=range(1, 25)):
    series = np.array(time_series)
    T = len(series)
    y_bar = series.mean()
    gamma_0 = series.var() * ((T - 1) / T)

    rho = []

    for k in k_series:
        y_t = series[k:]
        y_k = series[:-k]
        gamma_k = np.sum((y_t - y_bar) * (y_k - y_bar)) / T
        rho.append(gamma_k / gamma_0)

    return np.array(rho)

# --------------------------------------------------------------
# 4. Monthly growth rates
# --------------------------------------------------------------

df["mgr"] = 100 * (np.log(df["revenue"] / df["revenue"].shift(1)))
df_mg = df.dropna(subset=["mgr"])

monthly_acf = acf(df_mg["mgr"], nlags=24, fft=False)
monthly_own = own_acf(df_mg["mgr"])

plot_monthly = pd.DataFrame({
    "acf": monthly_acf[1:],       # drop lag 0
    "time": np.arange(1, len(monthly_acf)),
    "two_SE": two_SE,
    "min_two_SE": -two_SE
})

plt.figure()
plt.plot(plot_monthly["time"], plot_monthly["acf"], color="blue")
plt.axhline(two_SE, color="red")
plt.axhline(-two_SE, color="green")
plt.title("Monthly growth rates, revenue-passenger kilometres")
plt.show()

# --------------------------------------------------------------
# 5. Histogram of unconditional distribution
# --------------------------------------------------------------

plt.figure()
sns.histplot(df_mg["mgr"] / 100, bins=20)
plt.title("Monthly growth rates of revenue-passenger kilometres")
plt.show()

# --------------------------------------------------------------
# 6. AR(1) effect of large shock
# (x1 must be provided in the environment by running ex_2.py file first)
# --------------------------------------------------------------

# Example placeholder for x1
# Replace with actual series as in your R environment:
# x1 = your_series_here

ar_model = ARIMA(x1, order=(1, 0, 0)).fit()
fitted_vals = ar_model.fittedvalues

ar_plot_df = pd.DataFrame({
    "x": x1,
    "y": fitted_vals,
    "n": np.arange(1, len(x1)+1)
})

plt.plot(ar_plot_df["n"], ar_plot_df["x"], color="red")
plt.plot(ar_plot_df["n"], ar_plot_df["y"], color="blue")
plt.show()

# --------------------------------------------------------------
# 7. Empirical ACFs (acf.y1, acf.y2, acf.y3 must be defined, as in ex_2.py)
# --------------------------------------------------------------

# Example placeholders:
# acf_y1 = acf(y1)
# acf_y2 = acf(y2)
# acf_y3 = acf(y3)

acf_y = pd.DataFrame({
    "acf_y1": acf_y1,
    "acf_y2": acf_y2,
    "acf_y3": acf_y3,
    "n": np.arange(1, len(acf_y1)+1)
})
plt.figure()
plt.plot(acf_y["n"], acf_y["acf_y1"], color="blue")
plt.plot(acf_y["n"], acf_y["acf_y2"], color="red")
plt.plot(acf_y["n"], acf_y["acf_y3"], color="green")
plt.show()

# --------------------------------------------------------------
# 8. Mean reversion simulation
# --------------------------------------------------------------

T = 200
e = np.random.normal(0, 1, T)

y1 = np.zeros(T)
y2 = np.zeros(T)

for t in range(1, T):
    y1[t] = 0.8 * y1[t-1] + e[t]
    y2[t] = y2[t-1] + e[t]

plt.figure()
plt.plot(y1)
plt.title("Mean-reverting AR(1) process, φ = 0.8")
plt.show()

plt.figure()
plt.plot(y2)
plt.title("Random walk (φ = 1)")
plt.show()

# --------------------------------------------------------------
# 9. Theoretical ACF for AR(1), φ = 0.8
# --------------------------------------------------------------

phi = 0.8
rho_k = phi ** np.arange(1, 21)

plt.figure()
plt.plot(rho_k)
plt.title("Theoretical ACF of AR(1), φ = 0.8")
plt.show()
