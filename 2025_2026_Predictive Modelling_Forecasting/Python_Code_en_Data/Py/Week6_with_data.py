#%%
# week6.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import re

# ---------------------------------------
# Helper: convert "YYYY QX" to datetime
# ---------------------------------------
def quarter_to_date(qstring):
    year = int(re.search(r"\d{4}", qstring).group())
    quarter = re.search(r"Q[1-4]", qstring).group()
    if quarter == "Q1":
        return datetime(year, 1, 1)
    elif quarter == "Q2":
        return datetime(year, 4, 1)
    elif quarter == "Q3":
        return datetime(year, 7, 1)
    elif quarter == "Q4":
        return datetime(year, 10, 1)
    else:
        return None

# ---------------------------------------------------
# US Industrial Production
# ---------------------------------------------------
industrial_production = (
    pd.read_excel("../data/US industrial production.xlsx", skiprows=1)
    .rename(columns={"Datastream Code": "date",
                     "USIPTOT.G": "usiptot_g",
                     "USIPTOT.H": "usiptot_h"})
)

industrial_production["quarter"] = industrial_production["date"]
industrial_production["date"] = industrial_production["date"].apply(quarter_to_date)
industrial_production["quarter"] = industrial_production["quarter"].str.extract(r"(Q\d)")
industrial_production["log_usip"] = np.log(industrial_production["usiptot_h"])

# -----------------------------
# Plots 
# -----------------------------
plt.figure()
plt.plot(industrial_production["date"], industrial_production["log_usip"])
plt.title("US IP: log_usip")
plt.show()

plt.figure()
for q in ["Q1", "Q2", "Q3", "Q4"]:
    dfq = industrial_production[industrial_production["quarter"] == q]
    plt.plot(dfq["date"], dfq["log_usip"], label=q)
plt.legend()
plt.title("US IP by Quarter")
plt.show()

# Quarterly growth
industrial_production["quarterly_growth"] = \
    industrial_production["log_usip"].diff()

plt.figure()
for q in ["Q1","Q2","Q3","Q4"]:
    dfq = industrial_production[industrial_production["quarter"] == q]
    plt.plot(dfq["date"], dfq["quarterly_growth"], label=q)
plt.legend()
plt.title("Quarterly Growth")
plt.show()

# ---------------------------------------------------
# Deterministic seasonality
# ---------------------------------------------------
for q in ["Q1","Q2","Q3","Q4"]:
    industrial_production[f"D{q[-1]}"] = (industrial_production["quarter"] == q).astype(int)

# Build regression model
df = industrial_production.copy()
df["lag1"] = df["quarterly_growth"].shift(1)
df["lag2"] = df["quarterly_growth"].shift(2)
df["lag3"] = df["quarterly_growth"].shift(3)
df["lag4"] = df["quarterly_growth"].shift(4)
df["lag5"] = df["quarterly_growth"].shift(5)

X = df[["D1","D2","D3","D4","lag1","lag2","lag3","lag4","lag5"]]
y = df["quarterly_growth"]
XY = pd.concat([y, X], axis=1).dropna()

model = sm.OLS(XY["quarterly_growth"], XY[X.columns]).fit()
print(model.summary())


# ---------------------------------------------------
# Stochastic seasonality
# ---------------------------------------------------
industrial_production["annual_growth"] = \
    industrial_production["log_usip"].diff(4)

plt.figure()
plt.plot(industrial_production["date"], industrial_production["annual_growth"])
plt.title("Annual Growth (y_t - y_(t-4))")
plt.show()

# ARIMA annual growth
arima_annual = ARIMA(industrial_production["annual_growth"].dropna(),
                     order=(2,0,0),
                     seasonal_order=(0,0,1,4)).fit()
print(arima_annual.summary())

# Double differencing
industrial_production["double_dif"] = \
    industrial_production["quarterly_growth"] - industrial_production["quarterly_growth"].shift(4)

arima_dd = ARIMA(industrial_production["annual_growth"].dropna(),
                 order=(0,0,1),
                 seasonal_order=(0,0,1,4)).fit()
print(arima_dd.summary())

# Selection between filters
df = industrial_production.copy()
df["lag_annual_growth"] = df["annual_growth"].shift(1)
df["lag_quarterly_growth_4"] = df["quarterly_growth"].shift(4)
for i in range(1,5):
    df[f"lag_double_dif_{i}"] = df["double_dif"].shift(i)

cols = ["double_dif","lag_annual_growth","lag_quarterly_growth_4",
        "lag_double_dif_1","lag_double_dif_2","lag_double_dif_3","lag_double_dif_4"]

# Concatenate and drop NA to align indices
XY = df[["double_dif"] + cols[1:]].dropna()  # exclude double_dif from X columns
# Add constant for intercept
X = sm.add_constant(XY[cols[1:]])
y = XY["double_dif"]

# Fit OLS
usa_ip_dif = sm.OLS(y, X).fit()
print(usa_ip_dif.summary())

# ----------------------------------------------------
# Seasonal unit roots model
# ----------------------------------------------------
df = industrial_production.copy()
df["t"] = np.arange(len(df))

df["pi_1_term"] = df["log_usip"].shift(1) + df["log_usip"].shift(2) + df["log_usip"].shift(3) + df["log_usip"].shift(4)
df["pi_2_term"] = -df["log_usip"].shift(1) + df["log_usip"].shift(2) - df["log_usip"].shift(3) + df["log_usip"].shift(4)
df["pi_3_term"] = -df["log_usip"].shift(1) + df["log_usip"].shift(3)
df["pi_4_term"] = -df["log_usip"].shift(2) + df["log_usip"].shift(4)

for k in range(1,8):
    df[f"annual_growth_lag_{k}"] = df["annual_growth"].shift(k)

cols = ["D1","D2","D3","D4","t",
        "pi_1_term","pi_2_term","pi_3_term","pi_4_term"] + \
       [f"annual_growth_lag_{k}" for k in range(1,8)]

# Align endog (y) and exog (X) by dropping NA together
XY = df[["annual_growth"] + cols].dropna()

# Add constant for intercept
X = XY[cols]
y = XY["annual_growth"]

# Fit OLS
unit_root_mod = sm.OLS(y, X).fit()
print(unit_root_mod.summary())

# Restricted model for F-test
restricted_cols = [c for c in cols if c not in ["pi_2_term","pi_3_term","pi_4_term"]]

# Align endog (y) and exog (X) by dropping NA together
XY = df[["annual_growth"] + restricted_cols].dropna()

# Add constant for intercept
X = sm.add_constant(XY[restricted_cols])
y = XY["annual_growth"]

# Fit OLS
restricted_mod = sm.OLS(y, X).fit()

SSE_r = sum(restricted_mod.resid**2)
SSE_ur = sum(unit_root_mod.resid**2)
q = 3
k = unit_root_mod.df_model
n = unit_root_mod.nobs

F_test_val = ((SSE_r - SSE_ur)/q) / (SSE_ur/(n-k))
print("F-test:", F_test_val)

# ---------------------------------------------------
# Periodic autoregression
# ---------------------------------------------------
df = industrial_production.copy()
df["lag_usip"] = df["log_usip"].shift(1)

for q in ["Q1","Q2","Q3","Q4"]:
    df[f"D{q[-1]}"] = (df["quarter"] == q).astype(int)
    df[f"{q}_lag"] = df[f"D{q[-1]}"] * df["lag_usip"]

# Select columns for regression: dummies + interactions
cols = ["D1","D2","D3","D4","Q1_lag","Q2_lag","Q3_lag","Q4_lag"]

# Rename interaction columns to match cols list
df.rename(columns={"Q1_lag":"D1_lag_usip",
                   "Q2_lag":"D2_lag_usip",
                   "Q3_lag":"D3_lag_usip",
                   "Q4_lag":"D4_lag_usip"}, inplace=True)

cols = ["D1","D2","D3","D4","D1_lag_usip","D2_lag_usip","D3_lag_usip","D4_lag_usip"]

# Align y and X
XY = df[["log_usip"] + cols].dropna()
X = XY[cols]
y = XY["log_usip"]

# Fit OLS
periodic_model = sm.OLS(y, X).fit()
print(periodic_model.summary())

# ---------------------------------------------------
# French Industrial Production
# ---------------------------------------------------
fr = (
    pd.read_excel("../data/FR industrial production.xlsx", skiprows=1)
    .rename(columns={"Datastream Code":"date","FRIPTOT.G":"frip"})
)

fr["quarter"] = fr["date"]
fr["date"] = fr["date"].apply(quarter_to_date)
fr["quarter"] = fr["quarter"].str.extract(r"(Q\d)")
fr["log_frip"] = np.log(fr["frip"])

plt.figure()
plt.plot(fr["date"], fr["frip"])
plt.title("FR Industrial Production")
plt.show()

plt.figure()
plt.plot(fr["date"], fr["log_frip"].diff())
plt.title("French Quarterly Growth")
plt.show()

# ---------------------------------------------------
# Dow Jones Black Monday
# ---------------------------------------------------
dj = (
    pd.read_excel("../data/US Dow Jones Industrials index.xlsx", skiprows=1)
    .rename(columns={"Datastream":"date","DJINDUS":"index"})
)

dj["date"] = pd.to_datetime(dj["date"])
dj = dj[(dj["date"] >= "1980-12-29") & (dj["date"] <= "2005-12-30")]
dj["growth_rate"] = np.log(dj["index"]) - np.log(dj["index"].shift(1))

plt.figure()
plt.plot(dj["date"], dj["growth_rate"])
plt.title("Dow Jones Growth Rates")
plt.show()

# ---------------------------------------------------
# Additive Outlier Simulation
# ---------------------------------------------------
time = 200
tau = 100
sd = 1
phi_1 = 0.8
xie = 20

e = np.random.normal(0, sd, time)
x = np.zeros(time)
y = np.zeros(time)

for i in range(1, time):
    x[i] = phi_1 * x[i-1] + e[i]
    y[i] = x[i] + (xie if i == tau else 0)

plt.figure()
plt.plot(x, label="x")
plt.plot(y, label="y")
plt.legend()
plt.title("Additive Outlier Example")
plt.show()

plt.figure()
plt.plot(y[:-1], y[1:], linestyle='-')
plt.title("Phase Plot of y (AO ignored)")
plt.show()

# ---------------------------------------------------
# 9/11 Model
# ---------------------------------------------------
air = (
    pd.read_excel("../data/AEA revenue passenger kilometres.xlsx")
    .rename(columns={"Month":"month","AEA RPK TO":"revenue"})
)

air["month"] = pd.to_datetime(air["month"])
air["log_revenue"] = np.log(air["revenue"])
air["D911"] = (air["month"] >= "2001-09-11").astype(int)
air["t"] = np.arange(1, len(air)+1)
air["t_D911"] = air["t"] * air["D911"]

X = air[["t","t_D911"]]
y = air["revenue"]

arima_911 = ARIMA(y, order=(0,0,0),
                  seasonal_order=(1,0,0,12),
                  exog=X).fit()

print(arima_911.summary())

# %%
