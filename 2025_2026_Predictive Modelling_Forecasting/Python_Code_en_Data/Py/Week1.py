#%%
# Python code week 1
# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import re

sns.set_style("whitegrid")


# One way to describe a time series with a trend-------------------------------------------------------------------
# INSERT DATA HERE
data = XX

# Remove first year and create t = 1,...,T
data = data[data["YEAR"] != 1950].copy()
data["t"] = range(1, len(data) + 1)
# Create the transformed variables
data["log_ARG_100"] = np.log(data["ARGENTINA"]) * 100
data["log_BRA_100"] = np.log(data["BRAZIL"]) * 100
data["log_CHI_100"] = np.log(data["CHILE"]) * 100
data["log_COL_100"] = np.log(data["COLOMBIA"]) * 100
data["log_MEX_100"] = np.log(data["MEXICO"]) * 100

# Trend regressions for each country
ols_argentina = smf.ols("log_ARG_100 ~ t", data=data).fit()
ols_brazil    = smf.ols("log_BRA_100 ~ t", data=data).fit()
ols_chile     = smf.ols("log_CHI_100 ~ t", data=data).fit()
ols_colombia  = smf.ols("log_COL_100 ~ t", data=data).fit()
ols_mexico    = smf.ols("log_MEX_100 ~ t", data=data).fit()

print(ols_argentina.summary())



# The direction of a trend can change -----------------------------------------------------------------------------
# INSERT DATA HERE
stock_motorcycles = XX
stock_motorcycles = stock_motorcycles.rename(columns={"YEAR": "year", 
                                                      "MOTORSTOCK": "stock"})

plt.figure()
sns.lineplot(data=stock_motorcycles, x="year", y="stock")
plt.title("Stock of Motorcycles in NL")
plt.show()


# UK consumption data ---------------------------------------------------------------------------------------------
# INSERT DATA HERE
household = (pd.read_excel(XX)
               .query("Name != 'Datastream Code'")
               .rename(columns={
                   "Name": "period",
                   "UK FINAL CONSMPTN.EXPENDITURE: HOUSEHOLD - NATIONAL CONCEPT CON (SEASONALLY ADJUSTED)": 
                       "consumption_seasonally",
                   "UK HOUSEHOLD FINAL CONSMPTN. EXPENDITURE - NATIONAL CONCEPT CON (NOT SEASONALLY ADJUSTED)":
                       "consumption_not_seasonally"
               })[["period", "consumption_seasonally", "consumption_not_seasonally"]])

# create quarter dummy
household["quarter"] = household["period"].str.extract(r"(Q\d)")
household["t"] = range(1, len(household) + 1)

# Seasonal dummies
for q in ["Q1", "Q2", "Q3", "Q4"]:
    household[f"D{q[1]}"] = (household["quarter"] == q).astype(int)

# Convert to numeric
household["consumption_not_seasonally"] = pd.to_numeric(
    household["consumption_not_seasonally"], errors="coerce"
)

# dep_var_ns = 400 × log(c_t / c_{t-1})
household["dep_var_ns"] = 400 * np.log(
    household["consumption_not_seasonally"] /
    household["consumption_not_seasonally"].shift(1)
)

# Filter t ≥ 21
household_reg = household[household["t"] >= 21]

dummy_reg = smf.ols("dep_var_ns ~ D1 + D2 + D3 + D4 - 1", data=household_reg).fit()
print(dummy_reg.summary())


# Annual growth rates ---------------------------------------------------------------------------------------------
household["annual_growth"] = 100 * np.log(
    household["consumption_not_seasonally"] /
    household["consumption_not_seasonally"].shift(4)
)


# Aberrant observations -------------------------------------------------------------------------------------------
# INSERT DATA HERE
rev = (XX
         .rename(columns={"Month": "month", "AEA RPK TO": "revenue"}))

rev["month"] = pd.to_datetime(rev["month"])

plt.figure()
sns.lineplot(data=rev, x="month", y="revenue")
plt.title("AEA Revenue Passenger KM")
plt.show()

rev["dummy"] = (rev["month"] >= pd.to_datetime("2011-09-01")).astype(int)
rev["t"] = range(1, len(rev) + 1)


# Nonlinearity ----------------------------------------------------------------------------------------------------
#INSERT DATA HERE
unemp = XX

unemp["unemp_rate_s"] = pd.to_numeric(unemp["unemp_rate_s"], errors="coerce")
unemp["unemp_rate_ns"] = pd.to_numeric(unemp["unemp_rate_ns"], errors="coerce")

# create recession and expansion dummies
unemp["difference"] = unemp["unemp_rate_s"] - unemp["unemp_rate_s"].shift(1)

unemp["economic_state"] = np.where(unemp["difference"] > 0, 
                                   "recession", "expansion")

unemp["D_recession"] = (unemp["economic_state"] == "recession").astype(int)
unemp["D_expansion"] = (unemp["economic_state"] == "expansion").astype(int)

unemp["lag_4"] = -(unemp["unemp_rate_s"] - unemp["unemp_rate_s"].shift(4))

nonlinear_lm = smf.ols("difference ~ D_recession + D_expansion - 1", data=unemp).fit()
print(nonlinear_lm.summary())
