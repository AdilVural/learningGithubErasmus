# Statistics for Data Science
# Lecture 4
# In-class assignment 4.2
# Regression exercise with non-linear models

# %% Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# %% Load data
df = pd.read_csv('houseprice.csv')
print(df.head())

# %% Perform OLS estimation
mod = smf.ols(formula='price ~ lotsize', data=df)
res = mod.fit()
print("\n")
print(res.summary())

# %% Obtain R2
print("\nR2", res.rsquared)
print("-> R2 is not very high, but variable is quite important")

# %% H0: lotsize no impact (see t-value)
print(res.summary())

# %% Conclusion will match with (actually behind the scenes this is an equivalent method)
cortest = stats.pearsonr(df.lotsize, df.price)
print("p value of correlation test", cortest.pvalue)

# %% Consider alternative models
mod2 = smf.ols(formula='price ~ np.log(lotsize)', data=df)
res2 = mod2.fit()
print(res2.summary())
print("-> R2 is better!")

# %% log-log model
mod3 = smf.ols(formula='np.log(price)~np.log(lotsize)', data=df)
res3 = mod3.fit()
print(res3.summary())
print("-> Do not compare this R2 to the earlier ones (this one relates to log(price), not price itself!)")
print("-> If lotsize increases with 1%, the price increases by about 0.5%")

# %% Create plot with all fitted lines
plt.scatter(df.lotsize, df.price)
rng = np.arange(df.lotsize.min(), df.lotsize.max())
plt.plot(rng, res.predict(exog={"lotsize" : rng}), label="linear", color='r')
plt.plot(rng, res2.predict(exog={"lotsize" : rng}), label="price~log(size)", color='g')
plt.plot(rng, np.exp(res3.predict(exog={"lotsize" : rng})), label="log(price)~log(size) -- naive", color='m')
sigma2 = res3.scale
plt.plot(rng, np.exp(res3.predict(exog={"lotsize" : rng}) + 0.5*sigma2), label="log(price)~log(size) -- corrected", color='m', linestyle=":")
plt.legend()
