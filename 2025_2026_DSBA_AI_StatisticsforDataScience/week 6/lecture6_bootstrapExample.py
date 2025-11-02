# Statistics for Data Science
# Lecture 6
# Bootstrap example

#%% Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

#%% Load houseprice
df = pd.read_csv("houseprice.csv")

#%% Run a regressio
fit = smf.ols(formula="price~lotsize", data=df).fit()
print(fit.summary())
print(fit.params)
print(fit.conf_int())
fit.params["lotsize"]

#%% Run bootstrap for conf interval lotsize parameter
collectstats = []
for i in range(1_000):
    bootstrapsample = df.sample(frac=1.0, replace=True)
    bffit = smf.ols(formula="price~lotsize", data=bootstrapsample).fit()
    collectstats.append(bffit.params["lotsize"])


#%% Show results
plt.hist(collectstats, bins=50, density=True)
plt.xlabel("Impact of lotsize")
plt.ylabel("(Estimated) density of estimator")
plt.show()
print("95% confidence interval:", np.quantile(collectstats, [0.025, 0.975]))
print("bootstrap mean", np.mean(collectstats), ", original estimate", fit.params["lotsize"])
print("Regular confidence interval\n", fit.conf_int())
print("-> the bootstrap interval is a bit larger!")

#%%