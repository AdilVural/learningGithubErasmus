# Statistics for Data Science
# Lecture 3
# In-class assignment 3.1
# Simulation exercise with t-test

# %% Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# %% define constants
truemean = 0.05
var = 1
n = 100

# %% Generate the data
# data = np.random.normal(loc=truemean, scale=np.sqrt(var), size=n)
data = stats.norm(loc=truemean, scale=np.sqrt(var)).rvs(size=n)
print("Statistics of generated data:")
pd.DataFrame(data).describe()


# %% perform t-test
t_test_result = stats.ttest_1samp(data, popmean=0)
print(f"pvalue of t-test for H_0 of mean=0: {t_test_result.pvalue} (using scipy.stats)")


# %% or calculate p-value yourself
pval = 2 * stats.t(n - 1).cdf(-abs(np.mean(data) / np.sqrt(np.var(data, ddof=1) / n)))
print(f"pvalue of t-test for H_0 of mean=0: {pval} (manual version)")

# %% Calculate power for this exact setting
tp = sm.stats.TTestPower()
pw = tp.power(truemean/np.sqrt(var), nobs=n, alpha=0.05)
print(f"Power of test {pw}")

# %% Calculate n needed for power = 50%
needed_n = tp.solve_power(effect_size=truemean/np.sqrt(var), power=0.5, alpha=0.05)
print(f"Number of observations needed for power=50%: {needed_n}")

# %% Create power plots
tp.plot_power(dep_var='nobs', nobs=np.arange(1,500)*10, effect_size=[0, 0.05, 0.1, 0.25, 0.5]/np.sqrt(var))
plt.show()
# Note that the effect size is 'standardized' (true mean - hypothesized mean)/standard deviation
