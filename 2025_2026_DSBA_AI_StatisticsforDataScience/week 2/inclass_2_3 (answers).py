# Statistics for Data Science
# Lecture 2
# In-class assignment 2.3
# Do-it-yourself confidence intervals

# %% Load package
import numpy as np
from scipy import stats
import pandas as pd

# %% Useful functions given some variable x
x = stats.norm().rvs(100)

# %%
# mean
print( x.mean() )
# variance
print( x.var(ddof=1) )
# standard deviation 
print( x.std(ddof=1))
# number of observations (size of x)
print( x.size )
# or 
print(f"mean: {x.mean()}, variance: {x.var(ddof=1)}, std.dev: {x.std(ddof=1)}, size: {x.size}")

#######################
#  The assignments:   #
#######################
# %% Load data
df = pd.read_csv('houseprice.csv')
print(df.head())
df.describe()

# %% Calculate means
x = df.lotsize
res = x.describe()
display(res)

# %%
print(res['mean'], x.mean())
print(res['std']**2, x.var(), x.var(ddof=1))
print("Standard code does not do degrees of freedom correction!")

# %% Calculate standard error of mean
stderr    = np.sqrt(x.var(ddof=1)/x.size)
altstderr = x.std(ddof=1)/np.sqrt(x.size)

print(stderr, altstderr)

# %% Calculate confidence interval around mean 
alpha = 0.05
interval = x.mean() + stats.t.ppf((alpha/2, 1-alpha/2), x.size - 1)*stderr
print(interval)

# Confirm results
# You should obtain the interval [4967.998, 5332.533]
