# Statistics for Data Science
# Lecture 2
# In-class assignment 2.2
# Empirical distributions and fitting distributions

# If needed install packages first

# %% Load package
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats     # so we don't have to type scipy. in front of stats
import pandas as pd

#%% Some examples to get started
## Create some random data 
d = np.exp(-4 + stats.norm.rvs(size=1000) )

#%% Inspect this data 
# Note that there is no need to create a data frame, we can use the matplotlib.pyplot functions directly
plt.plot(d)
plt.title("Not so useful graph for this data")
plt.show()

plt.hist(d)
plt.ylabel("Frequency (=counts)")
plt.show()

plt.hist(d, density=True, bins=100)
plt.ylabel("density scale")
plt.show()

#%% Fit normal distribution
# We need to define the range of values where we want to search for the best fit
bounds = {'loc': (-4,4), 'scale': (0.0001,1)}
# Fit the normal distribution to data d within these bounds
res_norm = stats.fit(stats.norm, d, bounds)
# Print and plot the results
print(res_norm.params)
res_norm.plot()
plt.show()
print("=> Not so good fit")

#%% Fit lognormal distribution
bounds = {'scale': (0,1), 's':(0.0001, 1)}
res_lognorm = stats.fit(stats.lognorm, d, bounds)
print(res_lognorm.params)
res_lognorm.plot()
plt.show()
print("=> this is a much better fit")

#%% Fit exponential distribution
bounds = {'scale': (0.0001,1)}
res_exp = stats.fit(stats.expon, d, bounds)
print(res_exp)
res_exp.plot()
plt.show()
print("=> not too bad, but not a really good fit")

#%% Compare using negative log likelihood values (low is good)
print(res_norm.nllf(), res_lognorm.nllf(), res_exp.nllf())
print("=> Second (Log normal) is best (=lowest)")

#%% Compare using plot_type{“hist”, “qq”, “pp”, “cdf”}
res_lognorm.plot(plot_type="qq")
plt.show()
res_norm.plot(plot_type="qq")
plt.show()
res_exp.plot(plot_type="qq")
plt.show()
print("=> also here log normal is best")

#%% Compare using plot_type{“hist”, “qq”, “pp”, “cdf”}
res_lognorm.plot(plot_type="pp")
plt.show()
res_norm.plot(plot_type="pp")
plt.show()
res_exp.plot(plot_type="pp")
plt.show()

#%% Compare using plot_type{“hist”, “qq”, “pp”, “cdf”}
res_lognorm.plot(plot_type="cdf")
plt.show()
res_norm.plot(plot_type="cdf")
plt.show()
res_exp.plot(plot_type="cdf")
plt.show()


#######################
#  The assignments:   #
#######################

#%% Load the houseprice data
df = pd.read_csv('houseprice.csv')
df.head()

# %% Use hist() to show the empirical density of lotsize 
df.lotsize.hist(density=True, label="lotsize")
plt.legend()
plt.show()

#%% Fit a normal distribution to the lotsize and graphically inspect the fit
bounds = {"loc": (0,df.lotsize.max()), "scale": (0,10000)}
res_norm = stats.fit(stats.norm, df.lotsize, bounds)
print(res_norm.params)
res_norm.plot()
plt.show()

#%% Fit a log-normal distribution to the lotsize and graphically inspect the fit
bounds = {'scale': (0,df.lotsize.max()), 's':(0, 1)}
res_lognorm = stats.fit(stats.lognorm, df.lotsize, bounds)
print(res_lognorm.params)
res_lognorm.plot()
plt.show()

#%% Which one fits better?
print(res_norm.nllf(), res_lognorm.nllf())
res_lognorm.plot(plot_type="qq")
plt.show()
res_norm.plot(plot_type="qq")
plt.show()
print("=> the log normal distribution has a better fit")