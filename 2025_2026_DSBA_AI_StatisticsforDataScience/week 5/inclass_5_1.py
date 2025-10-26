# Statistics for Data Science
# Lecture 5
# In-class assignment 5.1
# Model diagnostics

# %% Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from olsdiagnostics import *   # first download the corresponding file from Canvas


# %% Load data and create observation names
statex77 = pd.read_csv('statex77.csv')
statex77.index = np.array(statex77["Unnamed: 0"])
statex77 = statex77.drop(columns=["Unnamed: 0"])
print(statex77.head())

#%% Use a QQ plot to inspect whether Murder is normally distributed 
# What is your conclusion
# and does this matter for a linear model explaining Murder?
TO DO: use sm.qqplot

#%% Create a linear model explaining Murder using Population, Income, Frost and Illiteracy
m = smf.ols(formula="Murder ~ Population + Income + Frost + Illiteracy", data=statex77)
m.summary()

#%% Show the basic diagnostics
# What do you conclude?
influence = OLSInfluence(m)
TO DO: apply functions from olsdiagnostics to influence object

#%% Continue with the model and consider
# Component-residual plot
TO DO: use sm.graphics.plot_ccpr_grid or sm.graphics.plot_ccpr

#%% RESET test
TO DO: use sm.stats.diagnostic.linear_reset(m, power=2)

#%% Normality of error term
TO DO: use qqresid from olsdiagnostics

### Part II

#%% Investigate independence
TO DO: use sm.stats.stattools.durbin_watson

#%% Test H0 of Homoskedasticity
TO DO: use sm.stats.diagnostic.het_breuschpagan

#%% Calculate heteroskedasticity consistent standard errors
TO DO: use m.get_robustcov_results(cov_type="HC3")

#%% Investigate presence of Multicollinearity
TO DO: use variance_inflation_factor from statsmodels.stats.outliers_influence

### Part III

#%% Test for outliers
TO DO: use m.outlier_test()

#%% Calculate Cook's distance
cd = influence.cooks_distance[0]
print(cd)

#%% Select those that are large (6 is the number of parameters in the model, here this includes the variance)
sel = cd > (4/(influence.nobs-influence.k_vars-1))
print(cd[sel])

#%% Summarize influence measures
TO DO: use influence.plot_influence

# Note influence plot shows:
# * Hat-values (leverage) horizontal vs studentized residuals vertical
# * Reference lines for studentized resid at −2 and +2
# * Reference lines for leverage at 2k/n and 3k/n
# * Size of bubble corresponds to Cook’s distance