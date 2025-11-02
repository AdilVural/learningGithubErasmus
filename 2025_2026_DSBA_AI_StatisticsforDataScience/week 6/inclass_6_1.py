# Statistics for Data Science
# Lecture 6
# In-class assignment 6.1
# Dummy variable regression

# %% Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ploteffect import *  # note: download first from canvas!

# %% Load data
df = pd.read_csv('houseprice.csv')
print(df.head())

#%% Transform variable to categorical
df["catbed"] = ""
df.loc[df.bedrooms<=2, "catbed"] = "small"
df.loc[df.bedrooms==3, "catbed"] = "medium"
df.loc[df.bedrooms>=4, "catbed"] = "large"
df["catbed"] = df["catbed"].astype("category")

#%% Test whether the price is different across the categories
# Regress price on the factor
m = smf.ols(TODO).fit()

# %% Interpret coefficients in regression model with factor
m.summary()

# %% Change the baseline in the regression model
m2 = smf.ols(TODO).fit()
print(m2.summary())

#%%