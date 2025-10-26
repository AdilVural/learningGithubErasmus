# Statistics for Data Science
# Lecture 6
# In-class assignment 6.2
# Logit models/GLM

# %% Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from model_selection import *
from ploteffect import *
import sklearn.metrics as skm

# %% Load data
df = pd.read_csv("website.csv")
df["active"] = (df["min"] > 0.0)*1
df["region"] = df["region1"]*1+df["region2"]*2+df["region3"]*3
df["region"] = df["region"].astype("category")
print(df.head())

# %% Part I
# Estimate parameters of a logit model
logit = smf.glm(formula="active ~ income + age + region", data = df, family = sm.families.Binomial(link=sm.families.links.Logit())).fit()
print(logit.summary())
# Note the omitted region1 in model
# Region 2 and 3 are both not significant, but one is negative and the other positive
# The difference between 2 and 3 may be significant!

# %% Plot predicted probabilities
plt.scatter(df.age, logit.predict())

#%% Plot effects
ploteffect(logit, "age")
plt.show()
ploteffect(logit, "income")
plt.show()
ploteffect(logit, "region")
# Plots look linear, but are a part of an S-curve

# %% Extra: Consider transformations of variables (add some transformations)
largelogit = smf.glm(formula="active ~ income + age + region + I(age**2) + np.log(age) + I(income**2) + np.log(income)", data = df, family = sm.families.Binomial(link=sm.families.links.Logit())).fit()
largelogit.summary()

#%% Note not much sign. anymore -> large correlation between variables
# Reduce using AIC
largelogit = smf.glm(formula="active ~ income + age + region +  I(age**2) + np.log(age) + I(income**2) + np.log(income)", data = df, family = sm.families.Binomial(link=sm.families.links.Logit()))
largelogit = backward_elimination_aic(largelogit)
largelogit.summary()

#%% Create new plot of effects
ploteffect(largelogit, "age")
plt.show()
ploteffect(largelogit, "income")
plt.show()
ploteffect(largelogit, "region")
