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
from model_selection import * # see updated file on canvas
from ploteffect import *  # see file on canvas
import sklearn.metrics as skm

# %% Load data
df = pd.read_csv("website.csv")
df["active"] = (df["min"] > 0.0)*1
df["region"] = df["region1"]*1+df["region2"]*2+df["region3"]*3
df["region"] = df["region"].astype("category")
print(df.head())

# %% Part I
# Estimate parameters of a logit model
logit = smf.glm(TODO).fit()
print(logit.summary())

# %% Plot predicted probabilities
plt.scatter(TODO)

#%% Plot effects
ploteffect(TODO)

# %% Extra: Consider transformations of variables (add some transformations)


#################
#### Part II ####
#################
#%% Try probit model using same variables
probit = smf.glm(TODO).fit()
probit.summary()

#%% Compare logit with probit in terms of deviance. Which one is better?
TODO

#%% Investigate predictive performance
# Generate confusion matrix
cm = skm.confusion_matrix(TODO) 
print(cm)
skm.ConfusionMatrixDisplay(cm).plot() ;

#%% Optional: plot differences in probs

#%%###############
#### PART III ####
##################
# Average Marginal Effects 
TODO

#%%