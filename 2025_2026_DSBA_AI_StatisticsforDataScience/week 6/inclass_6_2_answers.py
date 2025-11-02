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

#################
#### Part II ####
#################
#%% Try probit model using same variables
probit = smf.glm(formula=logit.model.formula, data = df, family = sm.families.Binomial(link=sm.families.links.Probit())).fit()
probit.summary()

#%% Compare logit with probit in terms of deviance. Which one is better?
print([logit.deviance, probit.deviance])
print("no real differece, but probit is slightly better (cannot use a formal test: no p-value!")

#%% Investigate predictive performance
# Generate confusion matrix
cm = skm.confusion_matrix(df.active, logit.predict() > 0.5, normalize='all') 
print(cm)
skm.ConfusionMatrixDisplay(cm).plot() ;
print("""
Predictive quality is not very good (does this mean the model is bad?)
-> No! This may just be difficult to predict (is not uncommon with human behavior)
""")

#%% plot differences in probs
plt.scatter(df.age, probit.predict()-logit.predict())
plt.xlabel("Age")
plt.ylabel("Probit - Logit prediction")
plt.show()

#%% Coefficients are quite different! (but ratio is almost constant)
tmp = pd.DataFrame(logit.params,columns=["logit"])
tmp["Probit"] = probit.params
tmp["Ratio"] = probit.params/logit.params
display(tmp)

#%%###############
#### PART III ####
##################
# Average Marginal Effects 
margeff = logit.get_margeff()
print(margeff.summary())

#%%