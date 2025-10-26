# Statistics for Data Science
# Lecture 7
# Additional example on brand choice

# %% Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from model_selection import *
from ploteffect import *
import sklearn.metrics as skm
from scipy import stats

#%% Load data
mydata = pd.read_csv("brandchoice.csv")
print("""
Data contains data on brand choices (Heinz vs Hunts ketchup)
Explanatory variables are price (of both) and two types of promotion indicators (display and feature)
""")
mydata.describe()

#%% Create logit to explain choice of HEINZ using log(price) difference
logit = smf.glm(formula = "HEINZ ~ I(np.log(PRICEHEINZ)-np.log(PRICEHUNTS)) + DISPLHEINZ + DISPLHUNTS + FEATHEINZ + FEATHUNTS", data = mydata, family=sm.families.Binomial(link=sm.families.links.Logit())).fit()
logit.summary()

#%% Consider price difference instead (without log)
altlogit = smf.glm(formula="HEINZ ~ I(PRICEHEINZ-PRICEHUNTS) + DISPLHEINZ + DISPLHUNTS + FEATHEINZ + FEATHUNTS", data = mydata, family = sm.families.Binomial(link=sm.families.links.Logit())).fit()
altlogit.summary()
[logit.deviance, altlogit.deviance]
# model with log price is preferred

#%% Consider model with log(prices) separately
alt2logit = smf.glm(formula="HEINZ ~ np.log(PRICEHEINZ) + np.log(PRICEHUNTS) + DISPLHEINZ + DISPLHUNTS + FEATHEINZ + FEATHUNTS", data = mydata, family = sm.families.Binomial(link=sm.families.links.Logit())).fit()
alt2logit.summary()
print("logit is nested in alt2logit -> we can formally test this (one restriction)")
print("pvalue:", 1- stats.chi2(1).cdf(logit.deviance-alt2logit.deviance))
# model with difference of log prices is not rejected at 5% (although it is a close call)
# we continue with the model with price differences

#%% Can we do the same for display?
diffdispl_logit = smf.glm(formula="HEINZ ~ I(np.log(PRICEHEINZ)-np.log(PRICEHUNTS)) + I(DISPLHEINZ-DISPLHUNTS) + FEATHEINZ+FEATHUNTS", data = mydata, family = sm.families.Binomial(link=sm.families.links.Logit())).fit()
diffdispl_logit.summary()
print("pvalue:", 1- stats.chi2(1).cdf(diffdispl_logit.deviance - logit.deviance))
# yes!

#%% and feature?
alldiff_logit = smf.glm(formula = "HEINZ ~ I(np.log(PRICEHEINZ)-np.log(PRICEHUNTS)) + I(DISPLHEINZ-DISPLHUNTS) + I(FEATHEINZ-FEATHUNTS)", data = mydata, family = sm.families.Binomial(link=sm.families.links.Logit())).fit()
alldiff_logit.summary()
print("pvalue:", 1- stats.chi2(1).cdf(alldiff_logit.deviance - diffdispl_logit.deviance))
# yes, as well!

#%% Final model
alldiff_logit.summary()
# all factors are significant

#%% Predictive performance
cm = skm.confusion_matrix(mydata.HEINZ, alldiff_logit.predict() > 0.5) 
print(cm)
skm.ConfusionMatrixDisplay(cm).plot() ;
print("""
Predictive quality is quite good
Note that there are only few HUNTS (y=0) purchases in the data
""")

#%%%
