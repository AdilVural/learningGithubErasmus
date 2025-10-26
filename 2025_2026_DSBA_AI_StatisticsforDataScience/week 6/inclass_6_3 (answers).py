# Statistics for Data Science
# Lecture 6
# In-class/take home assignment 6.3
# Bootstrapping the difference between predicted probabilities

#%% Import packags
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

#%% Load website data
df = pd.read_csv("website.csv")
df["active"] = (df["min"] > 0.0)*1
df["region"] = df["region1"]*1+df["region2"]*2+df["region3"]*3
df["region"] = df["region"].astype("category")

#%% An experiment step to see how to obtain the statistic of interest
l = smf.glm(formula = "active ~ region + age+ income", family = sm.families.Binomial(link=sm.families.links.Logit()), data=df).fit()
predlow  = l.predict(exog={'age': 40, 'income': 2000, 'region' : 1})
predhigh = l.predict(exog={'age': 40, 'income': 3000, 'region' : 1})
(predhigh - predlow)[0]

#%% Implement bootstrap
collectstats = []
for i in range(1000):
    bootstrapsample = df.sample(frac=1.0, replace=True)
    
    l = smf.glm(formula = "active ~ region + age+ income", family = sm.families.Binomial(link=sm.families.links.Logit()), data=bootstrapsample).fit()
    predlow  = l.predict(exog={'age': 40, 'income': 2000, 'region' : 1})
    predhigh = l.predict(exog={'age': 40, 'income': 3000, 'region' : 1})
    stat = (predhigh - predlow)[0]
    
    collectstats.append(stat)

print("done")
    
#%% Report results
plt.hist(collectstats, bins=50)
print("95% confidence interval:", np.quantile(collectstats, [0.025, 0.975]))
print("bootstrap mean", np.mean(collectstats))

#%%