# Statistics for Data Science
# Lecture 6
# In-class assignment 6.1
# Dummy variable regression

# %% Load packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ploteffect import *

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
m = smf.ols(formula="price~catbed", data=df).fit()
print(m.summary())

print("""
F-statistic shows that the two indicators are important. 
We reject the model with only a constant
""")

# %% Regress price on only a constant
print("We can also test this 'manually':")
onlyconstant = smf.ols(formula="price~1", data=df).fit()
print(onlyconstant.summary())
print(sm.stats.anova_lm(onlyconstant, m))
print("-> Same conclusion: significant differences")

# %% Interpret coefficients in regression model with factor
print(m.summary())
print("""
Houses with a medium number of bedrooms are 10,9k dollar cheaper than 
houses with a large number of bedrooms. Houses with a small number of bedrooms
are 30.03k dollar cheaper than houses with a large number of bedrooms
""")

# %% Change the baseline in the regression model
m2 = smf.ols(formula="price~C(catbed,Treatment('medium'))", data=df).fit()
print(m2.summary())
# Now the results indicate that large is 10,906 more expensive compared to
# medium (same result of course, but different numbers). Also the other coef
# now gives small vs medium. F-statistic is unchanged


# %% Extra: try this after controlling for lotsize
m3 = smf.ols(formula="price~lotsize+catbed", data=df).fit()
print(m3.summary())
print("\nNow we cannot use the F-statistic (as this also includes the impact of lotsize)\nConsider model without the categorical variable.")
small3 = smf.ols(formula="price~lotsize", data=df).fit()
print(small3.summary())

print("Test the nested models")
print(sm.stats.anova_lm(small3, m3))

print("""
-> Still significant effect of catbed, but coefficients now indicate effect keeping
lotsize constant!
""")

#%%