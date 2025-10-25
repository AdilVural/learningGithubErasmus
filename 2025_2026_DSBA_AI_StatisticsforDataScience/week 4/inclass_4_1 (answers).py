# Statistics for Data Science
# Lecture 4
# In-class assignment 4.1
# Regression exercise

# %% Load packages
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %% Load data
df = pd.read_csv('houseprice.csv')
print(df.head())

# %% Perform OLS estimation
mod = smf.ols(formula="price ~ lotsize", data=df)
res = mod.fit()
print(res.summary())

# %% Show some more results
print("\nParameters")
print(res.params)
print("\nIn-sample predictions")
print(res.fittedvalues)
print("\nOut-of-sample predictions")
print(res.predict(exog={'lotsize': [100,1000,10000]}))
a = 0.05
print(f"\nConfidence intervals alpha={a}")
print(res.conf_int(alpha=a))

# %% create scatter with fitted line
plt.scatter(df.lotsize, df.price)
plt.xlabel("lotsize")
plt.ylabel("price")
sm.graphics.abline_plot(model_results=res, color='red', ax=plt.gca())

# Interpretation
## price increases with about 6.6 for every additional sq foot lotsize

