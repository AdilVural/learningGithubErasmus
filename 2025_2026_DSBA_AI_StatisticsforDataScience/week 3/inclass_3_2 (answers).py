# Statistics for Data Science
# Lecture 3
# In-class assignment 3.2
# Comparing samples exercise

# %% Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# %% Load data
df = pd.read_csv('houseprice.csv')
print(df.head())

# %%
withairco    = df[df.airco == 1]
withoutairco = df[df.airco == 0]

# %% Plot both histograms
withairco.price.hist(label="with airco")
withoutairco.price.hist(label = "without airco", alpha = 0.5)
plt.legend()
plt.xlabel("Price")
plt.show()

# %% Inspect summary statistics
print(f"Without: mean={withoutairco.price.mean()}, var={withoutairco.price.var()}")
print(f"With:    mean={withairco.price.mean()}, var={withairco.price.var()}")

# %% Perform tests
v1 = withairco.price
v2 = withoutairco.price
var_ratio = v1.var()/(v2.var())
p_value = 1 - stats.f.cdf(var_ratio, v1.size - 1, v2.size - 1)
print("p-value of F test", p_value, "var_ratio", var_ratio)

# %% t-test for mean
res = stats.ttest_ind(withairco.price, withoutairco.price, equal_var=False)
print(f"Test for equal mean price for airco vs no airco pvalue= {res.pvalue}")
display(res)

# %% Try with log transformation
np.log(withairco.price).hist(label="with airco")
np.log(withoutairco.price).hist(label = "without airco", alpha = 0.5)
plt.legend()
plt.xlabel("log price")
plt.show()
print("--> distributions become closer to the normal distribution")

# %% Redo tests
v2 = np.log(withairco.price)
v1 = np.log(withoutairco.price)
var_ratio = v1.var()/(v2.var())
p_value = 1 - stats.f.cdf(var_ratio, v1.size - 1, v2.size - 1)
print("p-value of F test for equal variance", p_value, "var_ratio", var_ratio)

# %% Redo tests for mean
res = stats.ttest_ind(v1, v2, equal_var=False)
print(f"Test for equal mean log price for airco vs no airco pvalue= {res.pvalue}")

res = stats.ttest_ind(v1, v2, equal_var=True)
print(f"Test for equal mean log price for airco vs no airco pvalue= {res.pvalue}")

print("=> all give same conclusion")
