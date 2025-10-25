# Statistics for Data Science
# Lecture 4
# In-class assignment 4.3
# Interactions

# %% Load packages
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# %% Load data
statex77 = pd.read_csv('statex77.csv')
print(statex77.head())
# Available variables:
# Population: population estimate as of July 1, 1975
# Income:     per capita income (1974)
# Illiteracy: illiteracy (1970, percent of population)
# Life Exp:   life expectancy in years (1969–71)
# Murder:     murder and non-negligent manslaughter rate per 100,000 population (1976)
# HS Grad:    percent high-school graduates (1970)
# Frost:      mean number of days with minimum temperature below freezing (1931–1960) in capital or large city
# Area:       land area in square miles

# %%
no_interaction = smf.ols(formula="Murder ~ Income + Population", data=statex77)
res = no_interaction.fit()

# %%
print("\n")
print(res.summary())
# Interpretation
# * An increase of 1000 dollars in per capita income decreases the murder rate by 1.9 murders/100_000 (keeping population constant)
# * An increase of 1000 people in population increases the murder rate by 0.33 murders/100_000 (keeping income constant)
# Question: does the impact of income depend on the population? (and vice versa)
# -> study interactions

#%% Add interactions
interaction = smf.ols(formula="Murder ~ Income*Population" , data=statex77)
res_interaction = interaction.fit()

#%% Look at estimation results
print(res_interaction.summary())
print("\nNote: insignificance of Income does NOT mean that income has no effect")

#%% Visualize the "interaction effects"
inc_mean = statex77.Income.mean()
pop_low = statex77.Population.min()
pop_high = statex77.Population.max()

pred = res_interaction.predict({'Income': [inc_mean, inc_mean], 'Population': [pop_low, pop_high]})
plt.plot([pop_low, pop_high], pred, label="at mean income")

pred = res_interaction.predict({'Income': [0.90*inc_mean, 0.90*inc_mean], 'Population': [pop_low, pop_high]})
plt.plot([pop_low, pop_high], pred, label="at 90% of mean income")

pred = res_interaction.predict({'Income': [1.1*inc_mean, 1.1*inc_mean], 'Population': [pop_low, pop_high]})
plt.plot([pop_low, pop_high], pred, label="at 110% of mean income")

plt.xlabel("Population")
plt.ylabel("Murder rate")
plt.legend(loc="upper left")

print("\n=> Positive impact of income becomes weaker with increasing income")
