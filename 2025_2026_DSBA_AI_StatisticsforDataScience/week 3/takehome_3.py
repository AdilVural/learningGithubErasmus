# Statistics for Data Science
# Lecture 3
# Take home assignment 
# Correlations

#%% 
import pandas as pd
from scipy import stats

#%% Load data
df = pd.read_csv('houseprice.csv')
print(df.head())

#%% Visualize the correlation between some (continuous) variables 
df.plot('lotsize','price', kind="scatter")

#%% Calculate the correlation
print("Correlation:", stats.pearsonr(df.lotsize, df.price).statistic)

#%% Perform a hypothesis test on this correlation 
# (clearly formulate the hypotheses and the conclusion)
print("H0: Correlation=0, pvalue:", stats.pearsonr(df.lotsize, df.price).pvalue)
# H0: correlation equals 0, Ha: correlation ≠ 0
# Conclusion reject H0, there is a significant correlation
