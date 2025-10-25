# Statistics for Data Science
# Lecture 2
# In-class assignment 2.1
# Distributions, calculating probabilities and quantiles

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, expon  # this allows us to simply write norm instead of scipy.stats.norm

#%% Some examples to get started:
# Probability that a normally distributed variable (with mean 0 and standard deviation 1) is smaller or equal than 0
print(norm.cdf(0))

#%% Probability that a normally distributed variable (with mean 1 and standard deviation 2) is smaller or equal than 0
print(norm.cdf(0, loc=1, scale=2)) # or
print(norm(loc=1, scale=2).cdf(0))

#%% Probability that to get exactly 1 times head in 2 coin tosses (pmf = probability mass function => only for discrete distributions)
print(binom.pmf(1, n=2, p=0.5))
# see also
help(binom.pmf)
# for help on this function 

#%% Throw a fair coin 10 times and count the number of heads (repeat this function to see that you really get a random result)
print(binom.rvs(n=10, p=0.5, size=10))
# or
print(binom(n=10, p=0.5).rvs(10))

#%% repeat this 1000 times and calculate the average over these 1000 simulations
print(np.mean(binom(n=10, p=0.5).rvs(size = 1000)))

#%% Calculate the 0.5 quantile of the exponential distribution with rate 2 (ppf = percent point function)
print(expon.ppf(0.5, scale=1/2))

#######################
#  The assignments:   #
#######################

#%% Calculate the following probabilities
# Standard normally distributed variable is *larger* than 1.


#%% Normally distributed variable with mean 20 and variance 10 is smaller than 15.


#%% Getting (exactly) 15 times head in 30 coin tosses.


#%% Suppose that a soccer club has a 60% probability of winning each match they play.
# What is the probability that they do not win any of the first four matches of the year?


#%% Calculate a quantile
# Suppose that the waiting time for the bus has an exponential distribution with scale 10. 
# How many minutes does one have to wait at least on the 5% worst days?


