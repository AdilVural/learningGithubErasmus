# --------------------------------------------
# Day 1 Closing Assignment Solutions
# --------------------------------------------
# Authors of R version of the assignment: 
#          Andreas Alfons and Pieter Schoonees
#          Erasmus University Rotterdam
# --------------------------------------------
# Author of Python adaptation: 
#          Dennis Fok 
#          Erasmus University Rotterdam
# --------------------------------------------
# Lecturer:  Dennis Fok 
#            Erasmus University Rotterdam
# --------------------------------------------

# --------------------------------------------
# Exercise 1.1
# --------------------------------------------

#%% Load packages
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


#%% a
houseprice = pd.read_csv("houseprice.csv")

#%% b
display(houseprice)
display(list(houseprice))
houseprice.info()

#%% c
print(houseprice.shape)
print(houseprice.shape[0])
print(houseprice.shape[1])

#%% d
houseprice.plot();
pd.plotting.scatter_matrix(houseprice);
plt.show()
pd.plotting.scatter_matrix(houseprice[['price','lotsize','garagepl']]);
plt.show()
houseprice.describe()
