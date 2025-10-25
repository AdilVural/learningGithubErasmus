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

#%%
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


#%% a
houseprice = pd.read_csv("houseprice.csv")

#%% b
display(houseprice)
print(list(houseprice))
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

# --------------------------------------------
# Exercise 1.2
# --------------------------------------------
#%% a
print(houseprice.price.mean())

#%% b
print(houseprice.price.median()) ## skew to the right (median < mean)

#%% c
print('min ', houseprice.price.min())
print('max ', houseprice.price.max())

#%% d
print('range ', houseprice.price.max() - houseprice.price.min())
print('standard deviation ', houseprice.price.std())

#%% e
print(houseprice.airco.value_counts())
print(houseprice.driveway.value_counts())
print(houseprice.fullbase.value_counts())
print(houseprice.gashw.value_counts())
print(houseprice.recroom.value_counts())
print(houseprice.bedrooms.value_counts())

print(houseprice.bedrooms.value_counts().sort_index())

#%% f
print(houseprice.bedrooms.value_counts().sort_index())
print(sum(houseprice.bedrooms == 6))

# --------------------------------------------
# Exercise 1.3
# --------------------------------------------
#%% a
houseprice.price.plot.hist()
plt.show()
houseprice.price.plot.hist(bins=50)


#%% b
houseprice.stories.plot.hist();
plt.show()
houseprice.stories.value_counts().sort_index().plot.bar();

#%% c
houseprice.lotsize.plot.density()
plt.show()
stats.probplot(houseprice.lotsize, dist="norm", plot=plt);

# --------------------------------------------
# Exercise 1.4
# --------------------------------------------
#%% a
houseprice.plot('lotsize', 'price', kind='scatter')

#%% b
pd.crosstab(houseprice.bedrooms, houseprice.bathrms)


#%% c
display( pd.crosstab(houseprice.bedrooms, houseprice.prefarea) )
display( pd.crosstab(houseprice.bedrooms, houseprice.prefarea, margins=True, normalize='all') )
display( pd.crosstab(houseprice.bedrooms, houseprice.prefarea, margins=True, normalize='index') )
display( pd.crosstab(houseprice.bedrooms, houseprice.prefarea, margins=True, normalize='columns') )


#%% d
houseprice.boxplot('lotsize');
houseprice.boxplot('lotsize', by="stories")


# --------------------------------------------
# Exercise 2
# --------------------------------------------
#%% a
vwgolf = pd.read_csv("vwgolf.csv", parse_dates=['APKExpiry','OwnerSince','DateNLRegistration']).convert_dtypes()

#%% b
display(vwgolf.info())
display(vwgolf.describe())

#%% c
vwgolf.plot('Mileage', 'AskingPrice', kind = 'scatter')
#or
# plt.scatter('Mileage', 'AskingPrice', data=vwgolf)


#%% d
plt.scatter(vwgolf.Mileage, vwgolf.PriceNew - vwgolf.AskingPrice);
plt.gca().set_xlabel('Mileage');
plt.gca().set_ylabel('Price new - Asking price');


#%% e
vwgolf.PriceNew.plot.density()
vwgolf.PriceNew.plot.hist(density=True)

#%% f
vwgolf.boxplot('Mileage', by='Fuel');

#%% g
print('min ', vwgolf.NrOwners.min(), 'median ', vwgolf.NrOwners.median(), 'max ', vwgolf.NrOwners.max())

#%% h
diff = vwgolf.PriceNew - vwgolf.AskingPrice
diff.quantile(q=[0.0, 0.25, 0.5, 0.75, 1.00 ])


#%% i
pd.crosstab(vwgolf.Fuel, vwgolf.TransmissionAuto)

#%% j
print(vwgolf.Mileage.min(), vwgolf.Mileage.max())
print(vwgolf.Mileage.isna().sum())

#%%
