# R code week 1

# !!IMPORTANT!!
# This R code is made to aid you to reproduce the figures and tables from the slides of week 1.
# Note that the script does NOT actually work since the data is missing.
# load libraries
library(dplyr)
library(readxl)
library(ggplot2)
library(stringr)

# One way to describe a time series with a trend-------------------------------------------------------------------
# load data
##INSERT DATA HERE
data <- XX

# Remove first year and create t = 1, 2, 3, ..., T
data <- data %>%
  filter(year != 1950) %>%
  mutate(t = 1:n())

# Do OLS regression on the log(y) * 100
ols_argentina <- lm(log(argentina)*100 ~ t, data)

summary(ols_argentina)


# The direction of a trend can change -----------------------------------------------------------------------------
##INSERT DATA HERE
stock_motorcycles <- XX

# Create plot
ggplot(data = stock_motorcycles) +
  geom_line(aes(x = year, y = stock))


# UK consumption data ---------------------------------------------------------------------------------------------
##INSERT DATA HERE
household_consumption <- XX

# create quarter dummy
household_consumption <- household_consumption %>%
  mutate(quarter = str_extract(period, "Q."),
         t = 1:n()) %>%
  mutate(D1 = if_else(quarter == "Q1", 1, 0),
         D2 = if_else(quarter == "Q2", 1, 0),
         D3 = if_else(quarter == "Q3", 1, 0),
         D4 = if_else(quarter == "Q4", 1, 0),
         consumption_not_seasonally = as.numeric(consumption_not_seasonally),
         dep_var_ns = 400 * log(consumption_not_seasonally / lag(consumption_not_seasonally)))

# Filter to select the correct start date
household_consumption <- household_consumption %>%
  filter(t >= NN) ##INSERT NUMBER HERE to select the correct start date

# Do OLS regression
dummy_regression <- lm(dep_var_ns ~ D1 + D2 + D3 + D4 - 1, data = household_consumption)
summary(dummy_regression)


# Annual growth rates ---------------------------------------------------------------------------------------------
# Create variable for annual growth
household_consumption <- household_consumption %>%
  mutate(annual_growth = 100 * log(consumption_not_seasonally/lag(consumption_not_seasonally, n = 4)))


# Aberrant observations -------------------------------------------------------------------------------------------
##INSERT DATA HERE
revenue_per_passenger <- XX %>%
  mutate(month = lubridate::ymd(month))

# Create plot
ggplot(revenue_per_passenger) +
  geom_line(aes(x = month, y = revenue))

# Create dummy for period after 9/11
revenue_per_passenger <- revenue_per_passenger %>%
  mutate(dummy = if_else(month >= lubridate::ymd(20110901), 1, 0),
         t = 1:n())


# Nonlinearity ----------------------------------------------------------------------------------------------------
##INSERT DATA HERE
unemployment_rate <- XX

# Do OLS regression using the dummies for regression and expansion
nonlinearity_lm <- lm(difference ~ D_recession + D_expansion - 1, data = unemployment_rate)
summary(nonlinearity_lm)

