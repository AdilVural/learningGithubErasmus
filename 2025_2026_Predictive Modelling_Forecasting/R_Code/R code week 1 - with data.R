# R code week 1
# load libraries
library(dplyr)
library(readxl)
library(ggplot2)
library(stringr)

# One way to describe a time series with a trend-------------------------------------------------------------------
# load data
data <- read_excel("LA real GDP per capita.xlsx")

# Remove first year and create t = 1, 2, 3, ..., T
data <- data %>%
  filter(YEAR != 1950) %>%
  mutate(t = 1:n())

ols_argentina <- lm(log(ARGENTINA)*100 ~ t, data)
ols_brazil <- lm(log(BRAZIL)*100 ~ t, data)
ols_chile <- lm(log(CHILE)*100 ~ t, data)
ols_colombia <- lm(log(COLOMBIA)*100 ~ t, data)
ols_mexico <- lm(log(MEXICO)*100 ~ t, data)

summary(ols_argentina)


# The direction of a trend can change -----------------------------------------------------------------------------
stock_motorcycles <- read_excel("NL stock of motorcycles.xlsx") %>%
  select(year = YEAR, stock = MOTORSTOCK)

ggplot(data = stock_motorcycles) +
  geom_line(aes(x = year, y = stock))


# UK consumption data ---------------------------------------------------------------------------------------------
household_consumption <- read_excel("UK household final consumption.xlsx") %>%
  filter(Name != "Datastream Code") %>%
  rename(period = "Name",
         consumption_seasonally = "UK FINAL CONSMPTN.EXPENDITURE: HOUSEHOLD - NATIONAL CONCEPT CON (SEASONALLY ADJUSTED)",
         consumption_not_seasonally = "UK HOUSEHOLD FINAL CONSMPTN. EXPENDITURE - NATIONAL CONCEPT CON (NOT SEASONALLY ADJUSTED)") %>%
  select(period, consumption_seasonally, consumption_not_seasonally)

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

household_consumption <- household_consumption %>%
  filter(t >= 21)
dummy_regression <- lm(dep_var_ns ~ D1 + D2 + D3 + D4 - 1, data = household_consumption)
summary(dummy_regression)


# Annual growth rates ---------------------------------------------------------------------------------------------
household_consumption <- household_consumption %>%
  mutate(annual_growth = 100 * log(consumption_not_seasonally/lag(consumption_not_seasonally, n = 4)))


# Aberrant observations -------------------------------------------------------------------------------------------
revenue_per_passenger <- read_excel("AEA revenue passenger kilometres.xlsx") %>%
  select(month = Month, revenue = "AEA RPK TO") %>%
  mutate(month = lubridate::ymd(month))

ggplot(revenue_per_passenger) +
  geom_line(aes(x = month, y = revenue))

# Create dummy
revenue_per_passenger <- revenue_per_passenger %>%
  mutate(dummy = if_else(month >= lubridate::ymd(20110901), 1, 0),
         t = 1:n())


# Nonlinearity ----------------------------------------------------------------------------------------------------
unemployment_rate <- read_excel("US unemployment rate.xlsx") %>%
  filter(Name != "Datastream Code") %>%
  rename(period = "Name",
         unemp_rate_s = "US UNEMPLOYMENT RATE (SEASONALLY ADJUSTED)",
         unemp_rate_ns = "US CIVILIAN UNEMPLOYMENT RATE (NOT SEASONALLY ADJUSTED)") %>%
  select(period, unemp_rate_s, unemp_rate_ns) %>%
  mutate(unemp_rate_s = as.numeric(unemp_rate_s),
         unemp_rate_ns = as.numeric(unemp_rate_ns))

# create recession and expansion dummies
unemployment_rate <- unemployment_rate %>%
  mutate(economic_state = if_else((unemp_rate_s - lag(unemp_rate_s)) > 0, "recession", "expansion")) %>%
  mutate(D_recession = as.numeric(economic_state == "recession"),
         D_expansion = as.numeric(economic_state == "expansion"),
         difference = unemp_rate_s - lag(unemp_rate_s)) #Note this is not the correct way to determine a recession

unemployment_rate <- unemployment_rate %>%
  mutate(lag_4 = -(unemp_rate_s - lag(unemp_rate_s, 4)))

nonlinearity_lm <- lm(difference ~ D_recession + D_expansion - 1, data = unemployment_rate)
summary(nonlinearity_lm)

