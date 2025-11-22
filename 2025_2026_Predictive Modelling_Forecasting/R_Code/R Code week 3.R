# R code week 3

# !!IMPORTANT!!
# This R code is made to aid you to reproduce the figures and tables from the slides of week 3.
# Note that the script does NOT actually work since the data is missing.

# load libraries
library(dplyr)
library(readxl)
library(forecast)
library(ggplot2)

# AR(1) may not suffice -------------------------------------------------------------------
# load data
##INSERT DATA HERE
gdp_data <- XX

# Remove first year and create t = 1, 2, 3, ..., T
gdp_data <- gdp_data %>%
  filter(YEAR != 1950) %>%
  mutate(t = 1:n())

# Get growth
gdp_data <- gdp_data %>%
  mutate(growth_brazil = (BRAZIL - lag(BRAZIL))/lag(BRAZIL) * 100)

# Create line plot of growth
ggplot(gdp_data) +
  geom_line(aes(x = t, y = growth_brazil)) +
  ylim(-12, 12)

# Autocorrelations, GDP growth Brazil -----------------------------------------------------------------------------
# Calculate the acf for the growth of Brazil's GDP
acf_growth_brazil <- gdp_data %>%
  filter(!is.na(growth_brazil)) %>%
  pull(growth_brazil) %>%
  acf()

# Calculate the 2SE
two_SE <- 2 * 1/sqrt(nrow(filter(gdp_data, !is.na(growth_brazil))))

# Combine into data
acf_data <- tibble(acf = acf_growth_brazil$acf) %>%
  mutate(two_SE_plus = two_SE, two_SE_minus = two_SE * -1, i = 0:(n() - 1)) %>%
  filter(i != 0)

# Create plot
ggplot(acf_data) +
  geom_line(aes(x = i, y = acf), col = "blue") +
  geom_line(aes(x = i, y = two_SE_plus), col = "green") +
  geom_line(aes(x = i, y = two_SE_minus), col = "red")

# Partial autocorrelation -----------------------------------------------------------------------------------------
pacf_growth_brazil <- gdp_data %>%
  filter(!is.na(growth_brazil)) %>%
  pull(growth_brazil) %>%
  pacf()

# Combine into tibble
pacf_data <- tibble(pacf = pacf_growth_brazil$acf) %>%
  mutate(two_SE_plus = two_SE, two_SE_minus = two_SE * -1, i = 1:n())

# Create plot
ggplot(pacf_data) +
  geom_line(aes(x = i, y = pacf), col = "blue") +
  geom_line(aes(x = i, y = two_SE_plus), col = "green") +
  geom_line(aes(x = i, y = two_SE_minus), col = "red")


# Estimation results AR(2) -----------------------------------------------------------------------------------------
ar_2_data <- gdp_data %>%
  mutate(lag_1 = lag(growth_brazil),
         lag_2 = lag(growth_brazil, 2))
ar_2_ols <- lm(growth_brazil ~ lag_1 + lag_2, data = ar_2_data)

# Or using Arima instead of OLS
ar_2 <- Arima(gdp_data$growth_brazil, c(2, 0, 0))

# Autocorrelations monthly growth airline revenues -------------------------------------------------------------------------------------------
##INSERT DATA HERE
revenue_per_passenger <- XX

revenue_per_passenger <- revenue_per_passenger %>%
  select(month = Month, revenue = "AEA RPK TO") %>%
  mutate(month = lubridate::ymd(month))

# Get monthly growth
revenue_per_passenger <- revenue_per_passenger %>%
  mutate(growth = (revenue - lag(revenue))/(lag(revenue))*100,
         log_growth = 100 * (log(revenue/lag(revenue))))

# Get acf
acf_revenue <- revenue_per_passenger %>%
  filter(!is.na(growth)) %>%
  pull(growth) %>%
  acf()

# Annual growth ---------------------------------------------------------------------------------------------------
revenue_per_passenger <- revenue_per_passenger %>%
  mutate(annual_growth = (revenue - lag(revenue, 12))/(lag(revenue, 12))*100,
         log_annual_growth = 100 * (log(revenue/lag(revenue, 12))))

# Get acf
acf_annual_growth <- revenue_per_passenger %>%
  filter(!is.na(annual_growth)) %>%
  pull(annual_growth) %>%
  acf()

ggplot(revenue_per_passenger) +
  geom_line(aes(x = month, y = revenue))

# Autocorrelations of double differenced airline revenues ----------------------------------------------------------
# get double differences
revenue_per_passenger <- revenue_per_passenger %>%
  mutate(first_difference = log(revenue) - lag(log(revenue)),
         second_difference = first_difference - lag(first_difference,12))

acf_double_diff <- revenue_per_passenger %>%
  filter(!is.na(second_difference)) %>%
  pull(second_difference) %>%
  acf()
