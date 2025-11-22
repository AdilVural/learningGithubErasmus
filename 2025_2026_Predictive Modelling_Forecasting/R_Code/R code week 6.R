# R code week 6

# !!IMPORTANT!!
# This R code is made to aid you to reproduce the figures and tables from the slides of week 6.
# Note that the script does NOT actually work since the data is missing.

# load libraries
library(dplyr)
library(readxl)
library(tidyr)
library(ggplot2)
library(stringr)

# Modelling seasonality -------------------------------------------------------------------------------------------
##INSERT DATA HERE
industrial_production <- XX

industrial_production <- industrial_production %>%
  select(date = `Datastream Code`,
         usiptot_g = USIPTOT.G,
         usiptot_h = USIPTOT.H) %>%
  mutate(quarter = date) %>%
  mutate(date = case_when(
    str_detect(date, "Q1") ~ lubridate::ymd(str_c(str_extract(date, "[[:digit:]]{4}$"), "-01-01")),
    str_detect(date, "Q2") ~ lubridate::ymd(str_c(str_extract(date, "[[:digit:]]{4}$"), "-04-01")),
    str_detect(date, "Q3") ~ lubridate::ymd(str_c(str_extract(date, "[[:digit:]]{4}$"), "-07-01")),
    str_detect(date, "Q4") ~ lubridate::ymd(str_c(str_extract(date, "[[:digit:]]{4}$"), "-10-01")),
    TRUE ~ as.Date(NA)),
    quarter = str_extract(quarter, "Q+.")
  ) %>%
  mutate(log_usip = log(usiptot_h))

# Modeling seasonality
ggplot(industrial_production) +
  geom_line(aes(date, log_usip))

# Observations per quarter
ggplot(industrial_production) +
  geom_line(aes(date, log_usip, color = quarter))

# Growth rate y_t - y_(t-1)
industrial_production <- industrial_production %>%
  mutate(quarterly_growth = log_usip - lag(log_usip))

ggplot(industrial_production) +
  geom_line(aes(date, quarterly_growth, color = quarter))


# Deterministic seasonality ---------------------------------------------------------------------------------------
# Add dummies for the seasons
industrial_production <- industrial_production %>%
  mutate(D1 = as.numeric(quarter == "Q1"),
         D2 = as.numeric(quarter == "Q2"),
         D3 = as.numeric(quarter == "Q3"),
         D4 = as.numeric(quarter == "Q4"),
  )

lm(quarterly_growth ~ -1 + D1 + D2 + D3 + D4 + lag(quarterly_growth) + lag(quarterly_growth, 2) +
     lag(quarterly_growth, 3) + lag(quarterly_growth, 4) + lag(quarterly_growth, 5), industrial_production)


# Stochastic seasonality ------------------------------------------------------------------------------------------
# Growth rate y_t - y_(t-4)
industrial_production <- industrial_production %>%
  mutate(annual_growth = log_usip - lag(log_usip, 4))

ggplot(industrial_production) +
  geom_line(aes(date, annual_growth))

# Model for annual growth rates
# Use seasonal to only add the fourth MA term and not the first, second and third.
arima_annual_growth <- arima(industrial_production$annual_growth, order = c(2, 0, 0),
                             seasonal = list(order = c(0, 0, 1), period = 4))

# Double differencing
industrial_production <- industrial_production %>%
  mutate(double_dif = quarterly_growth - lag(quarterly_growth, 4))

arima_double_difference <- arima(industrial_production$annual_growth, order = c(0, 0, 1),
                                 seasonal = list(order = c(0, 0, 1), period = 4))

# USA IP - selection between differencing filters
usa_ip_dif_data <- industrial_production %>%
  mutate(lag_annual_growth = lag(annual_growth),
         lag_quarterly_growth_4 = lag(quarterly_growth, 4),
         lag_double_dif_1 = lag(double_dif, 1),
         lag_double_dif_2 = lag(double_dif, 2),
         lag_double_dif_3 = lag(double_dif, 3),
         lag_double_dif_4 = lag(double_dif, 4)) %>%
  select(double_dif, lag_annual_growth, lag_quarterly_growth_4, lag_double_dif_1,
         lag_double_dif_2, lag_double_dif_3, lag_double_dif_4)
usa_ip_dif <- lm(double_dif ~ ., usa_ip_dif_data)

summary(usa_ip_dif)

# USA IP - test for seasonal unit roots
usa_ip_unit_roots_data <- industrial_production %>%
  mutate(t = 1:n(),
         pi_1_term =  lag(log_usip) + lag(log_usip, 2) + lag(log_usip, 3) + lag(log_usip, 4),
         pi_2_term = -lag(log_usip) + lag(log_usip, 2) - lag(log_usip, 3) + lag(log_usip, 4),
         pi_3_term = -lag(log_usip) + lag(log_usip, 3),
         pi_4_term = -lag(log_usip, 2) + lag(log_usip, 4),
         annual_growth_lag_1 = lag(annual_growth),
         annual_growth_lag_2 = lag(annual_growth, 2),
         annual_growth_lag_3 = lag(annual_growth, 3),
         annual_growth_lag_4 = lag(annual_growth, 4),
         annual_growth_lag_5 = lag(annual_growth, 5),
         annual_growth_lag_6 = lag(annual_growth, 6),
         annual_growth_lag_7 = lag(annual_growth, 7)) %>%
  select(D1, D2, D3, D4, t, pi_1_term, pi_2_term, pi_3_term, pi_4_term,
         starts_with("annual_growth"))
usa_ip_unit_roots <- lm(annual_growth ~ -1 + ., usa_ip_unit_roots_data)
summary(usa_ip_unit_roots)

# Note that the value for the t-test for pi_1 can be read from the table
# For the value of F-test we need to estimate the restricted model and get the sum of squared error and the degrees of freedom
restricted_model <- update(usa_ip_unit_roots, .~.-pi_2_term - pi_3_term - pi_4_term)
SSE_r <- sum(restricted_model$residuals^2)
SSE_ur <- sum(usa_ip_unit_roots$residuals^2)
q <- 3
k <- usa_ip_unit_roots$rank
n <- sum(complete.cases(usa_ip_unit_roots_data))

F_test_value <- ((SSE_r - SSE_ur)/q)/(SSE_ur/(n - k))


# Periodic autoregression of order 1 ------------------------------------------------------------------------------
periodic_autoreg_data <- industrial_production %>%
  mutate(lag_usip = lag(log_usip)) %>%
  select(log_usip, lag_usip, D1, D2, D3, D4)

periodic_autoreg <- lm(log_usip ~ -1 + D1*lag_usip + D2*lag_usip + D3*lag_usip + D4*lag_usip - lag_usip, data = periodic_autoreg_data)



# Aberrant observations -------------------------------------------------------------------------------------------
# Load data
##INSERT DATA HERE
fr_industrial_production <- XX

fr_industrial_production <- fr_industrial_production %>%
  select(date = `Datastream Code`,
         frip = FRIPTOT.G) %>%
  mutate(quarter = date) %>%
  mutate(date = case_when(
    str_detect(date, "Q1") ~ lubridate::ymd(str_c(str_extract(date, "[[:digit:]]{4}$"), "-01-01")),
    str_detect(date, "Q2") ~ lubridate::ymd(str_c(str_extract(date, "[[:digit:]]{4}$"), "-04-01")),
    str_detect(date, "Q3") ~ lubridate::ymd(str_c(str_extract(date, "[[:digit:]]{4}$"), "-07-01")),
    str_detect(date, "Q4") ~ lubridate::ymd(str_c(str_extract(date, "[[:digit:]]{4}$"), "-10-01")),
    TRUE ~ as.Date(NA)),
    quarter = str_extract(quarter, "Q+.")
  ) %>%
  mutate(log_frip = log(frip))

# Plot May 1968
ggplot(fr_industrial_production) +
  geom_line(aes(x = date, y = frip))

# Plot quarterly growth rates
ggplot(fr_industrial_production) +
  geom_line(aes(x = date, y = log_frip - lag(log_frip)))

# Black Monday for the Dow Jones
# Load data
dow_jones <- XX %>%
  select(date = Datastream, index = DJINDUS) %>%
  mutate(date = lubridate::as_date(date)) %>%
  filter(date <= (lubridate::mdy("12-30-2005"))) %>%
  filter(date >= (lubridate::mdy("12-29-1980"))) %>%
  mutate(growth_rate = log(index) - log(lag(index)))

ggplot(dow_jones) +
  geom_line(aes(date, growth_rate))

# Additive outlier (AO) -------------------------------------------------------------------------------------------
# Additive outlier at time 100
time <- 200 # total observations
tau <- 100 # moment of shock
sd <- 1
mean <- 0
phi_1 <- 0.8
xie <- 20

# Create errors
e <- rnorm(time, mean, sd)

# Create time series
y <- x <- rep(0,time)

for (i in 2:time){
  x[i] <- phi_1*x[i-1] + e[i]
  y[i] <- x[i-1] + xie * (tau == i)
}

# Put into tibble for plot
plot_tibble <- tibble(times = rep(1:time, 2),
                      value = c(x, y),
                      series = rep(c("x", "y"), each = time))

ggplot(plot_tibble) +
  geom_line(aes(x = times, y = value, color = series))

# What happens if AO is ignored?
ggplot(filter(plot_tibble, series == "y")) +
  geom_path(aes(x = value, y = lag(value)))



# Simple model for 911 -------------------------------------------------------------------------------------------
airline_revenues <- XX %>%
  select(month = Month, revenue = `AEA RPK TO`) %>%
  mutate(month = lubridate::as_date(month),
         log_revenue = log(revenue))

# Add dummy for 9/11 and onwards
airline_revenues <- airline_revenues %>%
  mutate(D911 = as.numeric(month >= lubridate::ymd("2001-11-09")),
         t = 1:n(),
         t_D911 = t*D911)

x_reg <- airline_revenues %>%
  select(t, t_D911) %>%
  as.matrix()

arima_911 <- forecast::Arima(y = airline_revenues$revenue, order = c(0, 0, 0),
                seasonal = list(order = c(1, 0, 0), period = 12), # Seasonal term to include only the twelth AR term
                xreg = x_reg) # Use trend and dummy*trend as external regressors.

