# R code week 5
# load libraries
library(dplyr)
library(readxl)
library(tidyr)
library(ggplot2)
library(stringr)


# A trend ---------------------------------------------------------------------------------------------------------
# load data
gdp_data <- read_excel("data/LA real GDP per capita.xlsx")

# Remove first year and turn into long format for plot
gdp_data <- gdp_data %>%
  filter(YEAR != 1950) %>%
  mutate(CHILE = as.numeric(CHILE)) %>%
  pivot_longer(cols = -YEAR, names_to = "country", values_to = "GDP")

ggplot(gdp_data) +
  geom_line(aes(x = YEAR, y = GDP, color = country))

# Trend, expansions and recessions
industrial_production <- read_excel("data/US industrial production.xlsx", skip = 1) %>%
  select(date = `Datastream Code`,
         usiptot_g = USIPTOT.G,
         usiptot_h = USIPTOT.H) %>%
  mutate(date = case_when(
    str_detect(date, "Q1") ~ lubridate::ymd(str_c(str_extract(date, "[[:digit:]]{4}$"), "-01-01")),
    str_detect(date, "Q2") ~ lubridate::ymd(str_c(str_extract(date, "[[:digit:]]{4}$"), "-04-01")),
    str_detect(date, "Q3") ~ lubridate::ymd(str_c(str_extract(date, "[[:digit:]]{4}$"), "-07-01")),
    str_detect(date, "Q4") ~ lubridate::ymd(str_c(str_extract(date, "[[:digit:]]{4}$"), "-10-01")),
    TRUE ~ as.Date(NA))
  )

ggplot(industrial_production) +
  geom_line(aes(x = date, y = usiptot_g))

# Trend with different slopes
stock_motorcycles <- read_excel("data/NL stock of motorcycles.xlsx") %>%
  select(year = YEAR, stock = MOTORSTOCK)

ggplot(data = stock_motorcycles) +
  geom_line(aes(x = year, y = stock))


# No trend in a stationary AR(1) ----------------------------------------------------------------------------------
mu <- 10
phi_1 <- 0.8

# Create time series
Time <- 200
time <- 1:Time

#generate a sequence of random disturbances
e <- data.frame(
  e_1 = rnorm(Time, 0, 1),
  e_2 = rnorm(Time, 0, 1),
  e_3 = rnorm(Time, 0, 1),
  e_4 = rnorm(Time, 0, 1)
)

x1 <- x2 <- x3 <- x4 <- rep(mu,Time)

#generate time series
for (i in 2:Time){
  x1[i] <- (phi_1*(x1[i-1] - mu) + e$e_1[i]) + mu
  x2[i] <- (phi_1*(x2[i-1] - mu) + e$e_2[i]) + mu
  x3[i] <- (phi_1*(x3[i-1] - mu) + e$e_3[i]) + mu
  x4[i] <- (phi_1*(x4[i-1] - mu) + e$e_4[i]) + mu

}

# Put time series in a data frame
series <- data.frame(
  time = time,
  x1 = x1,
  x2 = x2,
  x3 = x3,
  x4 = x4
)

# Create four plots in one overview
par(mfrow=c(2,2))
with(series, plot(time, x1, type = 'l'))
with(series, plot(time, x2, type = 'l'))
with(series, plot(time, x3, type = 'l'))
with(series, plot(time, x4, type = 'l'))
par(mfrow=c(1,1))


# Trend in a stationary AR(1) -------------------------------------------------------------------------------------
delta <- 0.1
mu <- 10
phi_1 <- 0.8

# Create time series
Time <- 200
time <- 1:Time

#generate a sequence of random disturbances
e <- data.frame(
  e_1 = rnorm(Time, 0, 1),
  e_2 = rnorm(Time, 0, 1),
  e_3 = rnorm(Time, 0, 1),
  e_4 = rnorm(Time, 0, 1)
)

x1_trend <- x2_trend <- x3_trend <- x4_trend <- rep(mu,Time)

#generate time series
for (i in 2:Time){
  x1_trend[i] <- (phi_1*(x1_trend[i-1] - mu - delta * (i - 1)) + e$e_1[i]) + mu + delta * i
  x2_trend[i] <- (phi_1*(x2_trend[i-1] - mu - delta * (i - 1)) + e$e_2[i]) + mu + delta * i
  x3_trend[i] <- (phi_1*(x3_trend[i-1] - mu - delta * (i - 1)) + e$e_3[i]) + mu + delta * i
  x4_trend[i] <- (phi_1*(x4_trend[i-1] - mu - delta * (i - 1)) + e$e_4[i]) + mu + delta * i
}

# Put time series in a data frame
series_trend <- data.frame(
  time = time,
  x1_trend = x1_trend,
  x2_trend = x2_trend,
  x3_trend = x3_trend,
  x4_trend = x4_trend
)

par(mfrow=c(2,2))
with(series_trend, plot(time, x1_trend, type = 'l'))
with(series_trend, plot(time, x2_trend, type = 'l'))
with(series_trend, plot(time, x3_trend, type = 'l'))
with(series_trend, plot(time, x4_trend, type = 'l'))
par(mfrow=c(1,1))


# Trend-stationarity ----------------------------------------------------------------------------------------------
# load data
gdp_data_trend <- read_excel("data/LA real GDP per capita.xlsx") %>%
  filter(YEAR != 1950)

# Get dependent variable and t
gdp_data_trend <- gdp_data_trend %>%
  mutate(log_argentina = log(ARGENTINA),
         t = 1:n())

# Estimate mu and delta
mu_delta <- coef(lm(log_argentina ~ t, data = gdp_data_trend))

# Get actual, fit and residual
gdp_data_trend <- gdp_data_trend %>%
  mutate(fit = mu_delta[[1]] + mu_delta[[2]] * t,
         actual = log_argentina,
         residual = actual - fit)

# Create plot
ggplot(gdp_data_trend) +
  #  geom_line(aes(t, residual), color = "blue") + # note that in the slides different scales on the left and right axis
  # are used, which is not possible with ggplot.
  geom_line(aes(t, actual), color = "red") +
  geom_line(aes(t, fit), color = "green")

# Reversion to trend ----------------------------------------------------------------------------------------------
# Create dependent variable and t
industrial_production <-  industrial_production %>%
  mutate(log_usip = log(usiptot_g),
         t = 1:n())

# Get mu and delta
mu_delta_ip <- coef(lm(log_usip ~ t, data = industrial_production))

# Get fit, actual and residual
industrial_production <- industrial_production %>%
  mutate(fit = mu_delta_ip[[1]] + mu_delta_ip[[2]] * t,
         actual = log_usip,
         residual = actual - fit)

# Create plot
ggplot(industrial_production) +
  geom_line(aes(t, actual), color = "red") +
  #  geom_line(aes(t, residual), color = "blue") + # note that in the slides different scales on the left axis are used, which is not possible with ggplot
  geom_line(aes(t, fit), color = "green")


# Trend in a stationary AR(1) -------------------------------------------------------------------------------------
delta <- 0.1
phi_1 <- 0.99
mu <- 10

# Create time series
Time <- 200
time <- 1:Time

#generate a sequence of random disturbances
e <- data.frame(
  e_1 = rnorm(Time, 0, 1),
  e_2 = rnorm(Time, 0, 1),
  e_3 = rnorm(Time, 0, 1),
  e_4 = rnorm(Time, 0, 1)
)

x1_ts <- x2_ts <- x3_ts <- x4_ts <- rep(mu,Time)

#generate time series
for (i in 2:Time){
  x1_ts[i] <- (phi_1*(x1_ts[i-1] - mu - delta * (i - 1)) + e$e_1[i]) + mu + delta * i
  x2_ts[i] <- (phi_1*(x2_ts[i-1] - mu - delta * (i - 1)) + e$e_2[i]) + mu + delta * i
  x3_ts[i] <- (phi_1*(x3_ts[i-1] - mu - delta * (i - 1)) + e$e_3[i]) + mu + delta * i
  x4_ts[i] <- (phi_1*(x4_ts[i-1] - mu - delta * (i - 1)) + e$e_4[i]) + mu + delta * i
}

# Put time series in tibble
series_ts <- tibble(
  time = time,
  x1_ts = x1_ts,
  x2_ts = x2_ts,
  x3_ts = x3_ts,
  x4_ts = x4_ts
)

# Estimate fit
coefs_x1 <- coef(lm(x1_ts ~ time, data = series_ts))
coefs_x2 <- coef(lm(x2_ts ~ time, data = series_ts))
coefs_x3 <- coef(lm(x3_ts ~ time, data = series_ts))
coefs_x4 <- coef(lm(x4_ts ~ time, data = series_ts))

# Add to tibble
series_ts <- series_ts %>%
  mutate(fit_x1 = coefs_x1[[1]] + coefs_x1[[2]] * time,
         residual_x1 = x1_ts - fit_x1,
         fit_x2 = coefs_x2[[1]] + coefs_x2[[2]] * time,
         residual_x2 = x2_ts - fit_x2,
         fit_x3 = coefs_x3[[1]] + coefs_x3[[2]] * time,
         residual_x3 = x3_ts - fit_x3,
         fit_x4 = coefs_x4[[1]] + coefs_x4[[2]] * time,
         residual_x4 = x4_ts - fit_x4)

# Create plot for first time series
ggplot(series_ts) +
  geom_line(aes(time, fit_x1), color = "green") +
  geom_line(aes(time, x1_ts), color = "red") +
  geom_line(aes(time, residual_x1), color = "blue")



# Autocorrelations ------------------------------------------------------------------------------------------------
# Note that this only hodls for phi_1 = 1
delta <- 0.1

y_0 <- 10
exp_y_t <- function(t) {y_0 + delta * t}
y_t <- exp_y_t(1:45)
acf(y_t)

# Stochastic trend AR(1) -------------------------------------------------------------------------------------
delta <- 0.1
phi_1 <- 1
mu <- c(-2, -4, 10, 6)

# Create time series
Time <- 200
time <- 1:Time

#generate a sequence of random disturbances
e <- data.frame(
  e_1 = rnorm(Time, 0, 1),
  e_2 = rnorm(Time, 0, 1),
  e_3 = rnorm(Time, 0, 1),
  e_4 = rnorm(Time, 0, 1)
)

x1_st <- x2_st <- x3_st <- x4_st <- rep(mu[1],Time)

#generate time series
for (i in 2:Time){
  x1_st[i] <- (phi_1*(x1_st[i-1] - mu[1] - delta * (i - 1)) + e$e_1[i]) + mu[1] + delta * i
  x2_st[i] <- (phi_1*(x2_st[i-1] - mu[2] - delta * (i - 1)) + e$e_2[i]) + mu[2] + delta * i
  x3_st[i] <- (phi_1*(x3_st[i-1] - mu[3] - delta * (i - 1)) + e$e_3[i]) + mu[3] + delta * i
  x4_st[i] <- (phi_1*(x4_st[i-1] - mu[4] - delta * (i - 1)) + e$e_4[i]) + mu[4] + delta * i
}

# Put time series in tibble
series_st <- tibble(
  time = time,
  x1_st = x1_st,
  x2_st = x2_st,
  x3_st = x3_st,
  x4_st = x4_st
)

# Estimate fit
coefs_x1 <- coef(lm(x1_st ~ time, data = series_st))
coefs_x2 <- coef(lm(x2_st ~ time, data = series_st))
coefs_x3 <- coef(lm(x3_st ~ time, data = series_st))
coefs_x4 <- coef(lm(x4_st ~ time, data = series_st))

# Add to tibble
series_st <- series_st %>%
  mutate(x1_fit = coefs_x1[[1]] + coefs_x1[[2]] * time,
         x2_fit = coefs_x2[[1]] + coefs_x2[[2]] * time,
         x3_fit = coefs_x3[[1]] + coefs_x3[[2]] * time,
         x4_fit = coefs_x4[[1]] + coefs_x4[[2]] * time,
         x1_residual = x1_st - x1_fit,
         x2_residual = x2_st - x2_fit,
         x3_residual = x3_st - x3_fit,
         x4_residual = x4_st - x4_fit)

# Create plot for first time series
ggplot(series_st) +
  geom_line(aes(time, x1_fit), color = "green") +
  geom_line(aes(time, x1_st), color = "red") +
  geom_line(aes(time, x1_residual), color = "blue")


# Logs of USE industrial production  -----------------------------------------------------------------------------------
industrial_production <- industrial_production %>%
  mutate(log_usip_sa = 4*100*log(usiptot_g/lag(usiptot_g, 4))) %>% # Adjust for seasonality
  mutate(first_difference_log = log_usip_sa - lag(log_usip_sa))

industrial_production_lm <- lm(first_difference_log ~ lag(log_usip_sa) + t + lag(first_difference_log) + lag(first_difference_log, 2),
                               data = industrial_production)

summary(industrial_production_lm)

# Returns on the Dow Jones index ---------------------------------------------------------------------------------------
# Load data
dow_jones <- read_excel("data/US Dow Jones Industrials index.xlsx", skip = 1) %>%
  select(date = Datastream, index = DJINDUS) %>%
  mutate(date = lubridate::as_date(date)) %>%
  filter(date <= (lubridate::mdy("12-30-2005"))) %>%
  filter(date >= (lubridate::mdy("12-29-1989"))) %>%
  mutate(index = log(index) * 100)

# Calculate first differences
dow_jones <- dow_jones %>%
  mutate(first_difference = lag(index) - index,
         lag_index = lag(index))

# Calculate regression
dow_jones_lm <- lm(first_difference ~ lag_index, data = dow_jones)
summary(dow_jones_lm)
