# R code week 4

# !!IMPORTANT!!
# This R code is made to aid you to reproduce the figures and tables from the slides of week 4.
# Note that the script does NOT actually work since the data is missing.
# load libraries
library(dplyr)
library(readxl)
library(lmtest)
library(ggplot2)
library(moments)
library(tseries)

# Running example: Earthquakes -------------------------------------------------------------------
# load data
##INSERT DATA HERE
earthquakes <- XX
colnames(earthquakes) <- c('year', 'nr')

# plot
ggplot(earthquakes) +
  geom_line(aes(x = year, y = nr))

# plot estimation sample
estimation_sample <- earthquakes %>%
  filter(year <= 1980)
ggplot(estimation_sample) +
  geom_line(aes(x = year, y = nr))

# plot hold-out sample
hold_out_sample <- earthquakes %>%
  filter(year > 1980)
ggplot(hold_out_sample) +
  geom_line(aes(x = year, y = nr))


# (Partial) autocorrelations --------------------------------------------------------------------------------------
max_lag <- 12
acf_pacf <- tibble(lag = 1:max_lag,
                   acf = acf(estimation_sample$nr, max_lag)$acf[2:(max_lag + 1)],
                   pacf = pacf(estimation_sample$nr, max_lag)$acf)

# Note that R automatically colors negative numbers, whilst in the slides the significant (partial)
# autocorrelations are colored.


# AR(1) for Earthquakes -------------------------------------------------------------------------------------------
ar_1 <- lm(nr ~ lag(nr), data = estimation_sample)

summary(ar_1)

# Residuals -------------------------------------------------------------------------------------------------------
actuals_predict <- tibble(
  actual = estimation_sample$nr[2:nrow(estimation_sample)],
  fitted = ar_1$fitted.values,
  residual = ar_1$residuals
) %>%
  mutate(n = 1:n())

ggplot(actuals_predict, aes(x = n)) +
  geom_line(aes(y = actual), color = "red") +
  geom_line(aes(y = fitted), color = "green") +
  geom_line(aes(y = residual), color = "blue")

# Ljung box test --------------------------------------------------------------------------------------------------
#lb_test_result <- Box.test(actuals_predict$residual, lag = 1, type = c("Ljung-Box"), fitdf = 0)


# A more specific test for residual autocorrelation ---------------------------------------------------------------
# Example with ARMA(p, r)
p <- 2
r <- 2
arma_p_r <- arima(estimation_sample$nr, c(p, 0, r))
example_data <- estimation_sample %>%
  mutate(residual = as.numeric(arma_p_r$residuals))

# Add necessary lags for the auxiliary regression
aux_reg_data <- example_data %>%
  select(residual, nr)
for (i in 1:p) {
  aux_reg_data <- aux_reg_data %>%
    mutate(!!sym(str_c("lag_nr_", i)) := lag(nr, i))
}

for (j in 1:r) {
  aux_reg_data <- aux_reg_data %>%
    mutate(!!sym(str_c("lag_residual_", j)) := lag(residual, j))
}


# auxiliary regression
aux_reg <- lm(residual ~ . -nr, data = aux_reg_data)
R_squared <- summary(aux_reg)$r.squared
test_stat <- nrow(estimation_sample) * R_squared
critical_value <- qchisq(0.95, df = r)

if (test_stat < critical_value) {
  print("Failed to reject H0: The AR(p) model is adequate")
} else if (test_stat > critical_value) {
  print("H0: The AR(p) model is adequate is being rejected")
} else{
  print("Something unexpected happened.")
}


# Breusch_Godfrey Serial Correlation LM test ----------------------------------------------------------------------
#perform Breusch-Godfrey test
bg_test_result <- bgtest(nr ~ lag(nr), order = 2, data = estimation_sample)

bg_test_result
bg_test_result$coefficients

# Normality, approximately? ---------------------------------------------------------------------------------------
ggplot(actuals_predict) +
  geom_histogram(aes(x = residual), bins = 14)

summary(actuals_predict$residual)
sd(actuals_predict$residual)
skewness(actuals_predict$residual)
kurtosis(actuals_predict$residual)

jarque.test(actuals_predict$residual)


# Alternative models for Earthquakes ------------------------------------------------------------------------------
#AR(2)
ar_2 <- lm(nr ~ lag(nr) + lag(nr, 2), data = estimation_sample)

#ARMA(1, 1)
arma_1_1 <- arma(estimation_sample$nr, order = c(1, 1))

# Recursive 1-step-ahead forecasts --------------------------------------------------------------------------------
# AR(1) model
# Create forecasts
forecasts_ar_1 <- tibble(
  forecasts = predict(ar_1, newdata = bind_rows(tail(estimation_sample, 1), hold_out_sample))[-1],
  t = 1981:2005,
  two_SE = 2*sd(estimation_sample$nr)
)

# Create plot
ggplot(forecasts_ar_1) +
  geom_line(aes(x = t, y = forecasts), col = "blue") +
  geom_line(aes(x = t, y = forecasts + two_SE), col = "red") +
  geom_line(aes(x = t, y = forecasts - two_SE), col = "red")

# ARMA(1,1)-model
# Create forecasts
# Initialize for-loop
n_est <- nrow(estimation_sample)
n_hold_out <- nrow(hold_out_sample)
one_step_ahead_forecasts <- tibble(t = n_est:(n_est + n_hold_out - 1),
                                   forecast = NA_real_,
                                   error = NA_real_,
                                   actual = hold_out_sample$nr)

for (i in one_step_ahead_forecasts$t) {
  if (i == first(one_step_ahead_forecasts$t)) {
    # Get the last observations from the training set to calculate the first forecast
    lag_error <- last(arma_1_1$residuals)
    lag_y <- last(estimation_sample$nr)
  } else {
    # Get the lags for the other forecasts
    lag_error <- one_step_ahead_forecasts %>%
      filter(t == (i-1)) %>%
      pull(error)

    lag_y <- one_step_ahead_forecasts %>%
      filter(t == (i-1)) %>%
      pull(forecast)
  }

  # Calculate the predictions
  prediction_i <- coef(arma_1_1) %*% c(lag_y, lag_error, 1)

  # Fill in the tibble
  one_step_ahead_forecasts <- one_step_ahead_forecasts %>%
    mutate(forecast = if_else(t == i, prediction_i, forecast)) %>%
    mutate(error = forecast - actual)
}

forecasts_arma <- tibble(
  forecasts = one_step_ahead_forecasts$forecast,
  t = 1981:2005,
  two_SE = 2*sd(estimation_sample$nr)
)

# Create plot
ggplot(forecasts_arma) +
  geom_line(aes(x = t, y = forecasts), col = "blue") +
  geom_line(aes(x = t, y = forecasts + two_SE), col = "red") +
  geom_line(aes(x = t, y = forecasts - two_SE), col = "red")

# Show both data (estimation and hold-out) in plot
# Combine data
complete_forecasts_arma <- earthquakes %>%
  left_join(forecasts_arma %>% select(t, forecasts), by = c("year" = "t"))

ggplot(complete_forecasts_arma) +
  geom_line(aes(x = year, y = nr), col = "blue") +
  geom_line(aes(x = year, y = forecasts), col = "red")


# Forecasts from Earthquakes from ARMA(1,1) from 2005 onwards -----------------------------------------------------
# re-estimation
#ARMA(1, 1)
arma_1_1_all_data <- arma(earthquakes$nr, order = c(1, 1))
coef(arma_1_1_all_data)

# Get predictions
# Initialize for-loop
one_step_ahead_forecasts_full <- tibble(t = 2006:2020,
                                        forecast = NA_real_)
coefs <- coef(arma_1_1_all_data)

for (i in one_step_ahead_forecasts_full$t) {
  if (i == first(one_step_ahead_forecasts_full$t)) {
    # Get the last observations from the training set to calculate the first forecast
    lag_error <- last(arma_1_1_all_data$residuals)
    lag_y <- last(earthquakes$nr)
  } else {
    # Get the lags for the other forecasts
    lag_y <- one_step_ahead_forecasts_full %>%
      filter(t == (i-1)) %>%
      pull(forecast)

    lag_error <- 0
  }

  # Get step ahead prediction
  prediction_i <- coefs %*% c(lag_y, lag_error, 1)

  # Fill in the tibble
  one_step_ahead_forecasts_full <- one_step_ahead_forecasts_full %>%
    mutate(forecast = if_else(t == i, prediction_i, forecast))
}

# Create plot
ggplot(one_step_ahead_forecasts_full) +
  geom_line(aes(t, forecast))
