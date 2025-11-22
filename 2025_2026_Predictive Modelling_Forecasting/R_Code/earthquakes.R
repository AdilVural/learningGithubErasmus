###--- Setup; load packages and read data ---###

library(readxl)
library(ggplot2)
library(dplyr)
library(forecast)
library(tseries) # For Jarque-Bera test

quakes <- read_xlsx('data/earthquakes.xlsx', col_names = FALSE)
colnames(quakes) <- c('year', 'nr_quakes')
# create lagged first difference; will be used later
quakes$lag_difference <- lag(quakes$nr_quakes, n = 1) - lag(quakes$nr_quakes, n = 2)


###--- Some preliminary analysis ---###

# Some visualization
ggplot(quakes, aes(x = year, y = nr_quakes)) + geom_line(col = 'blue', size = 1) +
  xlab('Year') + ylab('Number of earthquakes') + ggtitle('Number of earthquakes 1900-2005') +
  theme(axis.text=element_text(size=11),
        axis.title=element_text(size=12),
        plot.title = element_text(hjust = 0.5, size = 13, face = 'bold'))


# Create estimation/train sample 1900-1980
estimation <- filter(quakes, year <= 1980)

# Some visualization
ggplot(estimation, aes(x = year, y = nr_quakes)) + geom_line(col = 'blue', size = 1) +
  xlab('Year') + ylab('Number of earthquakes') +
  ggtitle('Number of earthquakes 1900-1980; estimation sample') +
  theme(axis.text=element_text(size=11),
        axis.title=element_text(size=12),
        plot.title = element_text(hjust = 0.5, size = 13, face = 'bold'))


# Create hold-out/test sample after 1980
hold_out <- filter(quakes, year > 1980)

# Some visualization
ggplot(hold_out, aes(x = year, y = nr_quakes)) + geom_line(col = 'blue', size = 1) +
  xlab('Year') + ylab('Number of earthquakes') +
  ggtitle('Number of earthquakes 1981-2005; hold-out sample') +
  theme(axis.text=element_text(size=11),
        axis.title=element_text(size=12),
        plot.title = element_text(hjust = 0.5, size = 13, face = 'bold'))


max_lag = 12
# plot autocorrelation
acf_quakes <- acf(estimation$nr_quakes, lag.max = max_lag)
# plot partial autocorrelation
pacf_quakes <- pacf(estimation$nr_quakes, lag.max = max_lag)

quakes_stats <- data.frame(acf = acf_quakes$acf[-1], # remove first corelation
                           pacf = pacf_quakes$acf,
                           Q_stat = NA,
                           p_val = NA)

# Now compute p-values and Q-statistic
for (i in 1:max_lag) {
  # Compute Q-statistic
  corr_stats <- Box.test(estimation$nr_quakes, lag = i,
                         type = 'Ljung-Box')
  quakes_stats$Q_stat[i] <- corr_stats$statistic
  quakes_stats$p_val[i] <- corr_stats$p.value
}


###--- Start with the model-building process ---###

# Since the arima function does not compute p-values, we implement our
# own function that takes the output of the Arima function as input, and then
# computes the p-value and t-statistic.
compute_pval <- function(arima_res) {
  t_stat <- arima_res$coef / diag(arima_res$var.coef^0.5)
  pval <- 2*pt(-abs(t_stat), df = length(arima_res$residuals)-1)
  list(pval = pval, t_statistic = t_stat)
}


# We start with a simple ar(1) model;
ar1 <- Arima(estimation$nr_quakes, order = c(1, 0, 0))
compute_pval(ar1)

resid_ar1 <- data.frame(year = estimation$year,
                        residuals = ar1$residuals)

# plot the residuals for ar1 model
ggplot(resid_ar1, aes(x = year, y = residuals)) + geom_line(col = 'blue', size = 1) +
  geom_hline(yintercept = -sqrt(ar1$sigma2), linetype = 2) +
  geom_hline(yintercept = sqrt(ar1$sigma2), linetype = 2) +
  xlab('Year') + ylab('Residuals') + ggtitle('Residuals estimation sample; AR(1) model') +
  theme(axis.text=element_text(size=11),
        axis.title=element_text(size=12),
        plot.title = element_text(hjust = 0.5, size = 13, face = 'bold'))

fit_ar1 <- data.frame(year = estimation$year,
                      real = estimation$nr_quakes,
                      fit = ar1$fitted)

# lets plot the auto correlation functions:
acf(ar1$residuals)

# also plot a histogram of the residuals
ggplot(resid_ar1, aes(x = residuals)) + geom_histogram(col = 'skyblue', fill = 'skyblue', bins = 20)

# Now we test whether the residuals are normally distributed. Neither of the two tests
# reject the nul-hypothesis that the residuals follow a normal distribution). In other words, we
# can assume that the residuals follow a normal distribution.
# First we test normality with the Jarque-Bera test
jarque_res <- jarque.bera.test(ar1$residuals)
# and then with the more familiar Shapiro-Wilk test
shapiro_test <- shapiro.test(ar1$residuals)


# We can also plot the fit of the estimation sample for the ar1 model.
ggplot(fit_ar1, aes(x = year)) +
  geom_line(aes(y = fit), col = 'blue', size = 1) +
  geom_line(aes(y = real), col = 'red', size = 1) +
  xlab('Year') + ylab('Number of earthquakes') +
  ggtitle('Fitted (blue) vs real (red) values estimation sample; AR(1) model') +
  theme(axis.text=element_text(size=11),
        axis.title=element_text(size=12),
        plot.title = element_text(hjust = 0.5, size = 13, face = 'bold'))


# Now we try to improve the model by adding the lag of the first-difference: 'y_{t-1} - y_{t-2}'.
# Since this model cannot be constructed with the 'order = c(p, d, q)' argument, we pass the lag
# of the first difference as a variable/regressor.
ar_diff <- Arima(estimation$nr_quakes, order = c(1, 0, 0),
                 xreg = estimation$lag_difference)
compute_pval(ar_diff) # Lag_difference not significant

# Maybe we can improve the model by adding 2 moving average terms?
arma12 <- Arima(estimation$nr_quakes, order = c(1,0,2))
# get pvalue and t-statistic
compute_pval(arma12)

# Or maybe we can improve the model by adding a second lag of y
ar2 <- Arima(estimation$nr_quakes, order = c(2,0,0))
# get p-value and t-statistic for ar(2) model
compute_pval(ar2)

# Let's try an arma(1,1) model:
arma11 <- Arima(estimation$nr_quakes, order = c(1,0,1))
# get p-value and t-statistic for arma(1,1) model
compute_pval(arma11)


# now we can start making predictions. To start, we will do recursive
# 1-step ahead predictions. Note that the model is already trained. All
# we are doing, is fitting the model with the new data. The Arima() function
# does not re-train the model when you pass another model as an argument.
ar1_full <- Arima(quakes$nr_quakes, model = ar1)
# the coefficients should not change when we add new data.
identical(ar1_full$coef, ar1$coef) # Should evaluate to TRUE
# extract all 'fitted' values after 1980. This way of extracting works since in
# this case the fitted values in the ar1 model correspond to the observations in quakes.
pred_onestep <- ar1_full$fitted[quakes$year > 1980]

onestep_df <- data.frame(year = quakes$year,
                         real = quakes$nr_quakes,
                         predictions = c(rep(NA, nrow(estimation)), pred_onestep))
# plot the one step ahead predictions after 1980 with the real values and the historical
# number of earthquakes
ggplot(onestep_df, aes(x = year)) +
  geom_line(aes(y = predictions), col = 'blue', size = 1) +
  geom_line(aes(y = real), col = 'red', size = 1) +
  geom_vline(xintercept = 1980, linetype = 2, size = 1) +
  xlab('Year') + ylab('Number of Earthquakes') +
  ggtitle('One step ahead predictions (blue) vs real (red) number of earthquakes; AR(1) model') +
  theme(axis.text=element_text(size=11),
        axis.title=element_text(size=12),
        plot.title = element_text(hjust = 0.5, size = 13, face = 'bold'))

# Now we will add 95% confidence intervals to the previous plot. Note that the confidence interval is
# approximately given by the: predicted y +/- standarddev(y).
upper <- pred_onestep + 2*ar1$sigma2^0.5
lower <- pred_onestep - 2*ar1$sigma2^0.5
onestep_df$upper <- c(rep(NA, nrow(estimation)), upper)
onestep_df$lower <- c(rep(NA, nrow(estimation)), lower)

ggplot(onestep_df, aes(x = year)) +
  geom_line(aes(y = predictions), col = 'blue', size = 1) +
  geom_line(aes(y = real), col = 'red', size = 1) +
  geom_line(aes(y = upper), size = 0.5, linetype = 2) +
  geom_line(aes(y = lower), size = 0.5, linetype = 2) +
  geom_vline(xintercept = 1980, linetype = 2, size = 1) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
  xlab('Year') + ylab('Number of Earthquakes') +
  ggtitle('One step ahead predictions (blue) vs real (red) number of earthquakes; AR(1) model') +
  theme(axis.text=element_text(size=11),
        axis.title=element_text(size=12),
        plot.title = element_text(hjust = 0.5, size = 13, face = 'bold'))


# As a final exercise, we will now use the full earthquake time series to predict the
# number of earthquakes 15 years into the future.
# First, we re-train the ar1 model using the complete time series
ar1_new <- Arima(quakes$nr_quakes, order = c(1, 0, 0))
# forecast 15 years into the future
forecast_15yrs <- forecast(ar1_new, h = 15)

forecast_df <- data.frame(year = 1900:2020,
                          nr_quakes = c(quakes$nr_quakes, forecast_15yrs$mean),
                          upper = c(rep(NA, nrow(quakes)), forecast_15yrs$upper[, '95%']),
                          lower = c(rep(NA, nrow(quakes)), forecast_15yrs$lower[, '95%']))

ggplot(forecast_df, aes(x = year)) +
  geom_line(aes(y = nr_quakes), col = 'red', size = 1) +
  geom_vline(xintercept = 2005, linetype = 2, size = 1) +
  geom_line(aes(y = upper), size = 0.5, linetype = 2) +
  geom_line(aes(y = lower), size = 0.5, linetype = 2) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
  xlab('Year') + ylab('Number of Earthquakes') +
  ggtitle('Forecast of the number of earthquakes after 2005 with 95% C.I.; AR(1) model') +
  theme(axis.text=element_text(size=11),
        axis.title=element_text(size=12),
        plot.title = element_text(hjust = 0.5, size = 13, face = 'bold'))


