###--- Setup ---###

# Load libraries needed for analysis
library(forecast)
library(ggplot2)
library(lubridate)
library(tidyr)

load('bike_data.RData')


###--- Preliminary analysis ---###

# plot of bike data
ggplot(bikes, aes(x = date, y = count)) + 
  geom_line(col = 'blue', size = 0.7) + 
  ggtitle('Bike rentals January 2011 - December 2012' )


# note that there are some outliers. We will carelessly remove them. Note that
# in practice you will have to be more careful when removing outliers.
outlier_indices <- tsoutliers(bikes$count)$index

bikes <- bikes[-outlier_indices, ]

# plot of bike data without outliers:
ggplot(bikes, aes(x = date, y = count)) + 
  geom_line(col = 'blue', size = 0.7) + 
  ggtitle('Bike rentals January 2011 - December 2012 without outliers' )


###--- Model Training ---###

count <- bikes$count
n_obs <- length(count)
n_holdout <- 100

# transform original counts into time series (ts) object. Be carefull when taking
# a daily frequency of 365; the ts() model does NOT consider leap years (schrikkeljaar).
# this is problematic when the time series consists of many years. Also we removed outliers,
# which means that the frequency does NOT equal 365 days. For this reason, we do not specify 
# the frequency
count_ts <- ts(count)

# define the indices of our train sample
train_index <- 1:(n_obs - n_holdout)

# create train set/sample
train <- ts(count[train_index])

# create test set/sample
test <- ts(count[-train_index])

# find non-seasonal arima model that leads to lowest AIC value
fit_arima <- auto.arima(train, ic = 'aic', seasonal = FALSE)

# now we only want a non-seasonal ar(p) model, with a maximum of 7 lags, based on the AIC:
fit_ar <- auto.arima(train, max.d = 0, max.q = 0, ic = 'aic', max.p = 7,
                     seasonal = FALSE)

# now we only want a non-seasonal moving average model ma(q):
fit_ma <- auto.arima(train, max.d = 0, max.p = 0, ic = 'aic',
                     seasonal = FALSE)


###--- Model performance 100 days ahead ---###
# obtain predictions for the next 100 days using the forecast() function. 
# Note that these are NOT one-step ahead predictions
pred_arima <- forecast(test, h = n_holdout, model = fit_arima)
pred_ar <- forecast(test, h = n_holdout, model = fit_ar)
pred_ma <- forecast(test, h = n_holdout, model = fit_ma)

# construct new data frames for fit train data
model_res_train <- data.frame(date = bikes$date[train_index],
                              real = bikes$count[train_index],
                              arima = c(fit_arima$fitted),
                              ar = c(fit_ar$fitted),
                              ma = c(fit_ma$fitted))

model_res_train <- gather(model_res_train, dgp, count, -1, factor_key = TRUE)

ggplot(model_res_train, aes(x = date, y = count, col = dgp)) + geom_line(size=0.01)

forecasts_df <- data.frame(date = bikes$date[-train_index],
                          real = bikes$count[-train_index],
                          arima = c(pred_arima$mean),
                          ar = c(pred_ar$mean),
                          ma = c(pred_ma$mean))

forecasts_df <- gather(forecasts_df, dgp, count, -1, factor_key = TRUE)

# not very useful to make perdictions so far in the future.
ggplot(forecasts_df, aes(x = date, y = count, col = dgp)) + geom_line(size = 0.01) + 
  ggtitle(paste('Forecasts for the next', n_holdout, 'days'))


### --- Now we evaluate the 1-step ahead forecasts for the next 100 days ---###

# note that we use the whole dataset
full_arima <- Arima(count_ts, model = fit_arima)
# obtain the results of the 1 step ahead forecasts for the test dataset
arima_1_ahead <- full_arima$fitted[-train_index]
full_ar <- Arima(count_ts, model = fit_ar)
ar_1_ahead <- full_ar$fitted[-train_index]
full_ma <- Arima(count_ts, model = fit_ma)
ma_1_ahead <- full_ma$fitted[-train_index]

# actual number of rented bikes for the test set
real_count <- bikes$count[-train_index]

df_1_ahead <- data.frame(date = bikes$date[-train_index],
                         real = real_count,
                         arima = arima_1_ahead,
                         ar = ar_1_ahead,
                         ma = ma_1_ahead)

df_1_ahead <- gather(df_1_ahead, dgp, count, -1, factor_key = TRUE)

# not plot the results of the 1 step ahead forecasts for the test sample:
ggplot(df_1_ahead, aes(x = date, y = count, col = dgp)) + geom_line() + geom_point() +
  ggtitle('One step ahead forecasts for the test sample.')


# not fully clear which model performs best; use RMSE for the test set.
# The RMSE is simply the root of the MSPE
rmse_arima <- mean((arima_1_ahead - real_count)^2)^0.5
rmse_ar <- mean((ar_1_ahead - real_count)^2)^0.5
rmse_ma <- mean((ma_1_ahead - real_count)^2)^0.5










