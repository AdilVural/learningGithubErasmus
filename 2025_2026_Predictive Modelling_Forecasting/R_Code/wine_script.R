###--- Setup ---###

library(forecast)
library(ggplot2)
library(lubridate)

load('wine_sales.RData')


###--- Preliminary analysis ---###

# plot of wine data
ggplot(wine_sales, aes(x = date, y = sales)) + 
  geom_line(col = 'blue', size = 1) + 
  ggtitle('Historical wine sales in Australia since 1980' )


###--- Training the model ---###

n_obs <- length(wine_sales$sales)
n_holdout <- 12

# transform original sales into time series (ts) object
# since we are interested in modeling seasonality, it is crucial
# to include the frequency. In our case, we assume that a pattern 
# repeats itself every year -> frequency = 12 months. If a pattern 
# occurs every week, and we have daily data, then the frequency is 
# 1 week -> 7 days. If a patern repeats itself yearly, then the 
# frequency is 365 days if your observations are measured daily. There 
# is a problem when you have a leap year (schrikkeljaar) or if some days
# are not recorded, for example due to holidays. For this reason, it 
# is NOT recommended to use frequency = 365
wine_ts <- ts(wine_sales$sales, frequency = 12, start = c(1980,1))

# define the indices of our train sample
train_index <- 1:(n_obs - n_holdout)

# create train and test sample
train <- ts(wine_sales$sales[train_index], frequency = 12, start = c(1980, 1))

# extract last date
end_date <- wine_sales$date[n_obs]
# transform date into vector containing year and month
end <- c(year(end_date), month(end_date))
test <- ts(wine_sales$sales[-train_index], frequency = 12, end = end)

## The code below does not work as you would expect; the result is a vector
## and not a ts_object. This is problematic when you want to model seasonality
## since the frequency is necessary. Use the approach shown above or the 
## windows() function to subset time series.
# train <- wine_ts[train_index] # <- wrong
# test <- wine_ts[-train_index] # <- wrong


# find arima model that leads to lowest AIC value
fit_arima <- auto.arima(train, ic = 'aic')

# now we use the Arima() function to obtain the predictions of the
# out of sample observations. Note that since we pass the model as
# an argument, we do NOT train on the new dataset.
full_arima <- Arima(wine_ts, model = fit_arima)

# This does not work since you need at least 12 months of data due
# to the seasonality.
#pred_arima <- forecast(test, h = n_holdout, model = fit_arima)
#pred_ar <- forecast(test, h = n_holdout, model = fit_ar)


###--- Evaluate model performance ---###

train_res <- data.frame(date = wine_sales$date[train_index],
                        sales = wine_sales$sales[train_index],
                        arima = full_arima$fitted[train_index])

test_res <- data.frame(date = wine_sales$date[-train_index],
                       sales = wine_sales$sales[-train_index],
                       arima = full_arima$fitted[-train_index])


# plot of the in sample model fit (train set)
ggplot(train_res, aes(x = date)) + 
  geom_line(aes(y = sales), col = 'red', size = 0.65) + 
  geom_line(aes(y = arima), col = 'blue', size = 0.65) +
  geom_point(aes(y = sales), col = 'red', size  = 1) +
  geom_point(aes(y = arima), col = 'blue', size = 1) + 
  ggtitle('Real sales (red) vs fitted sales (blue).')

# plot of the out-of-sample model predictions (test set)
ggplot(test_res, aes(x = date)) + 
  geom_line(aes(y = sales), col = 'red', size = 0.65) + 
  geom_line(aes(y = arima), col = 'blue', size = 0.65) +
  geom_point(aes(y = sales), col = 'red', size  = 1) +
  geom_point(aes(y = arima), col = 'blue', size = 1) + 
  ggtitle('Real sales (red) vs predicted sales (blue).')
  


