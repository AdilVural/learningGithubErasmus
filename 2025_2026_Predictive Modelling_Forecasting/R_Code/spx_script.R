
###--- Setup ---###

library(ggplot2)
library(lubridate)
library(forecast)
library(dplyr)

load('S&P500.RData')


###--- Analysis of closing price (not very interesting) ---###

ggplot(spx, aes(x = date, y = close)) + geom_line(col = 'blue') + 
  ggtitle('S&P500 closing price from 2010-01-01 until 2018-06-29')

# train arima model. 
arima <- auto.arima(spx$close)

df_arima <- data.frame(date = spx$date,
                       real = spx$close,
                       fitted = arima$fitted)

# not very interesting. Usually you want to predict weather a stock will
# go up or down
ggplot(df_arima, aes(x = date)) + 
  geom_line(aes(y = real), col = 'red') +
  geom_line(aes(y = fitted), col = 'blue') +
  ggtitle('S&P500 closing price (red) and fitted values (blue)')

#autoarima vs random walk = AR(1) with drift.
random_walk <- Arima(spx$close, order = c(1,0,0), include.drift = TRUE)


###--- Analysis daily stock returns (more interesting) ---###

# It is more interesting to predict the returns between y_t and y_{t-1}.
# that is: (y_t - y_{t-1}) / y_t
y_returns <- (spx$close - dplyr::lag(spx$close)) / spx$close
# multiply with 100 to obtain procentual change
y_returns <- 100 * y_returns

df_returns <- data.frame(date = spx$date,
                         return = y_returns)

# plot the returns
ggplot(df_returns, aes(x = date, y = return)) + 
  geom_line(col = 'blue') +
  ggtitle('Daily returns S&P 500') + 
  ylab('Return in %')

# find ARIMA model for daily returns. As expected, the returns are 
# very difficult to predict; there is no strong relationship between 
# return today and return of previous day. The weight is only equal
# to -0.05. 
arima_returns <- auto.arima(y_returns)

df_returns$fitted <- arima_returns$fitted

# let's see how accurate the model is in the months May and June 2018:
returns_june <- df_returns[df_returns$date >= as.Date('2018-05-01'), ]

# Not a strong predictive performance. 
ggplot(returns_june, aes(x = date)) + 
  geom_line(aes(y = return), col = 'red') +
  geom_line(aes(y = fitted), col = 'blue') +
  ggtitle('Daily returns S&P 500 (red) and fitted values (red)') + 
  ylab('Return in %')






