# Lecture 5 Exercises
# Load libraries
library(ggplot2)
library(tseries)

# Lecture 5 Exercise 1:

# Create time series
T <- 300
trend <- 0:(T-1)
dt <- 0.1*trend

e <- rnorm(T,0,1)

x <- y <- rep(0,T)

for (i in 2:T) {
  x[i] <- 0.1*trend[i] + 0.5*(x[i-1] - 0.1*trend[i-1]) + e[i]
  y[i] <- y[i-1] + 0.1 + e[i]
}

# Put into data frame
series <- data.frame(
  time = 1:T,
  x = x,
  y = y,
  dt = dt
)

# Create plot
ggplot(series, aes(x = time)) +
  geom_line(aes(y=x, color = "x")) +
  geom_line(aes(y=y, color = "y")) +
  geom_line(aes(y=dt, color = "dt")) +
  xlab("time") + ylab("x,y,dt") +
  scale_colour_manual(
    "", values = c("x"="red","y"="blue","dt"="green"))


# Lecture 5 Exercise 2:
# Create time series with shock
T <- 300
add <- rep(0,T)
add[150:T] <- 5
time <- 1:T

e <- rnorm(T,0,1)

x <- y <- rep(0,T)

for (i in 2:T){
  x[i] <- add[i] + 0.8*(x[i-1] - add[i]) + e[i]
  y[i] <- 0.8*y[i-1] + e[i]
}

# Put into dataframe
series <- data.frame(
  time = time,
  x = x,
  y = y
)

# Create plot
ggplot(series, aes(x = time)) +
  geom_line(aes(y=x, color = "x")) +
  geom_line(aes(y=y, color = "y")) +
  xlab("time") + ylab("x,y") +
  scale_colour_manual("", values = c("x"="red","y"="blue"))

#Use the adf.test function from the tseries package for
#the Augmented Dickey-Fuller test
lag_order <- 1
test_x <- adf.test(x, k = lag_order)
test_y <- adf.test(y, k = lag_order)
