# R code week 2
# load libraries
library(dplyr)
library(readxl)
library(ggplot2)
library(stringr)
library(forecast)


# Log levels autocorrelations -------------------------------------------------------------------------------------
# load data
revenue_per_passenger <- read_excel("data/AEA revenue passenger kilometres.xlsx") %>%
  select(month = Month, revenue = "AEA RPK TO") %>%
  mutate(month = lubridate::ymd(month)) %>%
  mutate(log_revenue = log(revenue))

# Make use of the in-built R function
autocorrelations <- acf(revenue_per_passenger$log_revenue)

# Calculate the 2SE
two_SE <- 2* 1/sqrt(nrow(revenue_per_passenger))

# Create plot
plot_data <- tibble(acf = autocorrelations$acf) %>%
  mutate(time = 1:n(),
         two_SE = two_SE)
autocorrelations_plot <- ggplot(data = plot_data) +
  geom_line(aes(x = time, y = acf), color = 'blue') +
  geom_line(aes(x = time, y = two_SE), color = 'red') +
  ggtitle("log monthly revenue-passenger kilometers")

# Use own function and the information on the slide
#' Calculate the autocorrelation using the formula's from the slides
#'
#' @param time_series - a vector with a time serie
#' @param k_series - a vector of integers for the autocorrelation is calculated
#'
#' @return autocorrelation for each k
own_acf <- function(time_series, k_series = 1:24) {

  # initialize for-loop and set variables that do not vary over k
  rho <- rep(0, length(k_series))
  times <- length(time_series)
  gamma_0 <- var(time_series) * ((times - 1)/times)
  y_bar <- mean(time_series)

  # for each element in k,
  for (k in 1:length(k_series)) {
    # Calculate lag values
    lag_val <- tibble(y_t = time_series) %>%
      mutate(y_k = lag(time_series, k_series[[k]]))

    # Calculate inner product
    gamma_k <- lag_val %>%
      mutate(product = (y_t - y_bar) * (y_k - y_bar)) %>%
      filter(!is.na(product))

    # Sum and divide by times
    gamma_k <- sum(gamma_k$product)/times

    # Calculate the autocorrelation (denoted by the Greek letter rho)
    rho[[k]] <- gamma_k / gamma_0
  }
  return(rho)
}


# Monthly growth rates --------------------------------------------------------------------------------------------
monthly_growth_rate <- revenue_per_passenger %>%
  mutate(mgr = 100*log(revenue /lag(revenue))) %>%
  filter(!is.na(mgr))

monthly_acf <- acf(monthly_growth_rate$mgr, lag.max = 24) #Note that the acf-function adds an autocorrelation at 0
monthly_own_acf <- own_acf(monthly_growth_rate$mgr)

plot_data_monthly <- tibble(acf = monthly_acf$acf[-1]) %>%
  mutate(time = 1:n(),
         two_SE = two_SE,
         min_two_SE = -two_SE)

monthly_autocorrelations_plot <- ggplot(data = plot_data_monthly) +
  geom_line(aes(x = time, y = acf), color = 'blue') +
  geom_line(aes(x = time, y = two_SE), color = 'red') +
  geom_line(aes(x = time, y = min_two_SE), color = 'green') +
  ggtitle("Monthly growth rates, revenue-passenger kilometres")


# Unconditional distribution --------------------------------------------------------------------------------------
ggplot(data = monthly_growth_rate) +
  geom_histogram(aes(x = mgr/100), binwidth = 0.02) +
  ggtitle("Monthly growth rates of revenue-passenger kilometres")


# Effect of large shock -------------------------------------------------------------------------------------------
# Create AR(1) model, using the data from the exercise
ar_model <- Arima(x1, c(1, 0, 0)) # Change x1 to x2, or x3 respectively to create the plot for other values of phi
ar_plot_data <- tibble(y = ar_model$fitted,
                       x = x1) %>%
  mutate(n = 1:n())

ggplot(ar_plot_data) +
  geom_line(aes(x = n, y = x), color = "red") +
  geom_line(aes(x = n, y = y), color = "blue")


# Empirical autocorrelations --------------------------------------------------------------------------------------
# Using data from the exercise
acf_y <- tibble(acf.y1 = acf.y1$acf, acf.y2 = acf.y2$acf, acf.y3 = acf.y3$acf) %>%
  mutate(n = 1:n())

# Create plot
ggplot(acf_y) +
  geom_line(aes(x = n, y = acf.y1), color = "blue") +
  geom_line(aes(x = n, y = acf.y2), color = "red") +
  geom_line(aes(x = n, y = acf.y3), color = "green")


# Mean reversion --------------------------------------------------------------------------------------------------
# Create time series with mean = 0
T <- 200
time <- 1:T

e <- rnorm(T,0,1)

y1 <- y2 <- rep(0,T)

for (i in 2:T){
  y1[i] <- 0.8*y1[i-1] + e[i]
  y2[i] <- y2[i-1] + e[i]
}

# Gather data for ggplot
y_data <- tibble(y1 = y1, y2 = y2) %>%
  mutate(n = 1:n())

# Create plots
ggplot(y_data) +
  geom_line(aes(x = n, y = y1))

ggplot(y_data) +
  geom_line(aes(x = n, y = y2))


# Theoretical pattern, ACF of AR(1) with 𝜙_1=0.8 -----------------------------------------------------------------
phi_1 <- 0.8
rho_k <- rep(0, 20)
for (i in 1:length(rho_k)) {
  rho_k[[i]] <- phi_1^i
}

# turn into data for ggplot
rho_data <- tibble(rho_k = rho_k) %>%
  mutate(n = 1:n())

# Create plot
ggplot(rho_data) +
  geom_line(aes(x = n, y = rho_k))
