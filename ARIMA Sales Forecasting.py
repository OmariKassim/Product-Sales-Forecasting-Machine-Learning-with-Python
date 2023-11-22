#!/usr/bin/env python
# coding: utf-8

# # ARIMA Weekly Sale Forecasting

# Importing relevant python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime

# Loading data
data = pd.read_csv(r"C:\Users\omarm\Documents\Career Essentials & Job Apps\Portfolio\Weekly Sales Data.csv")
data


# Data Cleaning and Data Manipulation
data.isnull().sum()
data1 = data.dropna()
data1.dtypes()

data2 = pd.DataFrame(data1)
data2.dtypes

data2['Week Start Date'] = pd.to_datetime(data2['Week Start Date'], format='%d-%m-%Y')
data3 = pd.DataFrame(data2)
data3.dtypes

data3 = data3.rename(columns = {'Week Start Date':'Date','Sum of sales':'Sales'}).set_index('Date')
data3

# Visualizing the data distribution
data3.plot(figsize = (12,8))

# Stationarity test (Seasonality is visible already just by observation of the plot)
# ADF test
import statsmodels.tsa.stattools as sts
sts.adfuller(data3['Sales'])

# ADF indicates that the data is not stationary since the p-value is not less than or equal to 0.02 (and/or the test statistic is greater than the critical value). We will make it stationary by doing seasonal differencing (also due to the seasonality in the data)

## seasonal_period = 52 (Since we are dealing with weekly data)
data3['Sales SD'] = data3['Sales'].diff(periods = seasonal_period)
data3

# Plotting the differenced series and dropping NAs
data3['Sales SD'].plot()
data4 = data3.dropna()
data4

# Second Stationarity test
sts.adfuller(data4['Sales SD'])

# It is still not stationary. Let us try a different way (taking first-order difference)
data3['Sales FD'] = data3['Sales'].diff()
data3

# plotting the new first difference series and dropping NAs
data3.plot(subplots= True)
data4 = data3.drop(columns = ['Sales SD']).dropna()
data4

# Third Stationarity check
# First-order differencing could be enough for stationarity based on observation of the plot. 
sts.adfuller(data4['Sales FD'])
# YES, ADF indicates stationarity!


# Model Building
# Identifying ACF and PACF lags
import statsmodels.graphics.tsaplots as  sgt
ACF = sgt.plot_acf(data4['Sales FD'], lags = 40)
PACF = sgt.plot_pacf(data4['Sales FD'], lags = 40)
# Based on the PACF and ACF results, we will try with (1,1,1)


# Splitting the data for a training set and a testing set
len(data4)
training = data4[:116]
testing = data4[116:]

# Trying ARIMA Model
from statsmodels.tsa.arima.model import ARIMA
#order = (1,1,1)
Model1 = ARIMA(training['Sales FD'], order = (1,1,1))
Result1 = Model1.fit()
Result1.summary()

# Further examination of the goodness of fit using diagnostics plots
plot = Result1.plot_diagnostics()


# According to the plots of the diagnostics and the Ljung-Box results, the model shows goodness of fit. 
# The standardized residual plot has very little to no observable trend/pattern. 
# The histogram plus estimated density plot of residuals shows that the residuals are relatively normally distributed. 
# The Q-Q plot shows that there is normality of residuals and the correlogram shows that there is no autocorrelation of residuals as they are all not significantly different from zero. 
# The p-value of the Ljung-Box is greater than an alpha of 0.02, which means that there is no autocorrelation in the residual.


# We now predict the testing dataset to see how much of a good fit the model is
start_index = len(training)
end_index = len(training)+len(testing)-1
prediction = Result1.predict(start = start_index, end = end_index, dynamic = True)
training['Sales FD'].plot(legend = True, label = 'Training')
testing['Sales FD'].plot(legend = True, label = 'Testing')
prediction.plot(legend = True, label = 'Forecasting')
# It looks like the model is a fairly good fit for stationary data, but since it is ARIMA, it does not take into account the seasonal variation in the data. If we used the original sales rather than the differenced sales, the plot would look different due to the seasonal effect not being captured by the ARIMA model. Therefore, we should use seasonal ARIMA instead


# Forecasting future sales with SARIMAX
import statsmodels.api as sms
Fin_Model = sms.tsa.statespace.SARIMAX(data4['Sales'], order=(1,1,1), seasonal_order = (1,1,1,52))
Result2 = Fin_Model.fit()
Result2.summary()

# Examining the goodness of fit of the model and comparing it with the simpler ARIMA model
Diagnostics_check = Result2.plot_diagnostics()
# Based on the AIC and BIC, the Seasonal ARIMA model is better than the ARIMA model for this data. The diagnostics plots also show that the SARIMAX model is a better-fit model than the ARIMA model for this particular data

# Plotting the future sales forecast
forecast = Result2.predict(start = len(data4)-3, end = len(data4)+52)
data4['Sales'].plot(legend = True, label = 'Historical Sales')
forecast.plot(legend = True, label = 'Future Sales Forecast')

