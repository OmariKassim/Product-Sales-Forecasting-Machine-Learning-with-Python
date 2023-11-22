#!/usr/bin/env python
# coding: utf-8

# # ARIMA Weekly Sale Forecasting

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime


# In[2]:


data = pd.read_csv(r"C:\Users\omarm\Documents\Career Essentials & Job Apps\Portfolio\Weekly Sales Data.csv")
data


# In[3]:


data.isnull().sum()


# In[4]:


data1 = data.dropna()


# In[5]:


data1


# In[6]:


data1.dtypes()


# In[7]:


data2 = pd.DataFrame(data1)


# In[8]:


data2.dtypes


# In[22]:


data2['Week Start Date'] = pd.to_datetime(data2['Week Start Date'], format='%d-%m-%Y')


# In[31]:


data3 = pd.DataFrame(data2)


# In[33]:


data3.dtypes


# In[34]:


data3 = data3.rename(columns = {'Week Start Date':'Date','Sum of sales':'Sales'}).set_index('Date')
data3


# In[35]:


data3.plot(figsize = (12,8))


# #### Stationarity test
# seasonality is visible already just by observation

# In[38]:


# ADF test

import statsmodels.tsa.stattools as sts

sts.adfuller(data3['Sales'])


# #### ADF indicated that the data is not stationary since the p-value is not less than or equal to 0.02 (and/or the test statistic is greater than the critical value). We will make it stationary by doing seasonal differencing (also due to the seasonality in the data)

# In[40]:


#seasonal_period = 52
data3['Sales SD'] = data3['Sales'].diff(periods = seasonal_period)
data3


# In[41]:


data3['Sales SD'].plot()


# In[42]:


data4 = data3.dropna()
data4


# #### Second check for stationarity

# In[44]:


sts.adfuller(data4['Sales SD'])


# In[46]:


# It is still not stationary. Let us try a different way


# In[47]:


# Let us try taking the first difference, then we do the seasonal differencing


# In[48]:


data3['Sales FD'] = data3['Sales'].diff()


# In[49]:


data3


# In[51]:


data3.plot(subplots= True)


# In[56]:


data4 = data3.drop(columns = ['Sales SD']).dropna()
data4


# #### Stationarity check
# First order differencing could be enough for stationarity. 

# In[58]:


sts.adfuller(data4['Sales FD'])


# #### ADF indicates stationarity!

# ## Model Building
# #### Identifying ACF and PACF lags

# In[64]:


import statsmodels.graphics.tsaplots as  sgt

ACF = sgt.plot_acf(data4['Sales FD'], lags = 40)


# In[65]:


PACF = sgt.plot_pacf(data4['Sales FD'], lags = 40)


# In[74]:


# Based on the PACF and ACF results, we will try with (1,1,1)


# #### Splitting the data for a train set and a testing set

# In[75]:


len(data4)


# In[76]:


training = data4[:116]
testing = data4[116:]


# #### We apply SARIMA due to the seasonal effect in the data

# In[82]:


from statsmodels.tsa.arima.model import ARIMA


# In[99]:


#order = (1,1,1)
Model1 = ARIMA(training['Sales FD'], order = (1,1,1))
Result1 = Model1.fit()
Result1.summary()


# In[100]:


plot = Result1.plot_diagnostics()


# #### According to the plot of the diagnostics and the Ljung-Box results, the model shows a goodness of fit. 
# The standardizes residual plot has very little to no observable trend/pattern. 
# The histogram plus estimated density plot of residuals shows that the residuals are relatively normally distributed. 
# The Q-Q plot shows that there is normality of residuals and the correlogram shows that there is not autocorrelation of residuals as they are all not significantly different from zero. 
# The p-value of the Ljung-Box is greater than an alpha of 0.02, which means that there is no aurocorrelation in the residual.

# #### We now predict the testing dataset to see how much of a good fit the model is

# In[109]:


start_index = len(training)
end_index = len(training)+len(testing)-1
prediction = Result1.predict(start = start_index, end = end_index, dynamic = True)


# In[112]:


training['Sales FD'].plot(legend = True, label = 'Training')
testing['Sales FD'].plot(legend = True, label = 'Testing')
prediction.plot(legend = True, label = 'Forecasting')


# #### It looks like the model is a fairly good fit for stationary data, but it does not take into account the seasonal variation in the data. If we used the original sales rather than the differenced sales, the plot would look different due to the seasonal effect not being captured by the ARIMA model. Therefore, we should use seasonal ARIMA instead

# #### Forecasting future sales

# In[132]:


import statsmodels.api as sms
Fin_Model = sms.tsa.statespace.SARIMAX(data4['Sales'], order=(1,1,1), seasonal_order = (1,1,1,52))

Result2 = Fin_Model.fit()
Result2.summary()


# In[133]:


Diagnostics_check = Result2.plot_diagnostics()


# Based on the AIC and BIC, the SARIMA model is better than the ARIMA model for this data. Even the plots show that the SARIMAX is slightly of a model than ARIMA for this paricular data

# In[141]:


forecast = Result2.predict(start = len(data4)-3, end = len(data4)+52)


# In[142]:


data4['Sales'].plot(legend = True, label = 'Historical Sales')
forecast.plot(legend = True, label = 'Future Sales Forecast')


# In[ ]:




