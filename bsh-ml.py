#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn import metrics
import scipy.stats as stats

import statsmodels.api as sm
import numpy as np

from sklearn import preprocessing


# In[24]:


df = pd.read_csv('../Downloads/bike-sharing-hour.csv', index_col=0)
df


# In[8]:


df.columns


# In[25]:


df = pd.read_csv('../Downloads/bike-sharing-hour.csv', index_col=0)
df = df.drop('dteday', axis=1)
independent_variables = df.drop('cnt', axis=1)
x = independent_variables.values
y = df['cnt'].values

lr = LinearRegression(fit_intercept = True)
lr.fit(x, y)
y_pred = lr.predict(x)


# In[26]:


print('Coefficients = ', lr.coef_)


# In[27]:


print('Intercept = ', lr.intercept_)


# In[28]:


print('R^2 = ', lr.score(x, y))


# In[29]:


print('Root MSE = ', math.sqrt(metrics.mean_squared_error(y, y_pred)))


# In[30]:


x = independent_variables
y = df['cnt']

x2 = sm.add_constant(x)
ols = sm.OLS(y, x2)
est = ols.fit()

est.summary()


# In[ ]:


#The model after normalisation has an R^2 value of 1.000
#F-Statistic is 2.149e+31
#RMSE = 1.683500083800179e-13

