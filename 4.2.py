#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:35:14 2021

@author: hatus
"""

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import scipy.stats as stats
#% matplotlib inline

housing = pd.read_csv('./housing.csv')
housing.head()

model = smf.ols(formula = 'MEDV~LSTAT', data=housing).fit()

b0 = model.params[0]
b1 = model.params[1]
housing['BestResponse'] = b0 + b1*housing['RM']

housing['error'] = housing['MEDV'] - housing['BestResponse']

# plot your estimated line together with the points
plt.figure(figsize=(10, 10))
# See if the error drops after you use least square method
plt.title('Sum of sqaured error is {}'.format((((housing['error'])**2)).sum()))
plt.scatter(housing['RM'], housing['MEDV'], color='g', label='Observed')
#plt.plot(housing['RM'], housing['GuessResponse'], color='red', label='GuessResponse')
plt.plot(housing['RM'], housing['BestResponse'], color='yellow', label='BestResponse')
plt.legend()
plt.xlim(housing['RM'].min()-2, housing['RM'].max()+2)
plt.ylim(housing['MEDV'].min()-2, housing['MEDV'].max()+2)
plt.show()

model.summary()
#Durbin-Watson: 1.5 pos. correlated < Normal >2.5 neg. correlated #indep

#F-test: < 0.05 rechaza null, modelo es util

#Independence
plt.figure(figsize=(15, 8))
plt.title('Residual vs order')
plt.plot(housing.index, housing['error'], color='purple')
plt.axhline(y=0, color='red')
plt.show()

#Normality
z = (housing['error'] - housing['error'].mean())/housing['error'].std(ddof=1)
stats.probplot(z, dist='norm', plot=plt)
plt.title('Normal Q-Q plot')
plt.show()

#Equal variance
housing.plot(kind='scatter', x='RM', y='error', figsize=(15, 8), color='green')
plt.title('Residuals vs predictor')
plt.axhline(y=0, color='red')
plt.show()

