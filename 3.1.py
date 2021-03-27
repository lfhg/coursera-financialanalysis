#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:22:18 2021

@author: hatus
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
%matplotlib inline
from pandas.tools.plotting import scatter_matrix

data = pandas.DataFrame()

data['Population'] = [47, 48, 85, 20, 19, 13, 72, 16, 50, 60]

sample_without_replacement = data['Population'].sample(5, replace=False)

print(sample_without_replacement)
print(data['Population'].mean())
print(data['Population'].var(ddof=0)) #para sample, statistic tiene que ser con ddof=1 (N-1) (degree of freedom)
print(data['Population'].std(ddof=0))
print(data['Population'].shape[0])

# Sample mean and SD keep changing, but always within a certain range
Fstsample = pd.DataFrame(np.random.normal(10, 5, size=30))
print('sample mean is ', Fstsample[0].mean())
print('sample SD is ', Fstsample[0].std(ddof=1))


ms['logReturn'] = np.log(ms['Close'].shift(-1)) - np.log(ms['Close'])
# Lets build 90% confidence interval for log return
sample_size = ms['logReturn'].shape[0]
sample_mean = ms['logReturn'].mean()
sample_std = ms['logReturn'].std(ddof=1) / sample_size**0.5

# left and right quantile
z_left = norm.ppf(0.05)
z_right = norm.ppf(0.95)

# upper and lower bound
interval_left = sample_mean+sample_std*z_left
interval_right = sample_mean+sample_std*z_right


#null hypothesis, 2 tail
xbar = aapl['logReturn'].mean()
s = aapl['logReturn'].std(ddof=1)
n = aapl['logReturn'].shape[0]
zhat = (xbar-0)/(s/(n**0.5)) #null dice que mu=0. Si el resultado (zhat) es muy distinto que 0,
# rechazamos la null
alpha = 0.05
zleft=norm.ppf(alpha/2)
zright = -zleft
#reject: zhat>zright o zhat < zleft (alpha/2)

#null hypo, 1 tail: mu <= 0
#reject: zhat > zalpha

#p value: rechazo si p < zalfa
p = 2*(1-(norm.cdf(abs(zhat), 0, 1)))
#1 tail
p = 1-norm.cdf(z,0,1) # para mu>0
p = norm.cdf(z,0,1) # para mu < 0


#sm = scatter_matrix(housing, figsize=(10,10))