# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib

# import or own files and reload 
import random_variables
importlib.reload(random_variables)


# inputs
coeff = 5 # df in student, scale in exponential
size = 10**6
random_variable_type = 'student' 
# options : normal student uniform exponential chi-squared
decimals = 5

sim = random_variables.simulator(coeff, random_variable_type)
sim.generate_vector()
x = sim.vector

# task 1
# compute the rv vector
# DONE: sim.generate.vector()

# task 2  
# compute stats on the rv vector 
#  including the Jarque Bera test

# task 3 
# plot


# code
str_tittle = random_variable_type
if random_variable_type == 'normal':
    x = np.random.standard_normal(size=10**6)
elif random_variable_type == 'student':
    x = np.random.standard_t(df=coeff, size=size)
    str_tittle = str_tittle + 'df=' + str(coeff)
elif random_variable_type == 'uniform': 
    x = np.random.uniform(size=size)
elif random_variable_type == 'exponential': 
    x = np.random.exponential(scale=coeff, size=size)
    str_tittle = str_tittle + 'scale=' + str(coeff)
elif random_variable_type == 'chi-squared':
    x = np.random.chisquare(df=coeff, size=size)
    str_tittle = str_tittle + 'df=' + str(coeff)
    

# Test de normalidad: Jarque-Bera
mu = st.tmean(x)
sigma = st.tstd(x)
skewness = st.skew(x)
kurtosis = st.kurtosis(x)
jb_stat = size/6 * (skewness**2 + 1/4*kurtosis**2)
p_value = 1 - st.chi2.cdf(jb_stat, df=2)
is_normal = (p_value > 0.05) # equivalently jb < 6

# plot
str_tittle = random_variable_type
str_tittle += '\n' + 'mean=' + str(np.round(mu, decimals)) \
   + ' | ' + 'volatility=' + str(np.round(sigma, decimals)) \
   + '\n' + 'skewness=' + str(np.round(skewness, decimals)) \
   + ' | ' + 'kurtosis=' + str(np.round(kurtosis, decimals)) \
   + '\n' + 'JB stat=' + str(np.round(jb_stat,decimals)) \
   + ' | ' + 'p_value=' + str(np.round(p_value, decimals)) \
   + '\n' + 'is_normal=' + str(is_normal)  
plt.figure()
plt.hist(x,bins=100)
plt.title(str_tittle)
plt.show()



###############################
# loop of Jarque-bera normality test
###############################

n = 0
is_normal = True
str_tittle = 'normal'

while is_normal and n < 500:
    x = np.random.standard_normal(size=10**6)
    mu = st.tmean(x) # tmean
    sigma = st.tstd(x) #tstd
    skewness = st.skew(x)
    kurtosis = st.kurtosis(x)
    jb_stat = size/6 * (skewness**2 + 1/4*kurtosis**2)
    p_value = 1 - st.chi2.cdf(jb_stat, df=2)
    is_normal = (p_value > 0.05) # equivalently jb < 6
    print('n=' + str(n) +'| is_normal=' + str(is_normal))
    n += 1

str_tittle += '\n' + 'mean=' + str(np.round(mu, decimals)) \
   + ' | ' + 'volatility=' + str(np.round(sigma, decimals)) \
   + '\n' + 'skewness=' + str(np.round(skewness, decimals)) \
   + ' | ' + 'kurtosis=' + str(np.round(kurtosis, decimals)) \
   + '\n' + 'JB stat=' + str(np.round(jb_stat,decimals)) \
   + ' | ' + 'p_value=' + str(np.round(p_value, decimals)) \
   + '\n' + 'is_normal=' + str(is_normal)  


# plot
plt.figure()
plt.hist(x,bins=100)
plt.title(str_tittle)
plt.show()
