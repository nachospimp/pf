# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:24:01 2024

@author: Usuario
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as op 
import importlib
import random 



import market_dataclase12
importlib.reload(market_dataclase12)
import capm16 
importlib.reload(capm16)


# create offline instances of pca_model 
notional = 10 # in mn USD 
universe = ['^SPX', '^IXIC', '^MXX', '^STOXX', '^GDAXI', '^FCHI', '^VIX',\
            'XLK', 'XLF', 'XLV', 'XLP', 'XLY', 'XLE', 'XLI', 'XLC', 'XLU',\
            'SPY', 'EWW',\
            'IVW', 'IVE' , 'QUAL', 'MTUM' , 'SIZE', 'USMV',\
            'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'NFLX',\
            'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'MS', 'GS', 'BLK',\
            'BTC-USD',\
            'VFH', 'VHT', 'VGI', 'VTI', 'VOO']
rics = random.sample(universe, 5)


# rics = ['^MXX', '^SPX', 'XLK', 'XLF', 'XLV', 'XLP', 'XLY', 'XLE', 'XLI']
# rics = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'NFLX',\
#         'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'MS', 'GS', 'BLK']
# rics = ['^SPX', 'IVW', 'IVE' , 'QUAL', 'MTUM' , 'SIZE', 'USMV',\
#         'XLK', 'XLF', 'XLV', 'XLP', 'XLY', 'XLE', 'XLI', 'XLC', 'XLU']

# 0-7 / 8 - 15 
    
# # synchronise all the timeseries returns 
df = market_dataclase12.synchronise_returns(rics)
    
# compute the variance-covariance matrix and correlation matrices
mtx = df.drop(columns=['date'])
mtx_var_covar = np.cov(mtx, rowvar=False) * 252
mtx_correl = np.corrcoef(mtx, rowvar=False)

# min-var with eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(mtx_var_covar)
min_var_vector = eigenvectors[:,0]

# unit test for variance function 
variance_1 = np.matmul(np.transpose(min_var_vector), np.matmul(mtx_var_covar, min_var_vector))

#####################################
# min-var with scipy optimize minimize
#####################################

np.matmul(eigenvectors, np.transpose(eigenvectors))

# initial condition 
def portfolio_variance(x, mtx_var_covar):
    variance = np.matmul(np.transpose(x), np.matmul(mtx_var_covar, x))
    return variance 

# compute optimisation 
x0 = [notional / len(rics)] * len(rics)
l2_norm = [{"type": "eq", "fun": lambda x: sum(x**2) - 1}] # unitary in norm L2
l1_norm = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}] # unitary in norm L1
optimal_result = op.minimize(fun=portfolio_variance, x0=x0,\
                              args=(mtx_var_covar),\
                              constraints=l2_norm)
optimize_vector = optimal_result.x
variance_2 = optimal_result.fun

df_weights = pd.DataFrame()
df_weights['rics'] = rics
df_weights['min_var_vector'] = min_var_vector
df_weights['optimize_vector'] = optimize_vector

