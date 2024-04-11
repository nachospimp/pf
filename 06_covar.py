# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:06:10 2024

@author: Usuario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib

import market_dataclase12
importlib.reload(market_dataclase12)
import capm16 
importlib.reload(capm16)

# rics = ['^MXX', '^SPX', 'XLK', 'XLF', 'XLV', 'XLP', 'XLY', 'XLE', 'XLI']
# rics = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'NFLX',\
#         'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'MS', 'GS', 'BLK']
rics = ['^SPX', 'IVW', 'IVE' , 'QUAL', 'MTUM' , 'SIZE', 'USMV',\
        'XLK', 'XLF', 'XLV', 'XLP', 'XLY', 'XLE', 'XLI', 'XLC', 'XLU']

# 0-7 / 8 - 15 
    
# # synchronise all the timeseries returns 
df = market_dataclase12.synchronise_returns(rics)
    
# compute the variance-covariance matrix and correlation matrices
mtx = df.drop(columns=['date'])
mtx_var_covar = np.cov(mtx, rowvar=False) * 252
mtx_correl = np.corrcoef(mtx, rowvar=False)

# unitary test for correlation 
# correl = capm16.compute_correlation('XLP', 'XLY')

# compute eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(mtx_var_covar)
variance_explained = eigenvalues / np.sum(eigenvalues)
prod = np.matmul(eigenvectors, np.transpose(eigenvectors))

########################
# PCA for 2D visualisation
########################

#  compute min and max volatilities 
volatility_min = np.sqrt(eigenvalues[0])
volatility_max = np.sqrt(eigenvalues[-1])

# compute PCA base for 2D visualisation 
pca_vector_1 = eigenvectors[:,-1]
pca_vector_2 = eigenvectors[:,-2]
pca_eigenvalue_1 = eigenvalues[-1]
pca_eigenvalue_2 = eigenvalues[-2]
pca_variance_explained = variance_explained[-2:].sum()

# compute min variance portfolio 
min_var_vector = eigenvectors[:,0]
min_var_eigenvalue = eigenvalues[0]
min_var_variance_explained = variance_explained[0]


