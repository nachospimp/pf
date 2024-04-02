# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:05:01 2024

@author: Usuario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as op 
import importlib


# import or own files and reload 
import capm16
importlib. reload(capm16)

# inputs 
position_security = 'GOOG'
position_delta_usd = 10 # 10 mn USD
benchmark = '^SPX' 
hedge_securities = ['AAPL' , 'MSFT']

hedger = capm16.hedger(position_security, position_delta_usd, benchmark, hedge_securities)
hedger.compute_betas()
hedger.compute_hedge_weights()
hedge_weights_exact = hedger.hedge_weights


# # dedine the function to minimise 
# def cost_function(x):
#     f = (x[0] - 7.0)**2 + (x[1] + 5)**2 + (x[2] - 13)**2
#     return f 

# # initialiise optimisation 
# x0 = np.zeros([3,1])

# #  compute optimisation 
# optimal_result = op.minimize(fun=cost_function, x0=x0.flatten())

# # print
# print ('------')
# print('Óptimisation result:')
# print(optimal_result)

#  define the function to minimise 
betas = hedger.hedge_betas
target_delta = hedger.position_delta_usd
target_beta = hedger.position_beta_usd

def cost_function(x, betas, target_delta, target_beta):
    dimensions = len(x)
    deltas = np.ones([dimensions])
    f_delta = (np.transpose(deltas).dot(x).item()  + target_delta)**2
    f_beta  = (np.transpose(betas).dot(x).item()  + target_beta)**2
    f = f_delta + f_beta
    return f

# #  input parameters 
# dimensions = 5
# roots = np.random.randint(low=-20, high=20, size=dimensions)
# coeffs = np.ones([dimensions,1])


# initial condition 
x0 = - target_delta / len(betas) * np.ones([len(betas),1])

# compute optimisation 
optimal_result = op.minimize(fun=cost_function, x0=x0.flatten(),\
                             args=(betas,target_delta,target_beta))
hedge_weights_optimize = optimal_result.x

# print
print('------')
print('Optimisation result:')
print(optimal_result)
print('------')
