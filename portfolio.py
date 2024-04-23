# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:43:03 2024

@author: Usuario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as op 
import importlib
import random 


def portfolio_variance(x, mtx_var_covar):
    variance = np.matmul(np.transpose(x), np.matmul(mtx_var_covar, x))
    return variance 


import market_dataclase12
importlib.reload(market_dataclase12)
import capm16 
importlib.reload(capm16)

class manager:
    
    def __init__(self, rics, notional):
        self.rics = rics
        self.notional = notional 
        self.mtx_var_covar = None
        self.mtx_correl = None 
        
    def compute_covariance(self):
        df = market_dataclase12.synchronise_returns(self.rics)
        mtx = df.drop(columns=['date'])
        self.mtx_var_covar = np.cov(mtx, rowvar=False) * 252
        self.mtx_correl = np.corrcoef(mtx, rowvar=False)
        
    def compute_portfolio(self, portfolio_type='default'):
        
        # inputs, constraints and boundary conditions for the optimiser
        x0 = [self.notional / len(self.rics)] * len(self.rics)
        l1_norm = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}] # unitary in norm L1
        l2_norm = [{"type": "eq", "fun": lambda x: sum(x**2) - 1}] # unitary in norm L2
        
        # compute portafolios via optimiser 
        if portfolio_type == 'min_variance_l1':
            optimal_result = op.minimize(fun=portfolio_variance, x0=x0,\
                                      args=(self.mtx_var_covar),\
                                      constraints=(l1_norm))
            weights = optimal_result.x
        elif portfolio_type == 'min_variance_l2':
            optimal_result = op.minimize(fun=portfolio_variance, x0=x0,\
                                      args=(self.mtx_var_covar),\
                                      constraints=(l2_norm))
            weights = optimal_result.x
        else:
            weights = np.array(x0)
            
        # optimal porfolio
        optimal_portfolio = output(self.rics, self.notional)
        optimal_portfolio.type = portfolio_type
        optimal_portfolio.weights = self.notional * weights / sum(abs(weights))
        return optimal_portfolio
        
class output:
    
    def __init__(self, rics, notional):
        self.rics = rics
        self.notional = notional
        self.type = None 
        self.weights = None 
        pass
