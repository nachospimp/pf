# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:30:32 2024

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
import portfolio
importlib.reload(portfolio)

# inputs
notional = 15 # in mn USD 
universe = ['^SPX', '^IXIC', '^MXX', '^STOXX', '^GDAXI', '^FCHI', '^VIX',\
            'XLK', 'XLF', 'XLV', 'XLP', 'XLY', 'XLE', 'XLI', 'XLC', 'XLU',\
            'SPY', 'EWW',\
            'IVW', 'IVE' , 'QUAL', 'MTUM' , 'SIZE', 'USMV',\
            'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'NFLX',\
            'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'MS', 'GS', 'BLK',\
            'BTC-USD', 'CX', 'UBER', 'CEMEXCPO.MX'\
            'VFH', 'VHT', 'VGT', 'VTI', 'VOO']
    
# rics = ["NFLX","UBER","CEMEXCPO.MX"]
rics = random.sample(universe, 5)

print(rics)
# initialise the instance of the class 
port_mgr = portfolio.manager(rics, notional)
 
# compute correlation and variance-covariance matrix 
port_mgr.compute_covariance()

# compute the desired portfolio : output class = portfolio.output
port_min_variance_l1 = port_mgr.compute_portfolio('min_variance_l1')
port_min_variance_l2 = port_mgr.compute_portfolio('min_variance_l2')
port_equi_weight = port_mgr.compute_portfolio('equi_weight')
port_long_only = port_mgr.compute_portfolio('long_only')
port_markowitz = port_mgr.compute_portfolio('markowitz', target_return=0.25)
# plot histogramas of retutns for the desired portfolio
port_min_variance_l1.plot_histogram()
port_min_variance_l2.plot_histogram()
port_equi_weight.plot_histogram()
port_long_only.plot_histogram()
port_markowitz.plot_histogram()

# return_target = port_markowitz.target_return
# return_portfolio_long_only = np.round(port_mgr.returns.dot(port_long_only.weights), 6)
# return_portfolio_equi_weight = np.round(port_mgr.returns.dot(port_equi_weight.weights), 6)
# return_portfolio_markowitz = np.round(port_mgr.returns.dot(port_markowitz.weights), 6)



# df = pd.DataFrame()
# df['rics'] = rics 
# df['returns'] = port_mgr.returns
# df['volatilities'] = port_mgr.volatilities
# df['markowitz_weights'] = port_markowitz.weights
# df['markowitz_allocation'] = port_markowitz.allocation
# df['min_variance_weights'] = port_min_variance_l1.weights
# df['min_variance_allocation'] = port_min_variance_l1.allocation
# df['equi_weight_weights'] = port_equi_weight.weights
# df['equi_weight_allocation'] = port_equi_weight.allocation
# df['long_only_weights'] = port_long_only.weights
# df['long_only_allocation'] = port_long_only.allocation
