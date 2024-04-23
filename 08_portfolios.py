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
            'BTC-USD',\
            'VFH', 'VHT', 'VGT', 'VTI', 'VOO']
rics = random.sample(universe, 5)


# initialise the instance of the class 
port_mgr = portfolio.manager(rics, notional)
 
# compute correlation and variance-covariance matrix 
port_mgr.compute_covariance()

# compute the desired portfolio : output class = portfolio.output
port_min_variance_l1 = port_mgr.compute_portfolio('min_variance_l1')
port_min_variance_l2 = port_mgr.compute_portfolio('min_variance_l2')
port_equi_weight = port_mgr.compute_portfolio('equi_weight')
# port_volatility_wwighted = port_mgr.compute_potfolio('volatility_weighted')