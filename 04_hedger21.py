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
position_security = 'NVDA'
position_delta_usd = 10 # 10 mn USD
benchmark = '^SPX' 
hedge_universe = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'NFLX', 'SPY', 'XLK', 'XLF']
regularisation = 0.05


# compute correlations 
df = capm16.dataframe_correl_beta(benchmark, position_security, hedge_universe)
# computations
hedge_securities = ['AAPL', 'MSFT']
hedger = capm16.hedger(position_security, position_delta_usd, benchmark, hedge_securities)
hedger.compute_betas()
hedger.compute_hedge_weights(regularisation)

# variables
hedge_weights = hedger.hedge_weights
hedge_delta_usd = hedger.hedge_delta_usd
hedge_beta_usd = hedger.hedge_beta_usd
