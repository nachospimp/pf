# -*- coding: utf-8 -*-
"""
Created on Sat May 18 09:04:32 2024

@author: Usuario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as op 
import importlib
import random 

# import our own files and reload 
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
            'BTC-USD', 'CX', 'UBER', 'CEMEXCPO.MX',\
            'VFH', 'VHT', 'VGT', 'VTI', 'VOO']
    
# rics = random.sample(universe, 10)
rics = ["NFLX","UBER","CEMEXCPO.MX"]

# efficent frontier 
target_return = 0.35
include_min_variance = False
dict_portfolios = portfolio.compute_efficent_frontier(rics, notional, target_return, include_min_variance)
print(rics)

port = dict_portfolios['markowitz-target']
port.plot_histogram()