# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:35:55 2024

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
import options
importlib.reload(options)

inputs = options.inputs()
inputs.price = 69593 # S
inputs.time = 0 # t 
inputs.maturity = 1 # T (3/12, tres meses) (1, un año)
inputs.strike = 75000 # K 
inputs.interest_rate = 0.0446 # r 
inputs.volatility = 0.461 # sigma 
inputs.type = 'call' 
inputs.monte_carlo_size = 10**6

option_mgr = options.manager(inputs)
option_mgr.compute_black_scholes_price()
option_mgr.compute_monte_carlo_price()
option_mgr.plot_histogram()


