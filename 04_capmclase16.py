# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:30:54 2024

@author: Usuario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib

# import or own files and reload 
import market_dataclase12
importlib. reload(market_dataclase12)

benchmark = '^SPX' # x 
security = 'XLK' # y


#  initialise 
capm = market_dataclase12.capm(benchmark, security)
capm.synchronise_timeseries()
capm.plot_timeseries()
capm.compute_linear_regression()
capm.plot_linear_regression()




