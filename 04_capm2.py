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
import capm16
importlib. reload(capm16)

benchmark = '^SPX' # x 
security = 'AAPL' # y


#  initialise 
model = capm16.model(benchmark, security)
model.synchronise_timeseries()
model.plot_timeseries()
model.compute_linear_regression()
model.plot_linear_regression()




