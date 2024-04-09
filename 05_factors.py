# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:48:36 2024

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

security = 'AAPL' 
factors = ['^SPX', 'IVW', 'IVE', 'QUAL', 'MTUM', 'SIZE', 'USMV']
 
# compute correlations 
df = capm16.dataframe_factors(security, factors)

