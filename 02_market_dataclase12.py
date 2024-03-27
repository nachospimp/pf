# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:31:33 2024

@author: Usuario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib
import os

# import or own files and reload 
import random_variables1
importlib.reload(random_variables1)
import market_data
importlib. reload(market_data)


#  inputs 
directory = 'C:\\Users\\Usuario\\.spyder-py3\\2024\\Data\\' #Harcoded
ric = 'EWW'

# computations
dist = market_data.distribution(ric)
dist.load_timeseries()
dist.plot_timeseries()
dist.compute_stats()
dist.plot_histrogram()

# loop to check normality in real distributions 
rics = []
is_normals = []
for file_name in os.listdir(directory):
    print('file_name = ' + file_name)
    ric = file_name.split('.')[0]
    if ric== 'ReadMe':
        continue
    #compute stats
    dist = market_data.distribution(ric)
    dist.load_timeseries()
    dist.compute_stats()
    # generate list
    rics.append(ric)
    is_normals.append(dist.is_normal)
df = pd.DataFrame()
df['ric'] = rics
df['is_normal'] = is_normals
df = df.sort_values(by='is_normal', ascending=False)

