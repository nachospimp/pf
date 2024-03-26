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


#  inputs 
ric = '^MXX'


directory = 'C:\\Users\\Usuario\\.spyder-py3\\2024\\Data\\'
path = directory + ric + '.csv'
raw_data = pd.read_csv(path)
t = pd.DataFrame()
t['date']= pd.to_datetime(raw_data['Date'], yearfirst=True)
t['close'] = raw_data['Close'] 
t.sort_values(by='date', ascending=True)
t['close_previous'] = t['close'].shift(1)
t['return_close'] = t['close']/t['close_previous'] - 1 
t = t.dropna()
t = t.reset_index (drop=True)

# inputs
inputs = random_variables1.simulation_input()
inputs.rv_type = ric + ' | real data' 
# options : standard_normal normal student uniform exponential chi-squared
inputs.decimals = 5

# computations
sim = random_variables1.simulator(inputs)
sim.vector = t['return_close'].values
sim.inputs.size = len(sim.vector)
sim.str_tittle = sim.inputs.rv_type
sim.compute_stats()
sim.plot()

plt.figure()
t.plot(kind='line', x='date', y='close', grid=True, color='blue',\
       title='Time series of close prices for'+ric)
plt.show()

rics = []
is_normals = []
for file_name in os.listdir(directory):
    print('file_name = ' + file_name)
    ric = file_name.split('.')[0]
    if ric== 'ReadMe':
        continue
    #get dataframe
    path = directory + ric + '.csv'
    raw_data = pd.read_csv(path)
    t = pd.DataFrame()
    t['date']= pd.to_datetime(raw_data['Date'], yearfirst=True)
    t['close'] = raw_data['Close'] 
    t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1)
    t['return_close'] = t['close']/t['close_previous'] - 1 
    t = t.dropna()
    t = t.reset_index (drop=True)
    # computations
    sim = random_variables1.simulator(inputs)
    sim.vector = t['return_close'].values
    sim.inputs.size = len(sim.vector)
    sim.str_tittle = sim.inputs.rv_type
    sim.compute_stats()
    # generate list
    rics.append(ric)
    is_normals.append(sim.is_normal)
df = pd.DataFrame()
df['ric'] = rics
df['is_normals'] = is_normals

