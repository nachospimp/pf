# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:06:27 2024

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

class capm:

    # constructor
    def __init__(self, benchmark, security, decimals= 5):
        self.benchmark = benchmark
        self.security = security
        self.timeseries = None 
    
    def synchronise_timeseries(self):
        self.timeseries = market_dataclase12.synchronise_timeseries(self.benchmark, self.security)
        
    def plot_timeseries(self): 
        plt.figure(figsize=(12,5))
        plt.title('Time series of prices')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        ax = plt.gca()
        ax1 = self.timeseries.plot(kind='line', x='date', y='close_x', ax=ax, grid=True,\
                                  color='blue', label=self.benchmark)
        ax2 = self.timeseries.plot(kind='line', x='date', y='close_y', ax=ax, grid=True,\
                                  color='red', secondary_y=True, label=self.security)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()