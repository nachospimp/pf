# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:35:58 2024

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

class inputs:
    
    def __init__(self):
        self.price = None # S
        self.time = None # t 
        self.maturity = None # T 
        self.strike = None # K 
        self.interest_rate = None # r 
        self.volatility = None # sigma
        self.type = None # call or put 
        self.monte_carlo_size = None # no of simulations
        

class manager:
    
    def __init__(self, inputs):
        self.price = inputs.price # S
        self.time = inputs.time # t 
        self.maturity = inputs.maturity # T 
        self.time_to_maturity = inputs.maturity - inputs.time # T -t 
        self.strike = inputs.strike # K 
        self.interest_rate = inputs.interest_rate # r 
        self.volatility = inputs.volatility # sigma
        self.type = inputs.type # call or put 
        self.black_scholes_price = None
        self.monte_carlo_size = inputs.monte_carlo_size # no of simulations
        self.monte_carlo_price = None 
        self.monte_carlo_simulations = None 
        self.monte_carlo_confidence_interval = None 
        
    def compute_black_scholes_price(self):
        d1 = 1 / (self.volatility * np.sqrt(self.time_to_maturity)) \
            * (np.log(self.price / self.strike) \
            + (self.interest_rate + 0.5 * (self.volatility**2)) * self.time_to_maturity)
        d2 = d1 - self.volatility * np.sqrt(self.time_to_maturity)
        if self.type == 'call':
            price_option = self.price * st.norm.cdf(d1)\
                - self.strike * np.exp(-(self.interest_rate * self.time_to_maturity)) * st.norm.cdf(d2)
        elif self.type == 'put':
            price_option = self.strike * np.exp(-(self.interest_rate * self.time_to_maturity)) * st.norm.cdf(-d2)\
                - self.price * st.norm.cdf(-d1)
        self.black_scholes_price = price_option
        
    def compute_monte_carlo_price(self):
        N = np.random.standard_normal(self.monte_carlo_size) 
        price_underlying = self.price\
            * np.exp((self.interest_rate - 0.5 * (self.volatility**2)) * self.time_to_maturity \
            + self.volatility * np.sqrt(self.time_to_maturity) * N)
        self.monte_carlo_simulations = np.exp(-(self.interest_rate * self.time_to_maturity))\
            * np.array([max(s - self.strike, 0.0) for s in price_underlying])
        self.monte_carlo_price = np.mean(self.monte_carlo_simulations)
        self.monte_carlo_confidence_interval = self.monte_carlo_price + np.array([-1, +1])\
            * 1.96 * np.std(self.monte_carlo_simulations) / np.sqrt(self.monte_carlo_size)
            
    def plot_histogram(self):
        plt.figure()
        x = self.monte_carlo_simulations
        plt.hist(x, bins=100)
        plt.title('Monte Carlo Simulations')
        plt.show()
        
        
        
