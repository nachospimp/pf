# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib

# import or own files and reload 
import random_variables1
importlib.reload(random_variables1)


# inputs
inputs = random_variables1.simulation_input()
inputs.df = 666 # degrees_of_freedom or df in student and chi-squared
inputs.scale = 17 # scale in exponential
inputs.mean = 5 # mean in normal
inputs.std = 10 # standard deviation or std in normal
inputs.size = 10**6
inputs.rv_type = 'normal' 
# options : standard_normal normal student uniform exponential chi-squared
inputs.decimals = 5

# computations
sim = random_variables1.simulator(inputs)
sim.generate_vector()
sim.compute_stats()
sim.plot()

