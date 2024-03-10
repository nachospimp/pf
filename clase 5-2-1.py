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
coeff = 5 # df in student, scale in exponential
size = 10**6
random_variable_type = 'normal' 
# options : normal student uniform exponential chi-squared
decimals = 5

sim = random_variables1.simulator(coeff, random_variable_type)
sim.generate_vector()
sim.compute_stats()
sim.plot()

