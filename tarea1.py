# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:21:00 2024

@author: Usuario
"""

import numpy as np
import scipy.stats as st

def test_jarque_bera(x):
    # start your code
    skewness = st.skew(x)
    kurtosis = st.kurtosis(x)
    size =len(x)
    jb_stat = size/6 * (skewness**2 + 1/4*kurtosis**2)
    p_value = 1 - st.chi2.cdf(jb_stat, df=2)
    is_normal = (p_value > 0.05) # equivalently jb < 6(self.vector)
    
    # end of code 
    return jb_stat, p_value, is_normal

# unitary test 
np.random.seed(seed=6)
x1 = np.random.standard_normal(size=10**6)
np.random.seed(seed=7)
x2 = np.random.standard_normal(size=10**6)



    