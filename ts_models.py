import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from dateutil.relativedelta import relativedelta 
from scipy.optimize import minimize              

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product 
from tqdm import tqdm_notebook

def weighted_average(series, weights):
    """
        Calculate weighter average on series
    """
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series.iloc[-n-1] * weights[n]
    return float(result)


def func(X_tn, w_t):
	T = X_tn.shape[0]
	N = X_tn.shape[1]
	num_tn = (X_tn * w_t[:, None].flip(axis=0)).cumsum(axis=0)



