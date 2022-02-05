import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import pickle


# ----------------------------------------------------------------------------------
# define function for calculating mean and 95% CI
# ----------------------------------------------------------------------------------
def mean_CI(data):

    mean = np.mean(np.array(data))
    CI = ss.t.interval(
        alpha=0.95,
        df=len(data)-1,
        loc=np.mean(data),
        scale=ss.sem(data)
        )
    lower = CI[0]
    upper = CI[1]

    return mean, lower, upper

##def mean_CI(metric):
##    alpha  = 0.95
##    mean   = np.mean(np.array(metric))
##    p_up   = (1.0 - alpha)/2.0*100
##    lower  = max(0.0, np.percentile(metric, p_up))
##    p_down = ((alpha + (1.0 - alpha)/2.0)*100)
##    upper  = min(1.0, np.percentile(metric, p_down))
##    return mean, lower, upper

##def mean_CI(stat, confidence=0.95):
##    alpha  = 0.95
##    mean   = np.mean(np.array(stat))
##    p_up   = (1.0 - alpha)/2.0*100
##    lower  = max(0.0, np.percentile(stat, p_up))
##    p_down = ((alpha + (1.0 - alpha)/2.0)*100)
##    upper  = min(1.0, np.percentile(stat, p_down))
##    return mean, lower, upper
