import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns




def plot_residuals(y, yhat):
    plt.scatter(y, yhat)

    plt.xlabel('x = actual value')
    plt.ylabel('y = predicted value')
    plt.title('Residuals')
    plt.show()
    return

def regression_errors(y, yhat):
    
    df = pd.DataFrame(y, yhat)
    
    baseline = y.mean()
    
    df['baseline'] = baseline
    
    
    MSE = mean_squared_error(y, yhat)
    
    SSE = MSE * len(df)
    
    RMSE = mean_squared_error(y, yhat, squared=False)
    
    TSS = (mean_squared_error(y, df.baseline) *len(df))
    
    ESS = TSS - SSE
    
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    
    
    baseline = y.mean()
    
    df = pd.DataFrame(y)
    
    df['baseline'] = baseline
    
    MSE_baseline = mean_squared_error(y, df.baseline)
    
    SSE_baseline = MSE_baseline * len(df)
    
    RMSE_baseline = mean_squared_error(y, df.baseline, squared=False)
    
    return SSE_baseline, MSE_baseline, RMSE_baseline

def better_than_baseline(y, yhat):
    
    baseline = y.mean()
    
    df = pd.DataFrame(y)
    
    df['baseline'] = baseline
    
    r2 = r2_score(y, yhat)
    
    r2_baseline = r2_score(y, df.baseline)
    
    if r2 > r2_baseline:
        
        return True
    else:
        return False

