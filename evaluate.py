import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns



def x_train_feats(df, y):
    
    X_train = df
    
    lm = LinearRegression()
    
    model = lm.fit(X_train, y)
    
    return X_train, model


def plot_residuals(y, yhat):
    
    residuals = y - yhat
    
    plt.scatter(y, residuals)

    plt.xlabel('x = Home Value')
    plt.ylabel('y = Residuals')
    plt.title('Residuals vs Home Value')
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
    
    
    baseline = np.repeat(y.mean(), len(y))
    
    
    MSE_baseline = mean_squared_error(y, baseline)
    
    SSE_baseline = MSE_baseline * len(y)
    
    RMSE_baseline = mean_squared_error(y, baseline, squared=False)
    
    return SSE_baseline, MSE_baseline, RMSE_baseline

def better_than_baseline(y, yhat):
    
    baseline = np.repeat(y.mean(), len(y))
    
    r2 = r2_score(y, yhat)
    
    r2_baseline = r2_score(y, baseline)
    
    if r2 > r2_baseline:
        
        return True
    else:
        return False
    



