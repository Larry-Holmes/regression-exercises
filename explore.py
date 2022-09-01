# basic ds imports
import pandas as pd
import numpy as np
# viz imports
import seaborn as sns
import matplotlib.pyplot as plt

def plot_variable_pairs(train):
    sns.pairplot(data=train,
            corner=True,
            kind='reg')
    plt.show()

# def plot_categorical_and_continuous_vars(train, col_selection):
    # for 