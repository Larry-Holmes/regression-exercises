# basic ds imports
import pandas as pd
import numpy as np
# viz imports
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def plot_variable_pairs(df):
    samp_df = df.sample(frac=.083)
    sns.pairplot(data=samp_df,
            corner=True,
            kind='reg')
    plt.show()

def dtypes_to_list(df):
    num_type_list , cat_type_list = [], []

    for column in df:
        col_type = df[column].dtype
        if col_type == 'object':
            cat_type_list.append(column)
        if np.issubdtype(df[column], np.number) and \
             ((df[column].max() + 1) / df[column].nunique())  == 1 :
            cat_type_list.append(column)
        if np.issubdtype(df[column], np.number) and \
            ((df[column].max() + 1) / df[column].nunique()) != 1 :
            num_type_list.append(column)
    return num_type_list, cat_type_list  

def plot_categorical_and_continuous_vars(df, num_type_list, cat_type_list):
    
    for column in df:
        if column.isin(num_type_list):
            sns.lmplot(data=df, x=column, y=df.tax_value)
            plt.title(f'{column}')
            plt.xlabel(f'{column}')
            plt.ylabel('y = tax value')
        elif column.isin(cat_type_list):
            sns.barplot(data=df, x=column, y=df.tax_value)
            plt.title(f'{column}')
            plt.xlabel(f'{column}')
            plt.ylabel('y = tax value')