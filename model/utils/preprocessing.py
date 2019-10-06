#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module provide preprocessing functions
"""


import pandas as pd

def groupsize_filter(dataframe, column_name, min_groupsize=0):
    """
    Remove rows with id_column such that it has < min_groupzize rows in dataframe
    
    dataframe : pd.DataFrame
    
    column_name : str
        name of groupby column
    
    min_groupsize : int
        smallest possible groupsize that stay in dataframe
    """
    df_cnt = pd.DataFrame(
        dataframe.groupby(column_name).size(),
        columns=['count'])
    ids = list(set(df_cnt.query('count >= @min_groupsize').index))
    filtred_dataframe = dataframe[dataframe[column_name].isin(ids)]
    return filtred_dataframe

def train_test_split(dataframe):
    pass

