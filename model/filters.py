#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:28:14 2019

@author: erigara
"""


import pandas as pd

def groupsize_filter(dataframe, id_column, min_groupsize=0):
    """
    Remove rows with id_column such that it has < min_groupzize rows in dataframe
    
    dataframe : pd.DataFrame
        filtering dataframe
    id_column : str
        name of groupby column
    
    min_groupsize : int
        smallest possible groupsize that stay in dataframe
    """
    df_cnt = pd.DataFrame(
        dataframe.groupby(id_column).size(),
        columns=['count'])
    ids = list(set(df_cnt.query('count >= @min_groupsize').index))
    filtred_dataframe = dataframe[dataframe[id_column].isin(ids)]
    return filtred_dataframe