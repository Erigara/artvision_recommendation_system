#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module provide preprocessing functions
"""


import pandas as pd
import numpy as np

def groupsize_filter(df, column_name, min_groupsize=0):
    """
    Remove rows with id_column such that it has < min_groupzize rows in dataframe
    
    df : pd.DataFrame
    
    column_name : str
        name of groupby column
    
    min_groupsize : int
        smallest possible groupsize that stay in dataframe
    """
    df_cnt = pd.DataFrame(
        df.groupby(column_name).size(),
        columns=['count'])
    ids = list(pd.unique(df_cnt.query('count >= @min_groupsize').index))
    filtred_df = df.query(column_name+' in @ids')
    filtred_df = df[df[column_name].isin(ids)]
    return filtred_df

def get_train_test_split(df, ids_column_name, timestamp_column_name, train_coeff = 0.8):
    """
    Spliting dataframe into two datarames with fixed size. All data sorted by time and each
    train samle contain the same portion of each user's data

    df: pd.DataFrame
        dataframe needed to be split

    ids_column_name: str
        name of ids column

    timestamp_column_name: str
        name of users column

    train_coeff: int
        coefficient of train sample size

    return:
        train, test: pd.DataFrame
        dataframes which split by time. test dataframe contains elder user's ratings
    """
    cur_df = df.copy()
    cur_df.sort_values(by=[ids_column_name, timestamp_column_name], inplace=True)
    sizes = list(df.groupby(ids_column_name).size())
    train_sizes = [round(train_coeff * size) for size in sizes]
    train_indexer = np.array([True for i in range(sum(sizes))])
    # fill with False last (1-train_coeff) rows for every user_id
    current_size = 0
    for i in range(len(train_sizes)):
        train_indexer[current_size + train_sizes[i] : current_size + sizes[i]] = False
        current_size += sizes[i]
    test_indexer = ~train_indexer
    train, test = cur_df[train_indexer], cur_df[test_indexer]

    return train, test

