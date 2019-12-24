#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module provide preprocessing functions
"""


import pandas as pd
import numpy as np
from data_loaders import RatingData

def groupsize_filter(rating_data, by, min_groupsize=0):
    """
    Remove rows with id_column such that it has < min_groupzize rows in dataframe
    
    rating_data : RatingData
    
    by : str
        name of groupby column
    
    min_groupsize : int
        smallest possible groupsize that stay in dataframe
    """
    df_cnt = pd.DataFrame(
        rating_data.df.groupby(by).size(),
        columns=['count'])
    ids = list(pd.unique(df_cnt.query('count >= @min_groupsize').index))
    filtred_df = rating_data.df.query(by+' in @ids')
    return RatingData(filtred_df, *rating_data[1:])

def get_train_test_split(rating_data, by, train_coeff = 0.8):
    """
    Spliting dataframe into two datarames with fixed size. All data sorted by time and each
    train samle contain the same portion of each user's data

    df: pd.DataFrame
        dataframe needed to be split

    by: str
        name of ids column

    train_coeff: int
        coefficient of train sample size

    return:
        train, test: pd.DataFrame
        dataframes which split by time. test dataframe contains elder user's ratings
    """
    cur_df = rating_data.df.copy()
    cur_df.sort_values(by=[by, rating_data.timestamp_col_name], inplace=True)
    sizes = list(rating_data.df.groupby(by).size())
    train_sizes = [round(train_coeff * size) for size in sizes]
    train_indexer = np.array([True for i in range(sum(sizes))])
    # fill with False last (1-train_coeff) rows for every user_id
    current_size = 0
    for i in range(len(train_sizes)):
        train_indexer[current_size + train_sizes[i] : current_size + sizes[i]] = False
        current_size += sizes[i]
    test_indexer = ~train_indexer
    train_df, test_df = cur_df[train_indexer], cur_df[test_indexer]
    
    
    return RatingData(train_df, *rating_data[1:]), RatingData(test_df, *rating_data[1:])

