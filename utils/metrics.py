#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module provide collection of functions to evaluate recommendation system performance
"""
import pandas as pd
import numpy as np

def reg_rmse(data, penalty=0):
    """
    Compute regularized RMSE
    
    data : RatingData
    
    penalty : float
        regularization penalty
    
    return : float
        regularized RMSE
    """
    y = data.df[data.rating_col_name]
    y_hat = data.df[data.prediction_col_name]
    rmse = np.sqrt(((y - y_hat)**2).mean())
    return rmse + penalty

def reg_se(data, penalty=0):
    """
    Compute regularized square error
    
    data : RatingData
    
    penalty : float
        regularization penalty
    
    return : float
        regularized square error
    """
    y = data.df[data.rating_col_name]
    y_hat = data.df[data.prediction_col_name]
    
    se = ((y - y_hat)**2).sum()
    return se + penalty

def NDCG(data, by):
    """
    Compute NDCG

    data : RatingData
    
    by : str
        name of column to perform groubpy
        
    y : np.array of float
        true values
    
    y_hat : np.array of float
        predicted values
    
    return : float
        NDCG
    """
    
    def ndcg_single(x):
        n = len(x)
        p = np.arange(0, n)
        ratings = np.array(x)
        
        return np.sum((2 ** ratings - 1) / (np.log2(2 + p)))
    
    cur_df = data.df.sort_values(data.rating_col_name)
    ndcgs = cur_df.groupby(by).agg(
                ideal_ndcg = pd.NamedAgg(column=data.rating_col_name, 
                                         aggfunc=ndcg_single),
                prediction_ndcg = pd.NamedAgg(column=data.prediction_col_name, 
                                              aggfunc=ndcg_single)
            )
    n = len(ndcgs)
    return np.sum((1 / ndcgs['ideal_ndcg']) * ndcgs['prediction_ndcg'])/n
    