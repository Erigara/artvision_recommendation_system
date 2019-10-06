#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module provide collection of functions to evaluate recommendation system performance
"""
import numpy as np

def reg_rmse(y, y_hat, penalty=0):
    """
    Compute regularized RMSE
    
    y : np.array
        true values
    
    y_hat : np.array
        predicted values
    
    penalty : float
        regularization penalty
    
    return : float
        regularized RMSE
    """
    rmse = np.sqrt(((y - y_hat)**2).mean())
    return rmse + penalty    

def reg_se(y, y_hat, penalty=0):
    """
    Compute regularized square error
    
    y : np.array
        true values
    
    y_hat : np.array
        predicted values
    
    penalty : float
        regularization penalty
    
    return : float
        regularized square error
    """
    se = ((y - y_hat)**2).sum()
    return se + penalty

def hit_rate():
    pass
