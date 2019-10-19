#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module implement unpersonalized predictor based on mean and user's/item's deviation from mean
"""

from multiprocessing import Pool
import pandas as pd
import numpy as np
import logging
class BaselinePredictor:
    """
    Based on mean, user_means, item_means compute predicted rating
    rating(i, j) = user_means(i) + item_mean(j) - mean
    
    """
    def __init__(self, train_data):
        """
        Set predictor parameters
    
        train_data : RatingData
            rating data to train
        """
        self.mean = train_data.df[train_data.rating_col_name].mean()
        self.user_means = np.array(train_data.df.groupby(train_data.user_col_name).mean())
        self.item_means = np.array(train_data.df.groupby(train_data.item_col_name).mean())
    
    def predict_single(self, user, item):
        """
        Compute baseline_rating for user, item pair
        user_item : tuple
            (user, item)
            
        return : float
            baseline_rating = user_means(user) + item_mean(item) - mean
        """        
        try:
            user_mean = self.user_means[user]
            item_mean = self.item_means[item]
            
            baseline_rating = user_mean + item_mean - self.mean
        except (IndexError, KeyError):
            logging.warning('invalid ids pair ({}, {}) in user_item_iterable'.format(user, item))
            baseline_rating = np.nan 
            
        return baseline_rating
    
    def predict(self, data):
        """
        Compute baseline_rating for each user, item pair in user_item_iterable
        
        data : RatingData
            rating data
            
        return : np.array
            list of baseline_ratings
        """
        with Pool() as p:
            data.df[data.prediction_col_name] = np.fromiter(p.starmap(self.predict_single, 
                                                                      zip(data.df[data.user_col_name], 
                                                                          data.df[data.item_col_name])), 
                                                            dtype=np.float)
        return data