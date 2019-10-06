#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module implement unpersonalized predictor based on mean and user's/item's deviation from mean
"""

from multiprocessing import Pool
import numpy as np

class BaselinePredictor:
    """
    Based on mean, user_means, item_means compute predicted rating
    rating(i, j) = user_means(i) + item_mean(j) - mean
    
    """
    def __init__(self, 
                 mean=0, 
                 user_means=[], 
                 item_means=[]):
        """
        Set predictor parameters
        mean : float
            rating over all records
    
        user_means : object with __getitem__ method
            user_means[user_id] = user_mean 
    
        item_means : object with __getitem__ method
            item_means[item_id] = item_mean
        """
        self.mean = mean
        self.item_means = item_means
        self.user_means = user_means    
        
    def set_params(self, 
                   mean=None, 
                   user_means=None, 
                   item_means=None):
        """
        Change predictor parameters
        mean : float
            rating over all records
    
        user_means : object with __getitem__ method
            user_means[user_id] = user_mean 
    
        item_means : object with __getitem__ method
            item_means[item_id] = item_mean
        """
        
        def update_value(old_value, new_value):
            return old_value if new_value is None else new_value
        
        self.mean = update_value(self.mean, mean)
        self.item_means = update_value(self.item_means, item_means)
        self.user_means = update_value(self.user_means, user_means)
        
        
    
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
        except (IndexError, KeyError):
            raise Warning('invalid ids pair ({}, {}) in user_item_iterable'.format(user, item))
            
        baseline_rating = user_mean + item_mean - self.mean 
        
        return baseline_rating
    
    def predict(self, user_item_iterable):
        """
        Compute baseline_rating for each user, item pair in user_item_iterable
        
        user_item_iterable : iterable
            iterable structure of (user, item) pairs
            [(u_1, i_1), ... , (u_n, i_n)]
            
        return : np.array
            list of baseline_ratings
        """
        with Pool() as p:
            return np.fromiter(p.starmap(self.predict_single, user_item_iterable), dtype=np.float)