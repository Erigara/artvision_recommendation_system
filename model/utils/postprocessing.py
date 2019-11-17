#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module provide postprocessing functions
"""
import pandas as pd
from data_loaders.rating_data import RatingData

def truncate_rating(data, lower_bound, upper_bound):
            """
            Set values in column predictions so that every value  lower_bound <= value <= upper_bound
            
            data : RatingData
                rating data
                
            lower_bound : float
                smallest possible rating
            
            upper_bound : float
                biggest possible rating
                
            return : RatingData
                return RatingData where every rating in col prediction
                                = rating      if lower_bound <= rating <= upper_bound
                                = lower_bound if lower_bound > rating
                                = upper_bound if rating > upper_bound
            """
            ratings = data.df[data.prediction_col_name].to_numpy()
            lower = ratings < lower_bound
            ratings[lower] = lower_bound
            higher = ratings > upper_bound
            ratings[higher] = upper_bound
            data.df[data.prediction_col_name] = ratings
            return data
        

def make_recommendation(model, user_id, top_n=5):
    """
    Return top n predicted ratings for user_id across all items
    
    model : obj
        model to make predictions
    
    user_id : int
        user id
    
    top_n : int
        amount of movies in recommendations
        
    return : RatingData
        rating_data of top items for user
    """
    users_len, items_len = model.get_shape()
    if items_len:
        df = pd.DataFrame({'user_id' : [user_id for item_id in range(items_len)], 
                           'item_id' : [item_id for item_id in range(items_len)],
                           'rating'  : [0       for item_id in range(items_len)]})
        
        user_data = RatingData(df, 'user_id', 
                                   'item_id', 
                                   'rating',
                                   'prediction',
                                   'timestamp')
        user_data = model.predict(user_data)
        top_n_df = user_data.df.nlargest(top_n, user_data.prediction_col_name)
        
        return RatingData(top_n_df, 'user_id', 
                                    'item_id', 
                                    'rating',
                                    'prediction',
                                    'timestamp')
    else:
        return None