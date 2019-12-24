#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module provide postprocessing functions
"""
import pandas as pd
import logging
import time
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
        

def make_recommendation_userwise(model, columns, top_n=5):
    """
    Return top n predicted ratings for ids all ids in model using userwise approach
    
    model : obj
        model to make predictions
    
    columns : list of str
        column names to use
    
    top_n : int
        amount of movies in recommendations
        
    return : RatingData
        rating_data reccomendations
    """
    item_ids = model.get_item_ids()
    user_ids = model.get_user_ids()
    
    recommendations = []
    start_time = time.time()
    for i, user_id in enumerate(user_ids):
        preds = model.predict_for_user(user_id)
        frame = pd.DataFrame({columns[0] : [user_id for item_id in item_ids], 
                              columns[1] : item_ids,
                              columns[3] : preds})
        user_data = RatingData(frame, *columns)
        top_n_df = user_data.df.nlargest(top_n, user_data.prediction_col_name)
        recommendations.append(top_n_df)
        if i % 1000 == 0:
            logging.info(f'Process {i} users within {time.time() - start_time} seconds')
    recommendations = pd.concat(recommendations)
    return RatingData(recommendations, *columns)
    
def make_recommendation_itemwise(model, columns, top_n=100):
    """
    Return top n predicted ratings for ids all ids in model using itemwise approach
    
    model : obj
        model to make predictions
    
    columns : list of str
        column names to use
    
    top_n : int
        amount of movies in recommendations
        
    return : RatingData
        rating_data of top items for user
    """
    item_ids = model.get_item_ids()
    user_ids = model.get_user_ids()
    
    recommendations = []
    start_time = time.time()
    for i, item_id in enumerate(item_ids):
        preds = model.predict_for_item(item_id)
        frame = pd.DataFrame({columns[0] : user_ids, 
                              columns[1] : [item_id for user_id in user_ids],
                              columns[3] : preds})
        
        item_data = RatingData(frame, *columns)
        top_n_df = item_data.df.nlargest(top_n, item_data.prediction_col_name)
        recommendations.append(top_n_df)
        if i % 1000 == 0:
            logging.info(f'Process {i} items within {time.time() - start_time} seconds')
    recommendations = pd.concat(recommendations)
    return RatingData(recommendations, *columns)