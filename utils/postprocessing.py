#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module provide postprocessing functions
"""

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