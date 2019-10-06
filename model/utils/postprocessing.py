#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module provide postprocessing functions
"""

def truncate_rating(ratings, lower_bound, upper_bound):
            """
            ratings : np.array
                array of ratings
            
            lower_bound : float
                smallest possible rating
            
            upper_bound : float
                biggest possible rating
                
            return : np.array
                return array where every rating 
                                = rating      if lower_bound <= rating <= upper_bound
                                = lower_bound if lower_bound > rating
                                = upper_bound if rating > upper_bound
            """
            ratings = ratings.copy()
            lower = ratings < lower_bound
            ratings[lower] = lower_bound
            higher = ratings > upper_bound
            ratings[higher] = upper_bound
            return ratings