#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: erigara
"""

import pandas as pd
from data_loaders.rating_data import RatingData

def load_data(path, names):
    '''
    Read csv file from path, create new sequential indeces for users and items
    
    path : str
        path of csv file
    
    names : array-like
        column names to use, 
        column names must be in following order: [user_ids, items_ids, rating, prediction, timestamp] 
    
    return : RatingData
        dowloaded rating data
    '''
    df = pd.read_csv('../data/ratings.csv', 
                     names=[names[0], names[1], names[2], names[4]], 
                     skiprows=1)
    
    # create new sequential indeces for users and items
    '''
    user_ids, item_ids = 'new_' + names[0], 'new_' + names[1]
    df[user_ids] = df[names[0]].rank(method='dense').astype('int64') - 1
    df[item_ids] = df[names[1]].rank(method='dense').astype('int64') - 1
    names[0], names[1] = user_ids, item_ids
    '''
    return RatingData(df, *names)