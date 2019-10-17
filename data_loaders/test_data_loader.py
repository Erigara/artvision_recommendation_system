#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: erigara
"""

import pandas as pd
from collections import namedtuple

def load_data(path, names):
    '''
    Read csv file from path, create new sequential indeces for users and items
    
    path : str
        path of csv file
    
    names : array-like
        column names to use, 
        column names must be in following order: [user_ids, items_ids, rating, timestamp] 
    
    return : DF_data - namedtuple
        DF_data structure:
            df : pd.Dataframe 
                loaded dataframe
            user_ids : str
                name of new user_ids column
            item_ids : str
                name of new item_ids column
    '''
    df = pd.read_csv('../data/ratings.csv')
    
    # create new sequential indeces for users and items
    user_ids, item_ids = 'new_' + names[0], 'new_' + names[1]
    df[user_ids] = df[names[0]].rank(method='dense').astype('int64') - 1
    df[item_ids] = df[names[1]].rank(method='dense').astype('int64') - 1
    
    DF_data = namedtuple('DF_data', ['df', 'user_ids', 'item_ids'])
    return DF_data(df, user_ids, item_ids)