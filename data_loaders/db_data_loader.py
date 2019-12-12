#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara
"""
import psycopg2
import pandas as pd
import sys
sys.path.append('/data/artvision/recommendation_system')

from data_loaders.rating_data import RatingData

DEC2FLOAT = psycopg2.extensions.new_type(
    psycopg2.extensions.DECIMAL.values,
    'DEC2FLOAT',
    lambda value, curs: float(value) if value is not None else None)
psycopg2.extensions.register_type(DEC2FLOAT)

def load_data(path, names):
    '''
    Read from db connection, create new sequential indeces for users and items
    
    path : str
        path of csv file
    
    names : array-like
        column names to use, 
        column names must be in following order: [user_ids, item_ids, rating, prediction, timestamp] 
    
    return : RatingData
        dowloaded rating data
    '''
    conn = psycopg2.connect(dbname='rating_db', user='postgres',  host='localhost')
    cursor = conn.cursor(name='record_fetcher')
    cursor.itersize = 20000
    cursor.execute('SELECT userId, movieId, rating, timestamp FROM ratings')
    df = pd.DataFrame(data=cursor, columns=[names[0], names[1], names[2], names[4]])
    conn.close()
    # create new sequential indeces for users and items
    '''
    user_ids, item_ids = 'new_' + names[0], 'new_' + names[1]
    df[user_ids] = df[names[0]].rank(method='dense').astype('int64') - 1
    df[item_ids] = df[names[1]].rank(method='dense').astype('int64') - 1   
    names[0], names[1] = user_ids, item_ids
    '''
    return RatingData(df, *names)