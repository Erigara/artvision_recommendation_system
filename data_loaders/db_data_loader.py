#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara
"""
import psycopg2
from psycopg2 import sql
import pandas as pd
import sys
sys.path.append('/data/artvision/recommendation_system')

from data_loaders.rating_data import RatingData

DEC2FLOAT = psycopg2.extensions.new_type(
    psycopg2.extensions.DECIMAL.values,
    'DEC2FLOAT',
    lambda value, curs: float(value) if value is not None else None)
psycopg2.extensions.register_type(DEC2FLOAT)

def load_data(host, port, dbname, table, columns, user, password):
    '''
    Read rating data from db connection

    
    columns : array-like
        column names to use, 
        column names must be in following order: [user_id, item_id, rating, prediction, timestamp] 
    
    return : RatingData
        dowloaded rating data
    '''
    conn = psycopg2.connect(host=host,
                            port=port,
                            dbname=dbname,
                            user=user,
                            password=password)
    cursor = conn.cursor(name='record_fetcher')
    cursor.itersize = 20000
    template_query = sql.SQL("SELECT {user_id}, {item_id}, {rating}, {timestamp} FROM {table}")
    query = template_query.format(user_id=sql.Identifier(columns[0]),
                                  item_id=sql.Identifier(columns[1]),
                                  rating=sql.Identifier(columns[2]),
                                  timestamp=sql.Identifier(columns[4]),
                                  table=sql.Identifier(table))
    cursor.execute(query)
    df = pd.DataFrame(data=cursor, columns=[columns[0], columns[1], columns[2], columns[4]])
    conn.close()
    
    return RatingData(df, *columns)