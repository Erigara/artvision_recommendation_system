#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Download ratings from database and upload predictions 
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
        column names to use, u
        column names must be in following order: [user_id, item_id, rating, prediction, timestamp] 
    
    return : RatingData
        dowloaded rating data
    '''
    with psycopg2.connect(host=host,
                          port=port,
                          dbname=dbname,
                          user=user,
                          password=password) as conn:
        with conn.cursor(name='record_fetcher') as cursor:
            cursor.itersize = 20000
            template_query = sql.SQL("SELECT {user_id}, {item_id}, {rating}, {timestamp} FROM {table}")
            query = template_query.format(user_id=sql.Identifier(columns[0]),
                                  item_id=sql.Identifier(columns[1]),
                                  rating=sql.Identifier(columns[2]),
                                  timestamp=sql.Identifier(columns[4]),
                                  table=sql.Identifier(table))
            cursor.execute(query)
            df = pd.DataFrame(data=cursor, columns=[columns[0], columns[1], columns[2], columns[4]])
    return RatingData(df, *columns)


def upload_data(rating_data, host, port, dbname, table, columns, user, password):
    '''
    Read rating data from db connection

    
    columns : array-like
        column names to use, u
        column names must be in following order: [user_id, item_id, rating, prediction, timestamp] 
    
    return : RatingData
        dowloaded rating data
    '''
    records = rating_data.df.itertuples(index=False)
    with psycopg2.connect(host=host,
                            port=port,
                            dbname=dbname,
                            user=user,
                            password=password) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql.SQL("TRUNCATE {table}").format(table=sql.Identifier(table)))
            template_query = sql.SQL("INSERT INTO {table} ({user_id}, {item_id}, {prediction}) VALUES {values}")
            values = ','.join(cursor.mogrify("(%s,%s,%s)", record).decode() for record in records)
            query = template_query.format(user_id=sql.Identifier(columns[0]),
                                  item_id=sql.Identifier(columns[1]),
                                  prediction=sql.Identifier(columns[3]),
                                  table=sql.Identifier(table),
                                  values=sql.SQL(values))
            cursor.execute(query)