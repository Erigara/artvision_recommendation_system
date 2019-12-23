#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara
"""
import psycopg2
from psycopg2 import pool as pgpool
from psycopg2 import sql
import pandas as pd

from data_loaders.rating_data import RatingData

class AbsorberDB:
    def __init__(self, host, port, dbname, table, columns, user, password,
                 minconn=2, maxconn=10):
        self.pool = pgpool.ThreadedConnectionPool(minconn=minconn, 
                                                         maxconn=maxconn,
                                                         host=host,
                                                         port=port,
                                                         dbname=dbname,
                                                         user=user,
                                                         password=password)
        self.table = table
        self.columns = columns
        
        
    def close(self):
        """
        Close all connections
        """
        self.pool.closeall()
    
    def insert_records(self, records):
        """
        Insert records into table
        
        resords : arraylike
            arraylike that contain records
        """
        table = self.table
        columns = self.columns
        # get connection
        conn  = self.pool.getconn()
        with conn.cursor() as cursor:
            template_query = sql.SQL('''INSERT INTO {table} ({user_id}, {item_id}, 
                                     {rating}, {timestamp}) VALUES {values}''')
            values = ','.join(cursor.mogrify("(%s,%s,%s,%s)", record).decode() for record in records)
            query = template_query.format(user_id=sql.Identifier(columns[0]),
                                          item_id=sql.Identifier(columns[1]),
                                          rating=sql.Identifier(columns[2]),
                                          timestamp=sql.Identifier(columns[3]),
                                          table=sql.Identifier(table),
                                  values=sql.SQL(values))
            cursor.execute(query)
        conn.commit()
        # release the connection
        self.pool.putconn(conn)
        
    def get_random_records(self, records_num):
        """
        Get random records from table
        TODO : not realy random!!! with small records_num
        
        records_num : int
            number of records to take
        
        return : RatingData
            fetched random records
        """
        table = self.table
        columns = self.columns
        # get connection
        conn  = self.pool.getconn()
        
        with conn.cursor(name='record_fetcher') as cursor:
            cursor.itersize = min(20000, records_num)
            template_query = sql.SQL('''SELECT {user_id}, {item_id}, {rating}, {timestamp} 
                                     FROM {table} TABLESAMPLE SYSTEM_ROWS(%s)''')
            query = template_query.format(user_id=sql.Identifier(columns[0]),
                                          item_id=sql.Identifier(columns[1]),
                                          rating=sql.Identifier(columns[2]),
                                          timestamp=sql.Identifier(columns[4]),
                                          table=sql.Identifier(table))
            cursor.execute(query, [records_num, ])
            frame = pd.DataFrame(data=cursor, columns=[columns[0], columns[1], columns[2], columns[4]])
        # release the connection
        self.pool.putconn(conn)
        
        return RatingData(frame, *columns)
    
    def _get_records_by_id(self, id_column, ids, min_records=10):
        """
        Get records from table where id_column id in ids
        and id appeare at least in min_records records
        
        id_column : int
            index of column in self.columns
        
        ids : arraylike
            list of ids to select from them
        
        min_records : int
            id appeare at least min_records times in records
        
        return : RatingData
            fetched filtred data for given ids
        """
        table = self.table
        columns = self.columns
        # get connection
        conn  = self.pool.getconn()
        
        with conn.cursor(name='record_fetcher') as cursor:
            cursor.itersize = 20000
            ids = ','.join(cursor.mogrify("%s", [_id,]).decode() for _id in ids)
            template_query = sql.SQL('''SELECT {user_id}, {item_id}, {rating}, {timestamp} 
                                     FROM {table}
                                     WHERE {id_column} IN ({ids})
                                     AND {id_column} IN (SELECT {id_column} 
                                                         FROM {table}
                                                         GROUP BY {id_column}
                                                         HAVING count(1) >= %s)''')

            query = template_query.format(user_id=sql.Identifier(columns[0]),
                                          item_id=sql.Identifier(columns[1]),
                                          rating=sql.Identifier(columns[2]),
                                          timestamp=sql.Identifier(columns[4]),
                                          id_column=sql.Identifier(columns[id_column]),
                                          ids=sql.SQL(ids),
                                          table=sql.Identifier(table))
            cursor.execute(query, [min_records, ])
            frame = pd.DataFrame(data=cursor, columns=[columns[0], columns[1], columns[2], columns[4]])
        # release the connection
        self.pool.putconn(conn)
        
        return RatingData(frame, *columns)

    def get_records_by_user(self, user_ids, min_records=10):
        """
        Get records from table where user_id id user_ids
        and user_id appeare at least min_records times in records
        
        user_ids : arraylike
            list of user ids to select from them
        
        min_records : int
            user_id appeare at least min_records times in records
        
        return : RatingData
            fetched filtred data for given user_ids
        """
        return self._get_records_by_id(0, user_ids, min_records)
    
    def get_records_by_item(self, item_ids, min_records=10):
        """
        Get records from table where item_id in item_ids
        and item_id appeare at least min_records times in records
        
        item_ids : arraylike
            list of user ids to select from them
        
        min_records : int
            item_id appeare at least min_records times in records
        
        return : RatingData
            fetched filtred data for given item_ids
        """
        return self._get_records_by_id(1, item_ids, min_records)