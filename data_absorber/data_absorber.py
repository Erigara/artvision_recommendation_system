#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module implement data absorber that collect data in db 
and prepare it to be used as train and test data in the model
"""

import queue
import threading
import logging

from utils.worker import template_worker

class DataAbsorber:
    def __init__(self, db, min_user_records, min_item_records, test_records_num):
        self._db = db
        self.reciever_queue = queue.Queue()
        self.transmitter_queue = queue.Queue()
        self.reciver = threading.Thread(target=self.reciever_worker)
        self.transmitter = threading.Thread(target=self.transmitter_worker)
        self.min_user_records = min_user_records
        self.min_item_records = min_item_records
        self.test_records_num = test_records_num
    
    @property
    def min_user_records(self):
        """
        Variable filtering users with small number of records
        """
        return self._min_user_records

    @min_user_records.setter
    def min_user_records(self, value):
        if value > 0:
            self._min_user_records = value
        else:
            raise ValueError(f'Minimal user records must be positive, but {value} is given')

    @property
    def min_item_records(self):
        """
        Variable filtering items with small number of records
        """
        return self._min_item_records

    @min_item_records.setter
    def min_item_records(self, value):
        if value > 0:
            self._min_item_records = value
        else:
            raise ValueError(f'Minimal item records must be positive, but {value} is given')

    @property
    def test_records_num(self):
        """
        Variable controling number of records in test data
        """
        return self._test_records_num

    @test_records_num.setter
    def test_records_num(self, value):
        if value > 0:
            self._test_records_num = value
        else:
            raise ValueError(f'Number of test records must be positive, but {value} is given')

    @property
    def transmitter_callback(self):
        """
        Callback called when transmitter  train and test data
        must accept tuple that collect train and test data 
        """
        return self._transmitter_callback

    @transmitter_callback.setter
    def transmitter_callback(self, callback):
        if callback.__code__.co_argcount == 1:
            self._transmitter_callback = callback
        else:
            raise ValueError(f'Callback function must take only 1 argument')
    
    def start(self):
        """
        Activate reciever and transmitter
        Can be started only once
        """
        self.reciver.start()
        self.transmitter.start()
    
    def stop(self):
        """
        Deactivate reciever and transmitter
        """
        self.reciever_queue.put(None)
        self.transmitter_queue.put(None)
    
    def reciever_func(self, records):
        """
        Function used inside reciver worker loop
        -> insert records into db 
        -> extract ids from records 
        -> put them into transmitter queue
        
        records : arraylike
            recieved records
        """
        logging.debug('Got records')
        # insert records into database
        self._db.insert_records(records)
        logging.debug('Records inserted')
        
        # get user ids
        user_ids = list(set([record[0] for record in records]))
        item_ids = list(set([record[1] for record in records]))
        logging.debug('User ids and Item ids extracted')
        
        self.transmitter_queue.put((user_ids, item_ids))
        logging.debug('User ids and Item ids putted into transmitter queue')

    def transmitter_func(self, ids):
        """
        Function used inside transmitter worker loop
        -> fetch train and test data from db
        -> call callback with fetched data
        
        ids : tuple
            tuple consist of two lists user_ids and item_ids
        """
        user_ids, item_ids = ids
        if len(user_ids) < len(item_ids):
            train_data = self._db.get_records_by_user(user_ids, min_records=self.min_user_records)
        else:
            train_data = self._db.get_records_by_item(item_ids, min_records=self.min_item_records)
        
        test_data = self._db.get_random_records(records_num=self.test_records_num)
        
        self.transmitter_callback((train_data, test_data))

    def reciever_worker(self):
        """
        Reciever worker loop
        """
        return template_worker(self.reciever_func, self.reciever_queue)
    
    def transmitter_worker(self):
        """
        Transmitter worker loop
        """
        return template_worker(self.transmitter_func, self.transmitter_queue)
        
    def recieve_records(self, records):
        """
        Put records in reciver queue
        
        records : list
            list of records

        """
        self.reciever_queue.put(records)