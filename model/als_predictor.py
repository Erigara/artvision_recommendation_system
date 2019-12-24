#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module implement rating predictor based on ALS matrix decomposition
"""
import multiprocessing as mp
import ctypes
import numpy as np
from collections import  OrderedDict

from utils.sparse_matrix_operations import (row_means_nonzero, 
                                                  col_means_nonzero)

import logging
import time

class ALS:
    """
    Find decomposition of matrix R onto matrices U, M so that R ~ U.T * M

    Implement Realization of ALS with Weighted-λ-Regularization from
    "Large-scale Parallel Collaborative Filtering for the Netflix Prize"
    """
    
    def __init__(self, features=10):
        """
        train_data : RatingData
            rating data to train
            
        features : int
            number of hidden features
        
        """

        self.features = features
        # init?
        self.initilized = False
        # matrix that represent users
        self.U = None
        # matrix that represent items
        self.M = None
        # use inner sequential ids
        self.users_len = 0
        self.items_len = 0
        self.user_ids_mapping = OrderedDict()
        self.item_ids_mapping = OrderedDict()
        
    def predict(self, data):
        """
        Predict rating for each (user, item) indecies pair in users, items

        data : RatingData
            rating rata
            
        return : np.array
            predicted ratings
        """

        def predict_worker(tuple_rows, size, queue, raw_predictions):
            predictions = np.reshape(np.frombuffer(raw_predictions), (size, ))
            while True:
                job = queue.get()
                if job == None:
                    break
                start, stop = job[0], job[0] + job[1]
                for i in range(start, stop):
                    predictions[i] = self.predict_single(*tuple_rows[i])
                queue.task_done()
            queue.task_done()
            
        if not self.initilized:
            logging.error('predictor must be fitted first!')
            raise RuntimeError('predictor must be fitted first!')
            
        size = len(data.df)
        tuple_rows =  list(zip(data.df[data.user_col_name], data.df[data.item_col_name]))
        
        # create shared memory array
        raw_predictions = mp.RawArray(ctypes.c_double, size)
        
        n_cpu  = mp.cpu_count()
        n_jobs = n_cpu
        
        q = size // n_jobs
        r = size  % n_jobs
 
        # spread colums ids matrix between jobs
        jobs = []
        first_col = 0
        for i in range(n_jobs):
            cols_in_job = q
            if (r > 0):
                cols_in_job += 1
                r -= 1
            jobs.append((first_col, cols_in_job))
            first_col += cols_in_job
        
        queue = mp.JoinableQueue()
        for job in jobs:
            queue.put(job)
        for i in range(n_cpu):
            queue.put(None)
        
        # run workers
        workers = []
        for i in range(n_cpu):
            worker = mp.Process(target = predict_worker,
                            args = (tuple_rows, size, queue, raw_predictions))
            workers.append(worker)
            worker.start()
        
        queue.join()
        
        # make array from shared memory    
        predictions = np.reshape(np.frombuffer(raw_predictions), (size,))
        
        data.df[data.prediction_col_name] = predictions
       
        return data
    
    def predict_single(self, user, item):
        if not self.initilized:
            logging.error('predictor must be fitted first!')
            raise RuntimeError('predictor must be fitted first!')
        try:
            # map ids to inner ids
            dense_user = self.user_ids_mapping[user]
            dense_item = self.item_ids_mapping[item]
            rating_hat = np.dot(self.U[:, dense_user], self.M[:, dense_item])
        except KeyError:
            #logging.warning('invalid ids pair ({}, {}) in user_item_iterable'.format(user, item))
            rating_hat = np.nan
        return rating_hat
      
    def predict_for_user(self, user):
        if not self.initilized:
            logging.error('predictor must be fitted first!')
            raise RuntimeError('predictor must be fitted first!')
        try:
            # map ids to inner ids
            dense_user = self.user_ids_mapping[user]
            rating_hat = (self.M.T @ self.U[:, dense_user]).flatten()
        except KeyError:
            #logging.warning('invalid ids pair ({}, {}) in user_item_iterable'.format(user, item))
            rating_hat = np.nan
        return rating_hat
    
    def predict_for_item(self, item):
        if not self.initilized:
            logging.error('predictor must be fitted first!')
            raise RuntimeError('predictor must be fitted first!')
        try:
            # map ids to inner ids
            dense_item = self.idem_ids_mapping[item]
            rating_hat = (self.U.T @ self.M[:, dense_item]).flatten()
        except KeyError:
            #logging.warning('invalid ids pair ({}, {}) in user_item_iterable'.format(user, item))
            rating_hat = np.nan
        return rating_hat
    
    def get_shape(self):
        """
        Return (self.U.shape[1], self.M.shape[1]) or (None, None)
        """
        if self.M is None or self.U is None:
            return (None, None)
        else:
            return (self.U.shape[1], self.M.shape[1])

    def get_user_ids(self):
        """
        Return: 
            list of user ids used in model
        """
        if self.user_ids_mapping:
            return list(self.user_ids_mapping.keys())
        else:
            return None

    def get_item_ids(self):
        """
        Return:
            list of item ids used in model
        """
        if self.item_ids_mapping:
            return list(self.item_ids_mapping.keys())
        else:
            return None