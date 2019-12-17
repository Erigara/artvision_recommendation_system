#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module implement rating predictor based on ALS matrix decomposition
"""
import multiprocessing as mp
import ctypes
import numpy as np
import pandas as pd
from scipy import sparse as sps
from collections import namedtuple, defaultdict
from multiprocessing import Pool

from model.utils.sparse_matrix_operations import (row_means_nonzero, 
                                                  col_means_nonzero)
from model.utils.metrics import reg_rmse, reg_se
from model.utils.compute_matrix import ComputeMatrix
from model import Losses

import logging
import time

class ALS:
    """
    Find decomposition of matrix R onto matrices U, M so that R ~ U.T * M

    Implement Realization of ALS with Weighted-λ-Regularization from
    "Large-scale Parallel Collaborative Filtering for the Netflix Prize"
    """
    TrainStruct = namedtuple('TrainStruct', ['compute_u',
                                             'compute_m',
                                             'user_I',
                                             'item_I',
                                             'users_len',
                                             'items_len'])
    
    def __init__(self, features=10, regularization='l2', lmbda = 0.3):
        """
        train_data : RatingData
            rating data to train
            
        features : int
            number of hidden features
            
        regularization : str
            'l2' for l2 regularization
            'wl2' for weighted lambda regularization
        
        lmbda : float
            regularization weight
        
        """

        if regularization == 'wl2':
            self.reg_penalty = self.weighted_reg_penalty
            self.regularization = 'wl2'
        elif regularization == 'l2':
            self.reg_penalty = self.l2_penalty
            self.regularization = 'l2'
        else:
            raise ValueError('regularization can be "l2" or "wl2" not "{}"'.format(regularization))

        self.features = features
        self.lmbda = lmbda
        # init?
        self.initilized = False
        # matrix that represent users
        self.U = None
        # matrix that represent items
        self.M = None
        # use inner sequential dense ids
        self.user_ids_mapping = None
        self.item_ids_mapping = None
        
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
            
    def fit(self, train_data,
                  test_data,
                  test_metric=reg_rmse,
                  epochs=5,
                  eps=0.001,
                  eval_every=1):
        """
        Perform iterative optimization algorithm to find matrices M, U so that
        they minimize loss function
        
        test_data : RatingData
            rating data to test
        
        test_metric : function
            function to eval als performance on test dataset
            
        epochs : int 
            number of iterations in algorithm
        
        eps : float
            algorithm stops if difference between last_loss and loss smaller than eps
        """
        train_struct = self.fit_init(train_data)
        train_losses = []
        test_losses = []
        last_loss = None
        
        train_start_time = time.time()

        for epoch in range(epochs):
            self.optimization_step(train_struct)
            self.train_data = self.predict(train_data)
            loss = reg_se(train_data, penalty=self.reg_penalty(train_struct))
            
            if last_loss and abs(loss - last_loss) < eps:
                last_loss = loss
                break
            if epoch % eval_every == 0:
                test_data = self.predict(test_data)
                # fill nan with 0
                test_data.df[test_data.prediction_col_name].fillna(0)
                test_loss = test_metric(test_data)
                
                train_losses.append(loss)
                test_losses.append(test_loss)
                
                logging.info("\n========== Epoch {} ==========".format(epoch))
                logging.info("Train loss: {}".format(train_losses[-1]))
                if last_loss and last_loss < loss:
                    logging.warning("WARNING - Loss Increasing") 
                logging.info("Test loss: {}".format(test_losses[-1]))

            last_loss = loss
        
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        
        logging.info("\n========== Train complete! ==========")
        logging.info("Epochs: {}".format(epoch))
        logging.info("Train loss: {}".format(train_losses[-1]))
        logging.info("Test loss: {}".format(test_losses[-1]))
        logging.info("Totall train time : {:.6f}  sec.".format(train_time))

        return Losses(train_losses, test_losses)
    
    def fit_init(self, train_data):
        """
        Initilize matices M, U using train data,
        create additional data structures necessary for training predictor
        
        train_data : RatingData
            rating data to train
            
        return : ALS.TrainStruct
            namedtuple of additional data structures
        """
        user_ids = train_data.df[train_data.user_col_name]
        item_ids = train_data.df[train_data.item_col_name]
        
        dense_user_ids = user_ids.rank(method='dense').astype('int64') - 1
        dense_item_ids = item_ids.rank(method='dense').astype('int64') - 1
        
        self.user_ids_mapping = dict(user_id_pair for user_id_pair in zip(user_ids, dense_user_ids))
        self.item_ids_mapping = dict(item_id_pair for item_id_pair in zip(item_ids, dense_item_ids))
        
        
        ratings = train_data.df[train_data.rating_col_name]
        
        users_len = int(max(dense_user_ids) + 1)
        items_len = int(max(dense_item_ids) + 1)
        
        # init matrices U and M
        if self.U is None:    
            self.U = self.init_U(dense_user_ids, dense_item_ids, ratings, users_len)
        if self.M is None:
            self.M = self.init_M(dense_user_ids, dense_item_ids, ratings, items_len)
        
        tmp_df = pd.DataFrame({train_data.user_col_name : dense_user_ids,
                               train_data.item_col_name : dense_item_ids})
        # create user_I dict containing for key user np.array of items witch user rate
        user_I = defaultdict(lambda : np.array([], dtype=int))
        user_groups = tmp_df.groupby(train_data.user_col_name)[train_data.item_col_name]
        for user in sorted(user_groups.groups):
            user_I[user] = user_groups.get_group(user).to_numpy().astype(int)
        
        # create item_I dict containing for key item np.array of users whom rate this item
        item_I = defaultdict(lambda : np.array([], dtype=int))
        item_groups = tmp_df.groupby(train_data.item_col_name)[train_data.user_col_name]
        for item in sorted(item_groups.groups):
            item_I[item] = item_groups.get_group(item).to_numpy().astype(int)
        
        compute_u = self.create_compute_u(dense_user_ids, dense_item_ids, ratings, user_I)
        compute_m = self.create_compute_m(dense_user_ids, dense_item_ids, ratings, item_I)

        self.initilized = True

        return ALS.TrainStruct(compute_u, compute_m, user_I, item_I, users_len, items_len)

    def init_U(self, users, items, ratings, users_len):
        """
        Initialize matrix U so that it first row contain user's means
        
        train_data : RatingData
            rating data to train

        items_len : int
            amount of items

        return : np.array
            initilized matrix U
        """
        row_R = sps.csr_matrix((ratings, 
                                (users, 
                                 items)))
        U = np.random.randn(self.features, users_len)
        U[0, :] = row_means_nonzero(row_R)
        return U
    
    def init_M(self, users, items, ratings, items_len):
        """
        Initialize matrix M so that it first row contain item's means

        train_data : RatingData
            rating data to train

        items_len : int
            amount of items

        return : np.array
            initilized matrix M
        """
        col_R = sps.csc_matrix((ratings, 
                                (users, 
                                 items)))
        M = np.random.randn(self.features, items_len)
        M[0, :] = col_means_nonzero(col_R)
        return M

    def optimization_step(self, train_struct):
        """
        Perform single optimization step op algorithm.
        
        train_data : RatingData
            rating data to train
            
        train_struct : self.TrainStruct
            namedtuple of additional data structures
        """
        compute_U = ComputeMatrix(lambda : train_struct.compute_u, 
                                  self.features, 
                                  train_struct.users_len)
        compute_M = ComputeMatrix(lambda : train_struct.compute_m, 
                                  self.features, 
                                  train_struct.items_len)

        self.U = compute_U.compute_parallel()
        self.M = compute_M.compute_parallel()
  
    def create_compute_u(self, users, items, ratings, user_I):
        """
        Create function to compute column of U by it's index
        
        train_data : RatingData
            rating data to train
        
        user_I : dict
            dict containing for key user np.array of items witch user rate
            
        return : function
            function to compute columns of U
        """
        row_R = sps.csr_matrix((ratings, 
                                (users, 
                                 items)))
        E = np.eye(self.features)
        lmbda = self.lmbda
        
        def compute_u(i):
            """
            Compute column i of matrix U
        
            i : int 
                index of column U
        
            return : np.array with shape (self.features, )
                new column i of matrix U
            """
            I_i = user_I[i]
            n_i = len(I_i) if self.regularization == 'wl2' else 1
            M_i = self.M[:, I_i]
            A_i = M_i @ M_i.T + lmbda * n_i * E
            V_i = M_i @ row_R[i, I_i].T
            u_i = np.linalg.solve(A_i, V_i)
            return u_i.flatten()
        
        return compute_u
    
    def create_compute_m(self, users, items, ratings, item_I):
        """
        Create function to compute column of M by it's index
        
        train_data : RatingData
            rating data to train
        
        item_I : dict 
            dict containing for key item np.array of users whom rate this item
            
        return : function
            function to compute columns of M
        """      
        col_R = sps.csc_matrix((ratings, 
                                (users, 
                                 items)))
        E = np.eye(self.features)
        lmbda = self.lmbda
        
        def compute_m(j):
            """
            Compute column j of matrix M
        
            j : int
                index of column M
        
            return : np.array with shape (self.features, )
                new column j of matrix M
            """
            I_j = item_I[j]
            n = len(I_j) if self.regularization == 'wl2' else 1
            U_j = self.U[:, I_j]
            A_j = U_j @ U_j.T + lmbda * n * E
            V_j = U_j @ col_R[I_j, j]
            m_j = np.linalg.solve(A_j, V_j)
            return m_j.flatten()
        return compute_m
    
    def weighted_reg_penalty(self, train_struct):
        """
        Compute weighted lambda regularization penalty
        
        train_struct : self.TrainStruct
            namedtuple of additional data structures
            
        return : float 
            weighted lambda regularization penalty
        """
        user_I, item_I = train_struct.user_U, train_struct.item_I
        # Tikhonov regularization matrices
        Gu = self.U @ sps.diags([len(user_I[i]) for i in user_I])
        Gm = self.M @ sps.diags([len(item_I[i]) for i in item_I])
        
        penalty = self.lmbda * ((Gu**2).sum() + (Gm**2).sum())
        return penalty
    
    def l2_penalty(self, train_struct):
        """
        Compute l2 regularization penalty

        train_struct : self.TrainStruct
            namedtuple of additional data structures

        return : float 
            l2 penalty
        """
        penalty = self.lmbda * ((self.U**2).sum() + (self.M**2).sum())
        return penalty
    
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
            list of model ids used in model
        """
        if self.item_ids_mapping:
            return list(self.item_ids_mapping.keys())
        else:
            return None