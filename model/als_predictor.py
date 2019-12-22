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
from collections import namedtuple, defaultdict, OrderedDict
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
                                             'user_subset',
                                             'item_subset',
                                             'user_I',
                                             'item_I'])
    
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
        assert len(train_data.df) != 0
        user_ids = train_data.df[train_data.user_col_name]
        item_ids = train_data.df[train_data.item_col_name]
        ratings = train_data.df[train_data.rating_col_name]
        
        inner_user_ids = []
        inner_item_ids = []
        new_users_len = self.users_len
        new_items_len = self.items_len
        for user, item in zip(user_ids, item_ids):
            if not user in self.user_ids_mapping:
                self.user_ids_mapping[user] = new_users_len
                new_users_len += 1
            inner_user_ids.append(self.user_ids_mapping[user])
            
            if not item in self.item_ids_mapping:
                self.item_ids_mapping[item] = new_items_len
                new_items_len += 1
            inner_item_ids.append(self.item_ids_mapping[item])
            
        dense_user_ids = pd.Series(inner_user_ids).rank(method='dense').astype('int64') - 1
        dense_item_ids = pd.Series(inner_item_ids).rank(method='dense').astype('int64') - 1
        
        # update matrices U and M
        self.U = self.update_matrix(self.U, dense_user_ids, dense_item_ids, 
                                    ratings, new_users_len, self.users_len)
        self.M = self.update_matrix(self.M, dense_item_ids, dense_user_ids, 
                                    ratings, new_items_len, self.items_len)
        
        self.users_len = new_users_len
        self.items_len = new_items_len
        
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
        
        # create subset cols for users and items
        user_subset = sorted(list(set(inner_user_ids)))
        item_subset = sorted(list(set(inner_item_ids)))
        
        compute_u = self.create_compute_u(dense_user_ids, dense_item_ids, ratings, item_subset, user_I)
        compute_m = self.create_compute_m(dense_user_ids, dense_item_ids, ratings, user_subset, item_I)

        self.initilized = True

        return ALS.TrainStruct(compute_u, compute_m, user_subset, item_subset, user_I, item_I)
    
    def update_matrix(self, matrix, rows, cols, ratings, new_cols_amount, old_cols_amount):
        """
        Update given matrix so that it add new columns to matrix
        and init it's first row to have nonzero rows means of matrix R
        
        matrix : np.array
            matrix to be updated

        new_cols_amount : int
            new amount of columns in matrix
        
        old_cols_amount : int
            old amount of columns in matrix
            
        return : np.array
            updated matrix
        """
        row_R = sps.csr_matrix((ratings, 
                                (rows, 
                                 cols)))
        
        delta_cols = new_cols_amount - old_cols_amount
        if delta_cols != 0:
            addition_matrix = np.random.randn(self.features, delta_cols)
            addition_matrix[0, :] = row_means_nonzero(row_R)[-delta_cols:]
            
            if matrix is None:
                new_matrix = addition_matrix
            else:
                new_matrix = np.append(matrix, addition_matrix, axis=1)
        else:
            new_matrix = matrix
        return new_matrix

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
                                  len(train_struct.user_subset))
        compute_M = ComputeMatrix(lambda : train_struct.compute_m, 
                                  self.features, 
                                  len(train_struct.item_subset))

        self.U[:, train_struct.user_subset] = compute_U.compute_parallel()
        self.M[:, train_struct.item_subset] = compute_M.compute_parallel()
        
    def _create_compute(self, matrix_name, rows, cols, ratings, subset_cols, I):
        """
        Create function to compute column of matrix by it's index
        
        rows : arraylike
            row indices
            
        cols : arraylike
            column indices
        
        ratings : arraylike
            values of matrix R where R[i, j] = r[k] with k so that rows[k] = i 
            and cols[k] = j
        
        subset_cols : arraylike
            indices of matrix used to compute column
        
        I : arraylike
            arraylike structur where element I[i] arraylike containing columns of matrix R 
            so that intersection of row i and this columns has not 0 values
            
        return : function
            function to compute columns of U
        """
        row_R = sps.csr_matrix((ratings, 
                                (rows, 
                                 cols)))
        E = np.eye(self.features)
        lmbda = self.lmbda
        
        def compute_column(i):
            """
            Compute column i of matrix
        
            i : int 
                index of matrix's column
        
            return : np.array
                new column i of matrix with shape (self.features, )
            """
            submatrix = getattr(self, matrix_name)[:, subset_cols]
            I_i = I[i]
            n_i = len(I_i) if self.regularization == 'wl2' else 1
            matrix_i = submatrix[:, I_i]
            A_i = matrix_i @ matrix_i.T + lmbda * n_i * E
            V_i = matrix_i @ row_R[i, I_i].T
            column_i = np.linalg.solve(A_i, V_i)
            return column_i.flatten()
        return compute_column
    
    def create_compute_u(self, users, items, ratings, subset_cols, user_I):
        return self._create_compute('M', users, items, ratings, subset_cols, user_I)
    
    def create_compute_m(self, users, items, ratings, subset_cols, item_I):
        return self._create_compute('U', items, users, ratings, subset_cols, item_I)
    
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