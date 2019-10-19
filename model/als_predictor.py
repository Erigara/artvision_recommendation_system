#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module implement rating predictor based on ALS matrix decomposition
"""

import numpy as np
import pandas as pd
from scipy import sparse as sps
from collections import namedtuple, defaultdict
from model.utils.sparse_matrix_operations import (row_means_nonzero, 
                                                  col_means_nonzero)
from model.utils.metrics import reg_rmse, reg_se
from model.utils.compute_matrix import ComputeMatrix

import logging
import time

class ALS:
    """
    Find decomposition of matrix R onto matrices U, M so that R ~ U.T * M

    Implement Realization of ALS with Weighted-λ-Regularization from
    "Large-scale Parallel Collaborative Filtering for the Netflix Prize"
    """
    def __init__(self, train_data, features=10, regularization='l2', lmbda = 0.3):
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
        
        !!! To work properly ids must start form 0.
        """
        
        if regularization == 'wl2':
            self.reg_penalty = self.weighted_reg_penalty
            self.regularization = 'wl2'
        elif regularization == 'l2':
            self.reg_penalty = self.l2_penalty
            self.regularization = 'l2'
        else:
            raise ValueError('regularization can be "l2" or "wl2" not "{}"'.format(regularization))
        
        self.train_data = train_data
        self.users_len = int(max(train_data.df[train_data.user_col_name]) + 1)
        self.items_len = int(max(train_data.df[train_data.item_col_name]) + 1)
        self.features = features
        self.lmbda = lmbda
        # create and init matrices U and M
        self.U = np.random.randn(self.features, self.users_len)
        self.M = np.random.randn(self.features, self.items_len)
        self.U, self.M = self.init_U(), self.init_M()

        user_groups = train_data.df.groupby(train_data.user_col_name)[train_data.item_col_name]
        self.user_I = defaultdict(lambda : np.array([]))
        for user in sorted(user_groups.groups):
            self.user_I[user] = user_groups.get_group(user).values
        
        
        item_groups = train_data.df.groupby(train_data.item_col_name)[train_data.user_col_name]
        self.item_I = defaultdict(lambda : np.array([], dtype=int))        
        for item in sorted(item_groups.groups):
            self.item_I[item] = item_groups.get_group(item).values
        
    def fit(self, test_data,
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
        Losses = namedtuple('Losses', ['train_losses', 'test_losses'])
        train_losses = []
        test_losses = []
        last_loss = None
        
        train_start_time = time.time()

        for epoch in range(epochs):
            self.step()
            self.train_data = self.predict(self.train_data)
            loss = reg_se(self.train_data)
            
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

    def step(self):
        """
        Perform single optimization step op algorithm.
        """
        compute_U = ComputeMatrix(self.create_compute_u, self.features, self.users_len)
        compute_M = ComputeMatrix(self.create_compute_m, self.features, self.items_len)

        self.U = compute_U.compute_parallel()
        self.M = compute_M.compute_parallel()

    def predict(self, data):
        """
        Predicted rating for each (user, item) indecies pair in users, items

        data : RatingData
            rating rata
            
        return : np.array
            predicted ratings
        """
        ratings_hat = []
        
        for user, item in zip(data.df[data.user_col_name], data.df[data.item_col_name]):
            try:
                rating_hat = np.dot(self.U[:, user], self.M[:, item])
            except IndexError:
                logging.warning('invalid ids pair ({}, {}) in user_item_iterable'.format(user, item))
                rating_hat = np.nan
            ratings_hat.append(rating_hat)

        rating_hat = np.array(ratings_hat)
        
        data.df[data.prediction_col_name] = rating_hat
        return data

    def weighted_reg_penalty(self):
        """
        Compute weighted lambda regularization penalty
        
        return : float 
            weighted lambda regularization penalty
        """
        # Tikhonov regularization matrices
        Gu = self.U @ sps.diags([len(self.user_I[i]) for i in self.user_I])
        Gm = self.M @ sps.diags([len(self.item_I[i]) for i in self.item_I])
        
        penalty = self.lmbda * ((Gu**2).sum() + (Gm**2).sum())
        return penalty
    
    def l2_penalty(self):
        """
        Compute l2 regularization penalty
        
        return : float 
            l2 penalty
        """
        penalty = self.lmbda * ((self.U**2).sum() + (self.M**2).sum())
        return penalty
    
    def init_M(self):
        """
        Initialize matrix M so that it first row contain item's means
        
        return : np.array
            initilized matrix M
        """
        col_R = sps.csc_matrix((self.train_data.df[self.train_data.rating_col_name], 
                                (self.train_data.df[self.train_data.user_col_name], 
                                 self.train_data.df[self.train_data.item_col_name])))
        M = self.M.copy()
        M[0, :] = col_means_nonzero(col_R)
        return M
    
    def init_U(self):
        """
        Initialize matrix U so that it first row contain user's means
        
        return : np.array
            initilized matrix U
        """
        row_R = sps.csr_matrix((self.train_data.df[self.train_data.rating_col_name], 
                                (self.train_data.df[self.train_data.user_col_name], 
                                 self.train_data.df[self.train_data.item_col_name])))
        U = self.U.copy()
        U[0, :] = row_means_nonzero(row_R)
        return U
  
    def create_compute_u(self):
        """
        Create function to compute column of U by it's index
        
        return : function
            function to compute columns of U
        """
        M = self.M
        row_R = sps.csr_matrix((self.train_data.df[self.train_data.rating_col_name], 
                                (self.train_data.df[self.train_data.user_col_name], 
                                 self.train_data.df[self.train_data.item_col_name])))
        user_I = self.user_I
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
            M_i = M[:, I_i]
            A_i = M_i @ M_i.T + lmbda * n_i * E
            V_i = M_i @ row_R[i, I_i].T
            u_i = np.linalg.solve(A_i, V_i)
            return u_i.flatten()
        
        return compute_u
    
    def create_compute_m(self):
        """
        Create function to compute column of M by it's index
        
        return : function
            function to compute columns of M
        """
        U = self.U        
        col_R = sps.csc_matrix((self.train_data.df[self.train_data.rating_col_name], 
                                (self.train_data.df[self.train_data.user_col_name], 
                                 self.train_data.df[self.train_data.item_col_name])))
        item_I = self.item_I
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
            n = n = len(I_j) if self.regularization == 'wl2' else 1
            U_j = U[:, I_j]
            A_j = U_j @ U_j.T + lmbda * n * E
            V_j = U_j @ col_R[I_j, j]
            m_j = np.linalg.solve(A_j, V_j)
            return m_j.flatten()
        return compute_m