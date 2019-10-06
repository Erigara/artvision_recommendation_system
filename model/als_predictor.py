#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module implement rating predictor based on ALS matrix decomposition
"""

import numpy as np
import pandas as pd
from scipy import sparse as sps
from model.utils.sparse_matrix_operations import (row_means_nonzero, 
                                                  col_means_nonzero)
from model.utils.metrics import reg_rmse, reg_se
from model.utils.compute_matrix import ComputeMatrix
from model.utils.create_rating import create_rating_matrix

import logging
import time
logging.basicConfig(level = logging.INFO)


class ALS:
    """
    Find decomposition of matrix R onto matrices U, M so that R ~ U.T * M
    
    Implement Realization of ALS with Weighted-λ-Regularization from
    "Large-scale Parallel Collaborative Filtering for the Netflix Prize"   
    """
    def __init__(self, users, items, ratings, features=10, regularization='l2', lmbda = 0.3):
        """
        users : arraylike of int
            users id
    
        items : arraylike of int
            items id
        
        ratings : arraylike of float
        
        features : int
            number of hidden features
            
        regularization : str
            'l2' for l2 regularization
            'wl2' for weighted lambda regularization
        
        lmbda : float
            regularization weight
        
        !!! To work properly ids must start form 0.
        """
        if not (len(users) == len(items) == len(ratings)):
            raise ValueError('users, items, ratings columns must have equal len')
        
        if regularization == 'wl2':
            self.reg_penalty = self.weighted_reg_penalty
            self.regularization = 'wl2'
        elif regularization == 'l2':
            self.reg_penalty = self.l2_penalty
            self.regularization = 'l2'
        else:
            raise ValueError('regularization can be "l2" or "wl2" not "{}"'.format(regularization))
        
        self.users = users
        self.items = items
        self.ratings = ratings
        
        self.users_len, self.items_len = int(max(users) + 1), int(max(items) + 1)
        
        self.features = features
        self.lmbda = lmbda
        # create matrices U and M
        self.U = np.random.randn(self.features, self.users_len)
        self.M = np.random.randn(self.features, self.items_len)
        
        self.user_I = [[] for i in range(self.users_len)]
        self.item_I = [[] for j in range(self.items_len)]
        for (user, item) in zip(users, items):
            self.user_I[user].append(item)
            self.item_I[item].append(user)
    
    def fit(self, epochs=5, eps=0.001):
        """
        Perform iterative optimization algorithm to find matrices M, U so that
        they minimize loss function
        
        epochs : int 
            number of iterations in algorithm
        
        eps : float
            algorithm stops if difference between last_loss and loss smaller than eps
        """
        last_loss = None
        train_start_time = time.time()
        
        self.U, self.M = self.init_U(), self.init_M()
        
        compute_U = ComputeMatrix(self.create_compute_u, self.features, self.users_len)
        compute_M = ComputeMatrix(self.create_compute_m, self.features, self.items_len)
        
        for epoch in range(epochs):
            self.U = compute_U.compute_parallel()
            self.M = compute_M.compute_parallel()
            # compute loss
            ratings_hat = self.predict(self.users, self.items)
            loss = reg_se(self.ratings, ratings_hat, self.reg_penalty())
            
            if last_loss and abs(loss - last_loss) < eps:
                last_loss = loss
                break
            if epoch % (epochs / 10) == 0:
                logging.info("\n========== Epoch {} ==========".format(epoch))
                logging.info("Train loss: {}".format(loss))
                if last_loss:
                    logging.info("Loss delta: {}".format(loss - last_loss))
                if last_loss and last_loss < loss:
                    logging.warning("WARNING - Loss Increasing") 
                    
            last_loss = loss
        
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        
        logging.info("\n========== Train complete! ==========")
        logging.info("Epochs: {}".format(epoch))
        logging.info("Train loss: {}".format(loss))
        logging.info("Totall train time : {:.6f}  sec.".format(train_time))
            
            
    def weighted_reg_penalty(self):
        """
        Compute weighted lambda regularization penalty
        
        return : float 
            weighted lambda regularization penalty
        """
        # Tikhonov regularization matrices
        Gu = self.U @ sps.diags([len(i) for i in self.user_I])
        Gm = self.M @ sps.diags([len(i) for i in self.item_I])
        
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
    
    def predict(self, users, items):
        """
        Predicted rating for each (user, item) indecies pair in users, items
        
        users : arraylike of int
            users id
            
        items : arraylike of int
            items id
    
        return : np.array
            predicted ratings
        """
        rating_hat = np.array([np.dot(self.U[:, user], self.M[:, item]) for user, item in zip(users, items)])
        return rating_hat
    
    def init_M(self):
        """
        Initialize matrix M so that it first row contain item's means
        
        return : np.array
            initilized matrix M
        """
        col_R = sps.csc_matrix((self.ratings, (self.users, self.items)))
        M = self.M.copy()
        M[0, :] = col_means_nonzero(col_R)
        return M
    
    def init_U(self):
        """
        Initialize matrix U so that it first row contain user's means
        
        return : np.array
            initilized matrix U
        """
        row_R = sps.csr_matrix((self.ratings, (self.users, self.items)))
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
        row_R = sps.csr_matrix((self.ratings, (self.users, self.items)))
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
        col_R = sps.csc_matrix((self.ratings, (self.users, self.items)))
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