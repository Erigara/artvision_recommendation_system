#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 14:00:32 2019

@author: erigara
"""
import multiprocessing as mp
import ctypes
import numpy as np
import pandas as pd
from scipy import sparse as sps
from model.sparse_matrix_utills import (row_means_nonzero, 
                                        col_means_nonzero, 
                                        norm2, 
                                        block_mat_mult)
class ALS:
    """
    Realization of ALS with Weighted-λ-Regularization from 
    "Large-scale Parallel Collaborative Filtering for the Netflix Prize"   
    """
    def __init__(self, users, items, ratings, features=10, lmbda = 0.3):
        """
        users : object with __getitem__ method
            user ids
    
        items : object with __getitem__ method
            item ids
        
        ratings : object with __getitem__ method
        
        features : int
            number of hidden features
        
        reg_weight : float
            regularization weight
        
        """
        if not (len(users) == len(items) == len(ratings)):
            raise Warning('users, items, ratings columns must have equal len')
        
        self.row_R = sps.csr_matrix((ratings, (users, items)))
        self.col_R = sps.csc_matrix((ratings, (users, items)))
        
        self.users_len, self.items_len = self.col_R.shape
        self.n = len(ratings)
        self.features = features
        self.lmbda = lmbda
        
        self.E = np.eye(features)
        
        np.random.seed(505)
        # create matrices U and M
        self.U = np.random.random((features, self.users_len))
        self.U[0, :] = row_means_nonzero(self.row_R)
        self.M = np.random.random((features, self.items_len))
        self.M[0, :] = col_means_nonzero(self.col_R)
        
        self.user_I = [[] for i in range(self.users_len)]
        self.item_I = [[] for j in range(self.items_len)]
        for (user, item) in zip(users, items):
            self.user_I[user].append(item)
            self.item_I[item].append(user)
            
    def _compute_u(self, i):
        """
        compute column i of matrix U
        
        i : int 
            index of column U
        
        return : np.array with shape (self.features, )
            new column i of matrix U
        """
        I_i = self.user_I[i]
        n = len(I_i)
        M_i = self.M[:, I_i]
        A_i = M_i @ M_i.T + self.lmbda * n * self.E
        V_i = M_i @ self.row_R[i, I_i].T
        u_i = np.linalg.solve(A_i, V_i)
        return u_i.flatten()
    
    def _compute_m(self, j):
        """
        compute column j of matrix M
        
        j : int
            index of column M
        
        return : np.array with shape (self.features, )
            new column j of matrix M
        """
        I_j = self.item_I[j]
        n = len(I_j)
        U_j = self.U[:, I_j]
        A_j = U_j @ U_j.T + self.lmbda * n * self.E
        V_j = U_j @ self.col_R[I_j, j]
        m_j = np.linalg.solve(A_j, V_j)
        return m_j.flatten()
        
    def _compute_matrix_parallel(self, compute_column, length, jobs_per_cpu=1):
        """
        Multiproccess computation of matrix (M or U)
        
        compute_column : function
            function to compute new column of matrix (self._compute_m, self._compute_u)
        
        length : int (self.users_len of self.items_len)
            number of columns in new matrix
        
        return : np.array with shape (self.features, length)
            new matrix (M or U)
        """
        # create shared memory array
        raw_matrix = mp.RawArray(ctypes.c_double, self.features * length)
        
        n_cpu  = mp.cpu_count()
        n_jobs = jobs_per_cpu *  n_cpu
        
        q = length // n_jobs
        r = length % n_jobs
 
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
            worker = mp.Process(target = self._compute_matrix_worker,
                            args = (compute_column, length, queue, raw_matrix))
            workers.append(worker)
            worker.start()
        
        queue.join()
        
        # make array from shared memory    
        matrix = np.reshape(np.frombuffer(raw_matrix), (self.features, length))
        return matrix
    
    def _compute_matrix_worker(self, compute_column, length, queue, raw_matrix):
        """
        Worker to compute columns of matrix
        """
        matrix = np.reshape(np.frombuffer(raw_matrix), (self.features, length))
        
        while True:
            job = queue.get()
            if job == None:
                break
            start, stop = job[0], job[0] + job[1]
            for i in range(start, stop):
                matrix[:, i] = compute_column(i)
            
            queue.task_done()
        queue.task_done()

        
    def _compute_matrix(self, compute_column, length):
        """
        Single process computation of matrix (M or U)
        
        compute_column   : function
            function to compute new column of matrix (self._compute_m, self._compute_u)
        
        length : int (self.users_len of self.items_len)
            number of columns in new matrix
        
        return : np.array with shape (self.features, length)
            new matrix (M or U)
        """
        matrix = np.zeros((self.features, length))
        for i in range(length):
            matrix[:, i] = compute_column(i)
        return matrix
    
    def fit(self, epochs=5, eps=0.001):
        last_loss = None
        for epoch in range(epochs):
            self.U = self._compute_matrix_parallel(self._compute_u,
                                                  self.users_len)
            
            
            self.M = self._compute_matrix_parallel(self._compute_m,
                                                  self.items_len)
            
            loss = self.eval_loss()
            if last_loss and abs(loss - last_loss) < eps:
                last_loss = loss
                break
            last_loss = loss
            if epoch % (epochs / 10) == 0:
                print("\n========== Epoch", epoch,"==========")
                if last_loss and last_loss < loss:
                    print("Train loss: ", loss, "  WARNING - Loss Increasing")
                else:
                    print("Train loss: ", loss)
                last_loss = loss
                print("RMSE={}".format(self.eval_rmse()))
                
                
        print("\n========== Train complete! ==========")
        print("Epochs={}".format(epoch+1))
        print("RMSE={}".format(self.eval_rmse()))
            
            
            
    def eval_loss(self):
        # Tikhonov regularization matrices
        Gu = self.U @ sps.diags([len(i) for i in self.user_I])
        Gm = self.M @ sps.diags([len(i) for i in self.item_I])
        loss = self.eval_rmse() + self.lmbda*((Gu**2).sum() + (Gm**2).sum())
        return loss
    
    def eval_rmse(self, block_size=1024):
        def mask_function():
            # create mask matrix
            Mask = self.col_R.copy()
            Mask[Mask.nonzero()] = 1
            def mask_apply(submatrix, i, j):
                hight, width = submatrix.shape
                submatrix = np.multiply(Mask[i:i+hight, j:j+width].toarray(), submatrix)
                return submatrix
            return mask_apply
        # very slow operation
        R_hat = block_mat_mult(self.U.T, self.M, 
                               block_size=block_size,
                               mask_function=mask_function())
        
        rmse = norm2(self.col_R - R_hat)/self.n
        return rmse
    
    def predict(self, users, items):
        R_hat = block_mat_mult(self.U.T, self.M, block_size=1024)
        return R_hat


    
        
        