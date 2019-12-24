#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara
"""
import multiprocessing as mp
import ctypes
import numpy as np

class ComputeMatrix:
    def __init__(self, create_compute_column, rows, cols):
        """
        create_compute_column   : function
            return function to compute columns of matrix
        
        rows : int 
            number of rows in new matrix
        
        cols : int 
            number of columns in new matrix
        
        """
        self.create_compute_column = create_compute_column
        self.cols = cols
        self.rows = rows
        
    def compute(self):
        """
        Single process computation of matrix
        
        return : np.array with shape (features, length)
            new matrix
        """
        rows = self.rows
        cols = self.cols
        compute_column = self.create_compute_column()
        
        matrix = np.zeros((rows, cols))
        for i in range(cols):
            matrix[:, i] = compute_column(i)
        return matrix
    
    def compute_parallel(self, args=(),  jobs_per_cpu=1):
        """
        Multiproccess computation of matrix (M or U)
        
        return : np.array with shape (features, length)
            new matrix
        """
        
        rows = self.rows
        cols = self.cols
        compute_column = self.create_compute_column(*args)
        
        # create shared memory array
        raw_matrix = mp.RawArray(ctypes.c_double, rows * cols)
        
        n_cpu  = mp.cpu_count()
        n_jobs = jobs_per_cpu *  n_cpu
        
        q = cols // n_jobs
        r = cols % n_jobs
 
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
                            args = (compute_column, rows, cols, queue, raw_matrix))
            workers.append(worker)
            worker.start()
        
        queue.join()
        
        # make array from shared memory    
        matrix = np.reshape(np.frombuffer(raw_matrix), (rows, cols))
        return matrix
    
    def _compute_matrix_worker(self, compute_column, rows, cols, queue, raw_matrix):
        """
        Worker to compute columns of matrix
        
        raw_matrix : mp.RawArray
            shared memory array
        """
        matrix = np.reshape(np.frombuffer(raw_matrix), (rows, cols))
            
        while True:
            job = queue.get()
            if job == None:
                break
            start, stop = job[0], job[0] + job[1]
            for i in range(start, stop):
                matrix[:, i] = compute_column(i)
            queue.task_done()
        queue.task_done()