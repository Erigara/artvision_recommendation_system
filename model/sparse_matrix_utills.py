#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utill functions to deal with sparse matrix

@author: erigara
"""

import scipy.sparse as sps
import numpy as np

import logging
import time
logging.basicConfig(level = logging.DEBUG)



def row_means_nonzero(inp_matrix):
    """
    return mean for every row in inp_matrix excluding zero elements
    
    inp_matrix : sps.spmatrix
    """
    matrix = inp_matrix.tocsr()
    nonzero_rows = np.diff(matrix.indptr)
    sums = np.array(matrix.sum(axis=1)).flatten()
    means = np.nan_to_num(sums / nonzero_rows)
    return means

def col_means_nonzero(inp_matrix):
    """
    return mean for every column in inp_matrix excluding zero elements
    
    inp_matrix : sps.spmatrix
    """
    matrix = inp_matrix.tocsc() 
    nonzero_cols = np.diff(matrix.indptr)
    sums = np.array(matrix.sum(axis=0)).flatten()
    means = np.nan_to_num(sums / nonzero_cols)
    return means

def norm2(matrix):
    return matrix.power(2).sum()
        
#TODO gpu speed up
def block_mat_mult(A, B, block_size=1024, mask_function=None):
    """
    Compute sparse C = A * B by blocks as 
    C[i : i + block_size , j : j + block_size] = A[i : i + block_size, :] * B[:, j : j + block_size]
    
    A : np.array (n, f)
        dense matrix
    B : np.array (f, m)
        dense matrix
    block_size : int
    
    mask_function : function
        function to make submatrix (C_block) sparse
    """
    if A.shape[1] != B.shape[0]:
        raise Warning('A 2 dim {} != B 1 dim {}'.format(A.shape[1], B.shape[0]))
    
    n = A.shape[0]
    m = B.shape[1]
    C = sps.lil_matrix((n, m))
    mask_func_time = 0
    assigment_time = 0
    mult_time = 0
    for i in range(0, n, block_size):
        for j in range(0, m, block_size):

            mult_time_start = time.time()
            C_block =  A[i : i + block_size, :] @ B[:, j : j + block_size]
            mult_time_end = time.time()
            
            hight, width = C_block.shape
            if mask_function:
                mask_func_start = time.time()
                C_block = mask_function(C_block, i, j)
                mask_func_end = time.time()
            
            # TODO bottleneck operation
            assigment_time_start = time.time()
            C[i : i + hight, j : j + width] = C_block
            assigment_time_end = time.time()
            
            mult_time += mult_time_end - mult_time_start
            assigment_time += assigment_time_end - assigment_time_start
            mask_func_time += mask_func_end - mask_func_start
            
    C = C.tocsc()
    logging.debug("MULT TIME : {}".format(mult_time))
    logging.debug("ASSIGMENT TIME : {}".format(assigment_time))
    logging.debug("MUSK FUNC TIME : {}".format(mask_func_time))
    return C
    
    