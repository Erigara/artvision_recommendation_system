#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module provide functions to deal with sparse matrix
"""

import scipy.sparse as sps
import numpy as np



def row_means_nonzero(inp_matrix):
    """
    Return mean for every row in inp_matrix excluding zero elements
    
    inp_matrix : sps.spmatrix
    
    return : np.array
        array of row's means
    """
    matrix = inp_matrix.tocsr()
    nonzero_rows = np.diff(matrix.indptr)
    nonzero_ids = nonzero_rows != 0
    sums = np.array(matrix.sum(axis=1)).flatten()
    means = sums
    means[nonzero_ids] /= nonzero_rows[nonzero_ids]
    return means

def col_means_nonzero(inp_matrix):
    """
    Return mean for every column in inp_matrix excluding zero elements
    
    inp_matrix : sps.spmatrix
    
    return : np.array
        array of column's means
    """
    matrix = inp_matrix.tocsc() 
    nonzero_cols = np.diff(matrix.indptr)
    nonzero_ids = nonzero_cols != 0
    sums = np.array(matrix.sum(axis=0)).flatten()
    means = sums
    means[nonzero_ids] /= nonzero_cols[nonzero_ids]
    return means        

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
    
    for i in range(0, n, block_size):
        for j in range(0, m, block_size):
            C_block =  A[i : i + block_size, :] @ B[:, j : j + block_size]
            
            hight, width = C_block.shape
            if mask_function:
                C_block = mask_function(C_block, i, j)

            # TODO bottleneck operation
            C[i : i + hight, j : j + width] = C_block
            
    C = C.tocsc()
    return C
    
    