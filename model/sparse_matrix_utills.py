#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utill functions to deal with sparse matrix

@author: erigara
"""

import scipy.sparse as sps
import numpy as np



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
        