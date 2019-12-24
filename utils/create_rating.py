#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara


"""

import numpy as np
from scipy import sparse as sps

def create_rating_matrix(users_len, items_len, scale=5, min_rating=1, increment=1, sparsity=0.9):
    """
    Create sparse rating matrix (user_len, item_len) 
    so that every row/column has at least 1 nonzero rating
    
    users_len : int
        number of rows in matrix 
    
    items_len : int
        number of columns in matrix
    
    scale : int
        totall number of possible ratings
        
    min_rating : float
        smallest posible rating
        
    increment : float
        distanse between rating(i) and rating(i+1)
    
    sparsity : float
        ratio of 0 in matrix
        if sparsity > 1 / min(users_len, items_len)
        than sparsity = 1 / min(users_len, items_len)
    
    return : sps.coo_matrix
        rating matrix
    """
    if users_len <= 0 or items_len <= 0:
        raise ValueError('number of rows/cols must be positive')
    if min_rating <= 0:
        raise ValueError('minimum rating must be positive')
    
    # set minimum possible sparsity 
    sparsity = min(1 - 1 / min([users_len, items_len]), sparsity)
        
    row_permutations = np.random.permutation(users_len)
    col_permutations = np.random.permutation(items_len)
    
    row_repeat = np.tile(row_permutations, items_len)
    col_repeat = np.tile(col_permutations, users_len)
    
    n = int(np.ceil((1- sparsity) * users_len * items_len))
    
    users = list(row_repeat[:n])
    items = list(col_repeat[:n])
    possible_ratings = [min_rating + increment * i for i in range(scale)]
    ratings = list(np.random.choice(possible_ratings, (n,)))
    
    return users, items, ratings