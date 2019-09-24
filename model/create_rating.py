#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 23:36:23 2019

@author: erigara
"""

import numpy as np


def create_rating(users_len, items_len):
    array = np.random.choice([0, 0, 0, 0, 0, 1, 2, 3, 4, 5], size=(users_len, items_len))
    users = []
    items = []
    ratings = []
    for i in range(users_len):
        for j in range(items_len):
            rating = array[i, j]
            if rating > 0:
                users.append(i)
                items.append(j)
                ratings.append(rating)
                
    return array, users, items, ratings