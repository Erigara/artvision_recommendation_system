#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara
"""
from collections import namedtuple

# define data type
RatingData = namedtuple('RatingData', ['df', 
                                       'user_col_name', 
                                       'item_col_name',
                                       'rating_col_name',
                                       'prediction_col_name',
                                       'timestamp_col_name'])