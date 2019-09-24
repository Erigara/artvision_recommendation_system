#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 00:46:07 2019

@author: erigara
"""
import pandas as pd
from model.baseline_predictor import BaselinePredictor
from model.als_predictor import ALS

rating_df = pd.read_csv('../data/ratings.csv')
users = rating_df['userId']
items = rating_df['movieId']

mean = rating_df['rating'].mean()
user_means = rating_df.groupby('userId')['rating'].mean()
item_means = rating_df.groupby('movieId')['rating'].mean()

bp = BaselinePredictor(mean, user_means, item_means, trunc=True)

rating_df['baseline_rating'] = bp.predict(zip(users, items))
rating_df['error'] = rating_df['rating'] - rating_df['baseline_rating']

als_error  = ALS(users, items, rating_df['error'], features=4, reg_weight=0.001)
als_rating = ALS(users, items, rating_df['rating'], features=4, reg_weight=0.001)

als_error.fit(100)
als_rating.fit(100)

rating_df['pure_als_rating']   = als_rating.predict(users, items)
rating_df['error_prediction']  = als_error.predict(users, items)
rating_df['combine_rating']    = rating_df['baseline_rating'] +  rating_df['error_prediction']

rating_df = pd.read_csv('../data/ratings.csv')