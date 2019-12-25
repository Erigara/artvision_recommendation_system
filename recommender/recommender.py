#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module implement recommender class used to get user's recommendations
"""
import time
from watchdog.observers import Observer
from watchdog.events import RegexMatchingEventHandler
from utils.serializer import load_model
from recommendations import make_recommendation_for_user

class ModelUpdateHandler(RegexMatchingEventHandler):
    def __init__(self, update_model):
        super().__init__(regexes=[r'.*\.mdl'], ignore_directories=True)
        self.update_model = update_model
    
    def on_modified(self, event):
        self.update_model(event.src_path)

class Recommender:
    def __init__(self, initial_model_path, model_storage_path, db,  min_records=10):
        observer = Observer()
        observer.schedule(ModelUpdateHandler(self.update_model), 
                          path=model_storage_path, recursive=False)
        self.db = db
        self.min_records = min_records
        self.model = load_model(initial_model_path)
        self.cache = {}
    
    def update_model(self, model_path):
        model = load_model(model_path)
        self.model = model
        # clear old model cache
        self.cache.clear()
        self.cache['defaultuser'] = self.db.get_item_avg_ratings(min_records=10)
    
    def make_recommendations(self, user_id, structure):
        if user_id in self.cache:
            predictions = self.cache[user_id]
        else:
            predictions = make_recommendation_for_user(self.model, user_id)
            self.cache[user_id] = predictions

        if predictions is None:
                predictions = self.cache['defaultuser']

        for category in structure:
            item_ids = structure[category]['id']
            recommendations = []
            for item_id in item_ids:
                recommendations.append(predictions.get(item_id, 0))
            structure[category]['recommendation'] = recommendations
        return structure