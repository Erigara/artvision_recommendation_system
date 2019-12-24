#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module implement model fitter that used to 
train model iteratively
"""
import queue
import threading
import os

from utils.serializer import save_model, load_model
from utils.worker import template_worker
from model.als_predictor import ALS
from model.als_trainer import ALSTrainer

class ModelFitter:
    def __init__(self,  model_path, fit_params, model_params, trainer_params, model_class, trainer_class):
        self.model_path = model_path
        self.fit_params = fit_params
        self.model_params = model_params
        self.trainer_params = trainer_params
        self.model_class = model_class
        self.trainer_class = trainer_class
        
        self.fitter = threading.Thread(target=self.fitter_worker)
        self.fitter_queue = queue.Queue()

    @property
    def model_path(self):
        """
        Path to model's file
        """
        return self._model_path

    @model_path.setter
    def model_path(self, path):
        if os.path.exists(path) or os.access(os.path.dirname(path), os.W_OK):
            self._model_path = path
        else:
            raise ValueError(f'Unvalid path: {path} is given!')
    
    @property
    def fit_params(self):
        """
        Parameters used to fit model
        """
        return self._fit_params

    @fit_params.setter
    def fit_params(self, params):
        if isinstance(params, dict):
            self._fit_params = params
        else:
            raise ValueError(f'Params must be dict, but {params} is given!')

    @property
    def model_params(self):
        """
        Parameters used to create new model
        """
        return self._model_params

    @model_params.setter
    def model_params(self, params):
        if isinstance(params, dict):
            self._model_params = params
        else:
            raise ValueError(f'Params must be dict, but {params} is given!')

    @property
    def trainer_params(self):
        """
        Parameters used to create trainer
        """
        return self._trainer_params

    @trainer_params.setter
    def trainer_params(self, params):
        if isinstance(params, dict):
            self._trainer_params = params
        else:
            raise ValueError(f'Params must be dict, but {params} is given!')

    @property
    def model_class(self):
        """
        Class of newly created model
        """
        return self._model_class

    @model_class.setter
    def model_class(self, clss):
        if isinstance(clss, type):
            self._model_class = clss
        else:
            raise ValueError(f'Model class must by class, but {clss} is given!')
    
    @property
    def trainer_class(self):
        """
        Class of newly created model
        """
        return self._trainer_class

    @trainer_class.setter
    def trainer_class(self, clss):
        if isinstance(clss, type):
            self._trainer_class = clss
        else:
            raise ValueError(f'Trainer class must by class, but {clss} is given!')

    def start(self):
        """
        Activate fitter
        Can be started only once
        """
        self.fitter.start()
    
    def stop(self):
        """
        Deactivate fitter
        """
        self.fitter_queue.put(None)
        self.fitter_queue.join()
        
    def pipeline(self, data):
        """
        Load model -> Train model -> Save model
        
        data : tuple
            tuple of train RatingData and test RatingData
        """
        # load model if exist
        model = load_model(self.model_path)
        if model is None:
            model = self.model_class(**self.model_params)
        
        # train model
        train_data, test_data = data
        trainer = self.trainer_class(model, **self.trainer_params)
        trainer.fit(train_data, test_data, **self.fit_params)
        # do something with losses
        
        # save model
        save_model(model, self.model_path)

    def fitter_worker(self):
        """
        Fitter worked loop
        """
        template_worker(self.pipeline, self.fitter_queue)
        
        
    def recieve_data(self, data):
        """
        Put train and test data in transmitter queue
        
        data : tuple
            tuple of train RatingData and test RatingData
        """
        train_data, test_data = data
        self.fitter_queue.put((train_data, test_data))
    
class ALSFitter(ModelFitter):
    """
    Class to train ALS model iteratively
    """
    def __init__(self,  model_path, fit_params, model_params, trainer_params):
        super().__init__(model_path, fit_params, model_params, trainer_params,
                         ALS, ALSTrainer)