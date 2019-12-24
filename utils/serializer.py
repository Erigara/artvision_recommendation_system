#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara
"""
import pickle
import os.path


def save_model(model, filename):
    """
    Model object serilization
    
    model : obj
        model to serilize
    
    filename : str
        serilization file name
    """
    with open(filename, 'wb') as output:
        pickled_model = pickle.dumps(model, pickle.HIGHEST_PROTOCOL)
        output.write(pickled_model)

def load_model(filename):
    """
    Model object deserilization
    
    filename : str
        serilization file name
    
    return : model
        deserilized model or None
    """
    if  not os.path.isfile(filename):
        return None
    
    with open(filename, 'rb') as inpt:
        model = pickle.load(inpt)
        return model
    
        