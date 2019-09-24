#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download/upload data form server

@author: erigara
"""

import requests 
# TODO make logging 


class DataLoader: 
    
    def __init__(self, url : str, port : int):
        """
        set init params for loader
        """
        self.url = url
    
    def download_data(self) -> dict:
        """
        download json file from server
        """
        resonse = requests.get(self.url)
        
        # TODO add response status check
        data = resonse.json()
        
        return data
    

    def upload_data(self, data : dict) -> None:
        """
        upload json file on server
        """
        response = requests.post(self.url, json=data)
        
        # TODO add status check
