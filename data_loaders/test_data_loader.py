#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: erigara
"""

import pandas as pd


class TestDataLoader:
    def __init__(self, data_path):  
        self.data_path = data_path

    def download_data(self) -> pd.DataFrame:
        """
        download csv file
        """
        with open(self.data_path, 'r') as data_csv:
            data = pd.read_csv(data_csv)
            return data
    

    def upload_data(self, data : pd.DataFrame) -> None:
        """
        upload csv file
        """
        with open(self.data_path, 'w') as data_csv:
            data.to_csv(data_csv)