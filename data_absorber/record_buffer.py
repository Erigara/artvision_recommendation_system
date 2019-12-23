#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara

Module implement record buffer used to store records with possibility to bind events on some actions
"""

class RecordBuffer:
    """
    RecordBuffer class used to temporary store records, 
    on full buffer call callback function and clear buffer
    """
    def __init__(self, buffer_size, on_full_buffer):
        """        
        buffer_size : int
            max buffer size
        on_full_buffer : function
            callback called when buffer_size = max_size
            must accept list that collect elements buffer as argument    
            should be non blocking
        """
        
        self._buffer = []
        self.on_full_buffer = on_full_buffer
        self.buffer_size = buffer_size
        
    @property
    def buffer_size(self):
        """
        Variable controlling buffer size
        """
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value):
        if value > 0:
            self._buffer_size = value
        else:
            raise ValueError(f'Buffer size must be positive, but {value} is given')
        
    @property
    def on_full_buffer(self):
        """
        Callback called when buffer is full
        must accept list that collect elements buffer as argument 
        """
        return self._on_full_buffer

    @on_full_buffer.setter
    def on_full_buffer(self, callback):
        if callback.__code__.co_argcount == 1:
            self._on_full_buffer = callback
        else:
            raise ValueError(f'Callback function must take only 1 argument')
        

    def add_record(self, user_id, item_id, rating, timestamp):
        """
        Add record to buffer
        
        user_id : obj
            user id
            
        item_id : obj
            item_id
            
        rating : float
            rating that user with user_id give to item with item_id
            
        timestamp : int
            rating timestamp

        """
        if len(self._buffer) < self._buffer_size:
            self._buffer.append([user_id, item_id, rating, timestamp])
            
        if len(self._buffer) == self._buffer_size:
            self._on_full_buffer(self._buffer)
            # clear buffer
            self._buffer = []
    