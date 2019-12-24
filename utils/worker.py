#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: erigara
"""

def template_worker(worker_func, queue):
    """
    Template implementation of worker loop

    worker_func : function
        function called when worker get item from queue
    
    queue: queue.Queue
        queue used to comunicate with worker
    """
    while True:
        item = queue.get()
        if item is None:
            break
        worker_func(item)
        queue.task_done()