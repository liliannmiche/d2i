# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:01:26 2013

@author: akusoka1
"""

from ibc_config import IBCConfig as cf
from sift.csift_extractor import csift
from nn.pdist import PDist
from multiprocessing.managers import BaseManager
from multiprocessing import Process
import time


# existing class has connection, extending to add useful functions
class QueueManager(BaseManager):
    """Will combine multiprocessing manager and queue system.
    """
    pass


class MPWorker(Process):
    """Worker template, has a temporary directory.
    """

    def __init__(self, wrk_idx):
        super(MPWorker, self).__init__()
        host_ip = open(cf._host, "r").read()
        
        QueueManager.register('qtask')
        QueueManager.register('qresult')
        mng = QueueManager(address=(host_ip, cf._port), authkey=cf._key)
        mng.connect()
        self.qtask = mng.qtask()
        self.qresult = mng.qresult()
        self.idx = wrk_idx  # for creating unique temp directory


    def run(self):
        pd = PDist()
        while True:
            task = self.qtask.get(block=True)
            if task == "empty": 
                # close worker if job is done
                self.qresult.put("done", block=True)
                self.qtask.task_done()
                return
            
            if task[0] == "csift":
                result = (csift(task[1], self.idx), task[2])
            elif task[0] == "nn":
                # task[2] are start and stop indices of descriptors
                result = (pd.get_knn(task[1]), task[2])
            else:
                result = "Unknown task type"
                
            # submitting results            
            self.qresult.put(result, block=True)
            self.qtask.task_done()
            time.sleep(0.01)







