# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:23:12 2013

@author: akusoka1
"""

from ibc_config import IBCConfig as cf
from mp_worker import MPWorker
from subprocess import Popen
import time

wrk = []
mng = None

def mp_start():
    # starting mutiprocessing manager
    global mng
    global wrk
    mng = Popen(["python", cf._ibc + "mp/mp_manager.py"])
    mng.poll()
    time.sleep(1)  # time to create "hostname.txt" file
    # starting parallel workers
    for i in xrange(cf._nr_wrk/2):
        wrk.append(MPWorker(i))
        wrk[-1].start()

def mp_boost_nn(): 
    # add workers for NN here, because
    # too many workers crash feature extraction
    global wrk
    for i in xrange(cf._nr_wrk - cf._nr_wrk/2):  # rest of the workers
        wrk.append(MPWorker(i))
        wrk[-1].start()

def mp_finalize():
    # terminating multiprocessing part
    for w in wrk:
        w.join()
    mng.terminate()
