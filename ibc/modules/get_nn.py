# -*- coding: utf-8 -*-
"""Processes an image and returns all data (properties + descriptors).

If something goes wrong, returns string describing the error.
@author: akusoka1
"""

from ibc_config import IBCConfig as cf
from mp.mp_master import MPMaster
from tables import openFile
import numpy as np
import cPickle
import os


class Master(MPMaster):
    """Implements "get_new_task" and "process_result" of Master class.
    """

    def __init__(self, kill_workers):
        super(Master, self).__init__(kill_workers, profiling=False)

        self.tasks = []   
        # preparing for job generation
        if cf._mode == "hdf5":
            self.hdf5 = openFile(cf._hdf5, "a")
            self.Regions = self.hdf5.root.Regions        
            self.Descriptors = self.hdf5.root.Descriptors
            # gathering a list of tasks
            for row in self.Regions.iterrows():
                if row["neighbours"][0,0] == -1:
                    self.tasks.append(row["index"])
        else:
            self.img_data = cPickle.load(open(cf._img_data, "rb"))
            self.tasks = range(len(self.img_data))
        
        # initializing reporting part
        self.task_max = len(self.tasks)/cf._nn_batch + 1
        self.task_curr = self.task_max


    def __del__(self):
        if cf._mode == "hdf5":
            self.hdf5.close()
            
    
    def _get_descr(self, batch):
        descrs = []
        if cf._mode == "hdf5":
            descrs = self.Descriptors.read_coordinates(batch, field="data")
        else:
            descrs = np.array([self.img_data[b][2] for b in batch])
        return descrs
        
        
    def get_new_task(self):
        # just yielding tasks here
        batch = []
        for idx in self.tasks:
            batch.append(idx)
            if len(batch) >= cf._nn_batch:
                yield ("nn", self._get_descr(batch), batch)
                batch = []
        # final batch
        yield ("nn", self._get_descr(batch), batch)

    
    def _process_result(self, result, flush):
        """Demo mode, save output as is.
        """
        (idx, dist), batch = result
        # save obtained neighbourhood parameters of each descriptor
        for b in range(len(batch)):
            self.img_data[batch[b]].append((idx[b,:], dist[b,:]))
        
        if flush:
            cPickle.dump(self.img_data, open(cf._img_data, "wb"), -1)     
    
    
    def _process_result_hdf5(self, result, flush):
        """Batch mode, write output to HDF5 file.
        """
        (inds, dist), batch = result
        # updating regions records
        for b in range(len(batch)):
            data = np.vstack((inds[b,:], dist[b,:])).T
            data.dtype = np.float64
            self.Regions.modify_column(batch[b], batch[b]+1,
                                       colname="neighbours", column=data)
        # saving if necessary
        if flush:
            self.Regions.flush()
    
    
    def process_result(self, result, flush):
        """Choose between batch and demo mode.        
        """
        self.task_curr -= 1
        if cf._mode == "hdf5":
            self._process_result_hdf5(result, flush)
        else:
            self._process_result(result, flush)
        
        

def calc_nn(kill_workers):
    master = Master(kill_workers)
    if master.tasks == []:
        print "No neighbours to calculate"
        # killing workers if needed
        if kill_workers:
            master.terminate_workers()
        return
    if cf._show_progress:
        print "Calculating nearest neighbours..."
    master.start()
    master.join()











