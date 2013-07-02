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
            self.Images = self.hdf5.root.Images
            self.Regions = self.hdf5.root.Regions        
            self.Descriptors = self.hdf5.root.Descriptors
            self.reg_last = self.Regions.attrs.last_index        
            self.des_last = self.Descriptors.attrs.last_index
            # gathering a list of tasks
            for row in self.Images.where("reg_count == -1"):
                self.tasks.append((row["index"], row["file_name"]))
        else:
            self.img_data = []
            imglist = cPickle.load(open(cf._img_data, "rb"))
            for item in imglist:
                # here using ("url", "filename") instead of "index" in hdf5,
                # because we need both filename and url later on
                self.tasks.append(((item[1], item[0]), item[0]))       
        
        # initializing reporting part
        self.task_max = len(self.tasks)
        self.task_curr = self.task_max


    def __del__(self):
        if cf._mode == "hdf5":
            self.hdf5.close()
            
        
    def get_new_task(self):
        # just yielding tasks here
        for idx, img_file in self.tasks:
            yield ("csift", img_file, idx)

    
    def _process_result(self, result, flush):
        """Demo mode, save output as is.
        """
        data, (url, filename) = result
        for i in xrange(len(data["regions"])):
            r = data["regions"][i]
            r.append(url)  # hide url in parameters list
            d = data["descriptors"][i]
            self.img_data.append([filename, r, d])

        if flush:
            cPickle.dump(self.img_data, open(cf._img_data, "wb"), -1)     
    
    
    def _process_result_hdf5(self, result, flush):
        """Batch mode, write output to HDF5 file.
        """
        data, idx = result

        # updating image record
        nregs = len(data["regions"])
        for irow in self.Images.iterrows(idx, idx+1):
            irow["reg_count"] = nregs        
            if nregs > 0:
                irow["reg_first"] = self.reg_last + 1
            irow.update()
                                      
        # writing regions and descriptors            
        for i in xrange(len(data["regions"])):
            rrow = self.Regions.row
            rrow["index"] = self.reg_last + i + 1
            rrow["img_class"] = irow["classN"]
            rrow["img_site"] = irow["site_index"]
            rrow["img_index"] = idx
            rrow["center"] = data["regions"][i][1]            
            rrow["radius"] = data["regions"][i][2]            
            rrow["cornerness"] = data["regions"][i][3]            
            rrow.append()
            
            drow = self.Descriptors.row
            drow["index"] = self.des_last + i + 1
            drow["classN"] = irow["classN"]
            drow["data"] = data["descriptors"][i]
            drow.append()
        
        # updating parameters
        self.reg_last += nregs
        self.des_last += nregs
        
        # writing data
        self.Regions.attrs.last_index = self.reg_last
        self.Descriptors.attrs.last_index = self.des_last
        if flush:
            self.Images.flush()
            self.Regions.flush()
            self.Descriptors.flush()
    
    
    def process_result(self, result, flush):
        """Choose between batch and demo mode.        
        """
        self.task_curr -= 1
        if cf._mode == "hdf5":
            self._process_result_hdf5(result, flush)
        else:
            self._process_result(result, flush)
        
        

def calc_descr(kill_workers):
    if cf._show_progress:
        print "Extracting descriptors..."
    master = Master(kill_workers)
    master.start()
    master.join()











