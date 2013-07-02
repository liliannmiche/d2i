# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:58:26 2013

@author: akusoka1
"""

from ibc_config import IBCConfig as cf
from tables import openFile
import cPickle
import numpy as np
import os


class Repr(object):
    """Separate class implementing representation calculations.
    """
    
    def __init__(self):
        C = cPickle.load(open(cf._C_file, "rb"))
        self.C = C["C"]
        self.L_mj = C["L_majority"]  # majority vote labels
        self.L_soft = C["L_soft"]  # soft labels

    def _repr(self, c, d):
        """Get the whole representation as a vector.
        
        c = indexes of neighbours
        d = distances to neighbours
        """
        l1 = self.L_mj.shape[1]
        l2 = self.L_soft.shape[1]
        result = np.zeros((l1 + l2, ), dtype=np.float64)
        for c1 in c:
            result[:l1] += self.L_mj[c1]
            result[l1:] += self.L_soft[c1]
        # divide by the amount of samples
        if c.shape[0] > 0:
            result = result / c.shape[0]
        return result
        

##############################################################################
# working funcitons


def get_repr(all_repr=False):          
    if cf._mode == "hdf5":
        _get_repr_hdf5(all_repr)
    else:
        _get_repr()


def _get_repr():
    """Get image representation from "_img_data" file.
    """
    obj = Repr()
    data = cPickle.load(open(cf._img_data, "rb"))


    img = list(set([d[0] for d in data]))  # get a list of unique image names
    repr1 = [[[],[]]] * len(img)  # repr[image_index][c,d][region]
    ws = [[]] * len(img)  # list of websites
    for d in data:
        imgidx = img.index(d[0])
        ws[imgidx] = d[1][-1]
        c, d = d[3]
        repr1[imgidx][0].append(c[0])
        repr1[imgidx][1].append(d)

    repr2 = []
    for i in xrange(len(repr1)):
        c = np.asarray(repr1[i][0])
        d = np.asarray(repr1[i][1])
        repr2.append([ws[i], obj._repr(c, d)])

    cPickle.dump(repr2, open(cf._img_data, "wb"), -1)


def _get_repr_hdf5(all_repr):
    """Filling representation of images from the given websites.
    """
    obj = Repr()
    db = openFile(cf._hdf5, "a")
    Websites = db.root.Websites
    Images = db.root.Images
    Regions = db.root.Regions

    images = []    
    if all_repr:  # recalculate everything
        for row in Images.iterrows():
            images.append([row["index"], row["reg_first"], row["reg_count"]]) 
    else:  # recalculate just new ones
        for img in open(cf._ws_descr).readlines():
            url = line.split(";")[1]
            wsidx = Websites.readWhere('url == "%s"'%url, field="index")[-1]
            imgrows = Images.readWhere('site_index == %d'%wsidx)
            for row in imgrows:
                images.append([row["index"], row["reg_first"], row["reg_count"]]) 

    for img in images:
        data = Regions.read(img[1], img[1]+img[2], field="neighbours")
        c = data[:, 0, 0]  # first of the neighbours
        d = data[:, 1, :]  # all distances
        r = obj._repr(c, d)  # obtain representation
        Images.modify_column(img[0], colname="img_repr", column=r)
        
    Images.flush()
    db.close()









































