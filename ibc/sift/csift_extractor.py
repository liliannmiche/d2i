# -*- coding: utf-8 -*-
"""Processes an image and returns all data (properties + descriptors).

If something goes wrong, returns string describing the error.
@author: akusoka1
"""

from ibc_config import IBCConfig as cf
from sift.DescriptorIO import readDescriptors
import numpy as np
import os
import shutil  # to remove directory with subdirectories


def csift(img_file, idx):
    """Calculates and returns descriptors of one image.
    """
    # creating temporary directory, preferably in RAM
    temp_dir = os.path.join(cf._temp_dir, str(idx))
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    
    # extracting descriptors, dropping console output
    # works for LINUX
    temp_file = os.path.join(temp_dir, "descriptors.bin")
    command = ('%s %s --detector harrislaplace --descriptor csift '
               '--output %s --outputFormat binary '
               '--keepLimited %d > /dev/null'
               % (cf._cD_bin, img_file, temp_file, cf._max_reg))
    os.system(command)  # running "colorDescriptors" for one image, one thread

    # reading data        
    regs, descrs = readDescriptors(temp_file)
    data = {}
    data["regions"] = []
    data["descriptors"] = []
    if regs.shape[1] == 5:  # if there are regions
        i = 0
        for region in regs:
            i0 = i  # idx
            i1 = np.int64(region[:2])  # center
            i2 = np.int64(region[2]*8.4853)  # radius
            i3 = np.float64(region[4])  # cornerness
            data["regions"].append([i0, i1, i2, i3])
            i += 1
        for descriptor in descrs:
            descr_uint8 = np.asarray(descriptor, dtype=np.uint8)
            data["descriptors"].append(descr_uint8)    
    
    os.remove(temp_file)  # cleaning temp directory
    return data    
    






























