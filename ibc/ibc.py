# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:02:59 2013

@author: akusoka1
"""

from ibc_config import IBCConfig as cf
from modules.hdf5_creator import init_hdf5
from modules.img_preprocessor import normalize_images
from modules.get_descriptors import calc_descr
from modules.get_nn import calc_nn
from modules.img_repr import get_repr
from modules.classifier import train_elm, run_elm
from mp.mp_support import *
from mp.mp_worker import MPWorker
import sys
import cProfile

##############################################################################
print "IBC started"
#profiler = cProfile.Profile()
#profiler.enable()

mp_start()
init_hdf5()
normalize_images()
calc_descr(kill_workers=False)
mp_boost_nn()
calc_nn(kill_workers=True)
mp_finalize()
get_repr(all_repr=False)
#train_elm()
run_elm(save_txt=True)

#profiler.disable()
#profiler.dump_stats(cf._dir + "try.prof")
print "IBC finished"
##############################################################################
sys.exit()  # suicide


