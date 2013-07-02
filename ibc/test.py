# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:20:16 2013

@author: akusoka1
"""

import tables
import bottleneck
from bottleneck import argpartsort
from scipy.spatial.distance import cdist
import numpy as np
import time
import sys


print tables.__version__
print bottleneck.__version__

a = np.random.randn(50, 384)
b = np.random.randn(128000, 384)

t0 = time.time()
dist = cdist(a, b, 'sqeuclidean')
print time.time() - t0

k = 3
t0 = time.time()
k_arg = argpartsort(dist, k, 1)[:, :k]  # getting K smallest indices, unordered
k_dist = dist[[[t]*k for t in range(50)], k_arg]  # get distances
idx = np.argsort(k_dist, 1)  # sort them
k_dist = k_dist[[[t]*k for t in range(50)], idx]  # apply sorting to distances
k_arg = k_arg[[[t]*k for t in range(50)], idx]  # apply sorting to arguments
print time.time() - t0
print k_arg[:2,:]
print k_dist[:2,:]

sys.exit()
print 'yolo'

#for i in xrange(10):
#    print dist[[i]*(k+1), cnn[i,:]],
#    print np.min(dist[i,:])