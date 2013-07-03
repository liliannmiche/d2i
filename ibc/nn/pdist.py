"""Computes pairwise distances.

Currently, 'cdist()' function from scipy is the fastest, although it uses only
one core. Using np.sum(np.power(dist,2),2) is 10x slower.
"""
from ibc_config import IBCConfig as cf
import cPickle
import numpy as np
import os
import scipy.spatial.distance as distance
from bottleneck import argpartsort  # finding k smallest elements fast


class PDist(object):
    """Implementation of pairwise distance calculation.

    Distance function is set to squared euclidean. No results for
    other functions yet, but they can do as well.

    Eats something like 30% of the total image classification time.
    Fast parallel BLAS or GPU implementation would be beneficial.
    """

    def __init__(self):
        """Initialize object with constant centroids.
        """        
        if os.path.isfile(cf._C_file):
            self.C = cPickle.load(open(cf._C_file, "rb"))["C"]
        else:  
            # centroids have not been initialized yet
            self.C = None


    def get_1nn(self, D):
        """Get indices of the first nearest neighbours.
        
        Transferring back indices only is faster that
        the whole distance matrix.
        """
        if self.C is None:
            return "Uninitialized centroids"
        
        dist = distance.cdist(D, self.C, 'sqeuclidean')
        inds = np.argmin(dist, 1)
        l = range(len(inds))
        dsts = dist[l, inds]
        return (inds, dsts)


    def get_knn(self, D):
        """Get indices of the first nearest neighbours.
        
        Transferring back indices only is faster that
        the whole distance matrix.
        """
        if self.C is None:
            return "Uninitialized centroids"

        
        dist = distance.cdist(D, self.C, 'sqeuclidean')
        L = D.shape[0]  # number of samples
        k = cf._nn_count  # number of nearest neighbours
        k_idx = argpartsort(dist, k, 1)[:, :k]  # getting K smallest indices, unordered
        k_dist = dist[[[t]*k for t in xrange(L)], k_idx]  # get distances, unordered
        idx = np.argsort(k_dist, 1)  # get correct ordering
        k_dist = k_dist[[[t]*k for t in xrange(L)], idx]  # apply ordering to distances
        k_idx = k_idx[[[t]*k for t in xrange(L)], idx]  # apply ordering to indices
        
        return (k_idx, k_dist)

