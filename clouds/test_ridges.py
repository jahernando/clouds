#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:54:59 2021

@author: hernando
"""

import numpy        as np
#import numpy.linalg as nplang

#import clouds.ridges as ridges
import clouds.clouds as clouds


def generate_image(nbins, ranges, fun):
    """
    
    Generate a image using a function and ranges with bins

    Parameters
    ----------
    nbins  : int or tuple(int), number of bins
    ranges : tuple( (a, b)), range in each axis
    fun    : function(x, y, z)

    Returns
    -------
    bins    : np.array, bins 
    centers : np.array, (x, y, z), 1D arrays center of the bins
    zs      : np.array, z-values matrix z[i, j, k]
              where i, j, k = x, y, z indices
    xmesh   : tuple((x, y, z)) ND-arrays with the meshes
              where x[:, k, k] = x; y[k, :, k] = y, z [k, k, :] = x
    """
    
    size   = len(ranges)
    nbins  = size *(nbins,) if type(nbins) == int else nbins
    bins   = [np.linspace(*range, nbin +1) 
              for nbin, range in zip(nbins, ranges)]

    centers = [clouds.ut_centers(ibin) for ibin in bins]
    
    # indexing 'ij', ensures access via coordinates x[i, j, k]
    xmesh  = np.meshgrid(*centers, indexing = 'ij')
    zs     = fun(*xmesh)
    
    return bins, centers, zs, xmesh