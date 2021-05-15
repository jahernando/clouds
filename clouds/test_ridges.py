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
    
    bins    = [np.linspace(*range, nbins +1) for range in ranges]

    centers = [clouds.ut_centers(ibin) for ibin in bins]
    #xcs = clouds.ut_centers(xbins)
    #ycs = clouds.ut_centers(ybins)
    
    xmesh  = np.meshgrid(*centers)
    xmesh.reverse()
    zs       = fun(*xmesh)
    
    return bins, centers, zs