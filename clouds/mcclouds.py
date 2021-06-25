#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 12:20:57 2021

@author: hernando
"""

import numpy          as np
import clouds.dclouds as dclouds


def mcimg(img, mccoors, mcenes, steps = None, x0 = None):
    
    bins  = dclouds._bins(img, steps, x0)
    mcimg, _  = np.histogramdd(mccoors, bins, weights = mcenes)
    
    return mcimg


def mcblobs(img, mccoors, mcenes, mcids, steps = None, x0 = None):
    
    bins   = dclouds._bins(img, steps, x0)
    mcblob = np.zeros(img.shape).flatten()
    
    for i in (1, 2):
        sel     = mcids == i
        if (np.sum(sel) <= 0): continue
        icoors  = dclouds.cells_selection(mccoors, sel)
        etrk, _ = np.histogramdd(icoors, bins, weights = mcenes[sel])
        imask   = np.argmax(etrk)
        mcblob[imask] = etrk.flatten()[imask]
        #print(etrk.flatten()[imask])
        #print(mcblob.flatten()[imask])
    
    return mcblob.reshape(img.shape)
 