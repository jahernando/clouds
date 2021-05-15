#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 10:23:32 2021

@author: hernando
"""

import numpy             as np
import matplotlib.pyplot as plt
from   clouds.clouds     import cells_select


def plotter(bins, mask, cells):
    
    ndim = len(cells)

    def _plot(var, name, perc = 0, nbins = 40, n = 2, i = 1, **kargs):
    
        v0   = np.percentile(var[mask], perc)
        vsel = var[mask] >= v0
        
        if (ndim == 3):
            plt.gcf().add_subplot(n, 2, i, projection = '3d')
        else:
            plt.subplot(n, 2, i)

        plt.gca().scatter(*cells_select(cells, vsel), 
                      c = var[mask][vsel], **kargs)
        plt.xlabel('x'); plt.ylabel('y'); plt.title(name)
    
        plt.subplot(n, 2, i+1)
        _, ubins, _ = plt.hist(var[mask], nbins, histtype = 'step')
        plt.xlabel(name)
        if (perc > 0):
            plt.hist(var[mask][vsel], ubins, histtype = 'step')
    
    return _plot