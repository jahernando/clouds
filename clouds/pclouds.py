#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 12:03:47 2021

@author: hernando
"""

import numpy             as np

import clouds.dclouds    as dclouds
import clouds.utils      as cu


import matplotlib.pyplot as plt

import matplotlib.colors   as colors
import matplotlib.cm       as colormap
from   mpl_toolkits.mplot3d import axes3d

plt.rcParams['image.cmap'] = 'rainbow'

fig  = lambda nx = 1, ny = 1, sz = 4: plt.figure(figsize = (sz * ny,  sz * nx))
ax3d = lambda nx = 1, ny = 1, i = 1 : plt.gcf().add_subplot(nx, ny, i, projection = '3d')
efig = plt.tight_layout
#cells_select = clouds.cells_select
hopts        = {'histtype': 'step'}


def to_color(weights, cmap = colormap.rainbow, alpha = 1.):
    
    shape = weights.shape
    scale = weights.flatten()
    norm      = colors  .Normalize(vmin = np.min(scale), vmax = np.max(scale), clip=True)
    mapper    = colormap.ScalarMappable(norm=norm, cmap= colormap.rainbow)
    fcolor = mapper.to_rgba(scale)
    fcolor = fcolor.reshape(shape + (4,))
    fcolor[..., 3] = alpha * fcolor[..., 3]
    
    return fcolor

    
def scatter(img, bins = None, mask = None, **kargs):
    
    ndim = img.ndim
    ax = plt.gca(projection = '3d') if ndim == 3 else plt.gca()

    bins = dclouds._bins(img)       if bins is None else bins
    mask = np.full(img.shape, True) if mask is None else mask
       
    mask = img > 0 if mask is None else mask
    cells, enes = dclouds._scells(img, bins, mask)
    scale = cu.ut_scale(enes)
    ax.scatter(*cells, c = scale, **kargs)
    
    return ax


def voxels(img, bins = None, mask = None, **kargs):
    
    ndim = img.ndim
    ax = plt.gca(projection = '3d') if ndim == 3 else plt.gca()
    
    
    bins = dclouds._bins(img)       if bins is None else bins
    mask = np.full(img.shape, True) if mask is None else mask
    cells, enes = dclouds._scells(img, bins, mask)
    
    if (ndim == 2):
        plt.gca().hist2d(*cells, bins = bins, weights = enes, **kargs)
        #plt.colorbar(c);
        return

    hist, _  = np.histogramdd(cells, bins = bins, weights = enes)
    xmesh    = np.meshgrid(*bins, indexing = 'ij')
    mask     = hist > 0
    #umask      = np.copy(mask)
    #filled     = np.swapaxes(umask, 0, 1).astype(bool)

    alpha = kargs['alpha'] if 'alpha' in kargs.keys() else 0.1
    cols  = to_color(hist, alpha)
 
    plt.gca(projection = '3d')
    ax.voxels(*xmesh, mask, facecolor = cols[mask], **kargs);
                     #edgecolor = cols[mask],
#                     **kargs);
    
    return


def quiver(img, edir, bins, mask = None,  **kargs):
    """ Draw the gradient of the cells
    """

    ndim  = img.ndim
    shape = img.shape
    steps = [bin[1] - bin[0] for bin in bins]
    x0    = [bin[0]          for bin in bins]
    mask  = np.full(shape, True) if mask is None else mask
    
    _, cells, _ = dclouds._cells(img, steps, x0, mask = mask)
    #print(cells)

    xdirs  = [steps[i] * edir[i][mask] for i in range(ndim)]
    opts = {'scale_units': 'xy', 'scale' : 2.} if ndim == 2 else {'length' : 0.4}

    plt.gca().quiver(*cells, *xdirs, **opts, **kargs)
    
    return