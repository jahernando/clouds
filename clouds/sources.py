#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 10:50:20 2021

@author: hernando
"""

import numpy         as np
#import scipy.ndimage as ndimg

#import numpy.linalg as nplang

#import clouds.ridges as ridges
import clouds.utils   as cu

def taylor(a0 = 0, a = (0, 0), b = (0, 0), c = (0,)):
    
    ndim = len(a)
    assert len(a) == len(b), 'invalid number of a and b coeficients'
    ok   = len(c) == 1 if ndim == 2 else len(c) == 3
    assert ok, 'not valid number of c coeficients'
    
    grad = np.array(a)
    
    hess    = np.zeros((ndim, ndim))
    k= 0
    for i in range(ndim):
        for j in range(i+1, ndim):
            hess[i, j] = c[k]
            k += 1
    hess += hess.T
    for i, bi in enumerate(b): hess[i, i] = bi
        
    def fun(x):
        y  = a0
        for i in range(ndim):
            y += grad[i] * x[i]
            y += hess[i, i] * x[i] * x[i]/2
            for j in range(i+1, ndim):
                y += hess[i, j] * x[i] * x[j]
        return y
    
    return fun, grad, hess


def from_function(fun, nbins, ranges):
    """
    
    Generate a image using a function and ranges with bins

    Parameters
    ----------
    nbins  : int or tuple(int), number of bins
    ranges : tuple( (a, b)), range in each axis
    fun    : function(x, y, z)

    Returns
    -------
    zs      : np.array, z-values matrix z[i, j, k]
              where i, j, k = x, y, z indices
    bins    : np.array, bins 
    """
    
    size   = len(ranges)
    nbins  = size *(nbins,) if type(nbins) == int else nbins
    bins   = [np.linspace(*range, nbin +1) 
              for nbin, range in zip(nbins, ranges)]

    xmesh  = cu.ut_mesh(bins)
    zs     = fun(xmesh)
    
    return zs, bins


def from_histogram(coors   : np.array, 
                   weights : np.array,
                   steps   : tuple  = None):
    
    ndim   = len(coors)
    steps  = np.ones(ndim) if steps is None else steps
    bins   = [cu.arstep(x, step, True) for x, step in zip(coors, steps)] 
    img, _ = np.histogramdd(coors, bins = bins, weights = weights)

    return img, bins


def points(npixels = 50,
           npoints = 5,
           ndim = 2):
    
    img    = np.zeros(ndim*(npixels,))
    indices = [np.random.choice(range(npixels), ndim) for i in range(npoints)]
    indices = [tuple(index) for index in indices]
    for index in indices:
        img[tuple(index)] = 1
        
    return img, indices


def disks(npixels   = 250, 
          npoints   = 4,
          ndim      = 2,
          maxradius = 10):
   
    minradius = 1
    
    bins    = [np.linspace(0, npixels, npixels + 1).astype(int) for i in range(ndim)]
    xmesh   = cu.ut_mesh(bins)
    radius  = np.random.uniform(minradius, maxradius, npoints)
    indices = [np.random.choice(range(maxradius, npixels - maxradius), ndim) \
                                for i in range(npoints)]
        
    shape   = ndim * (npixels,)
    img     = np.zeros(shape)
    for i, index in enumerate(indices):
        point    = 0.5 + index
        irad     = 0
        for k in range(ndim): 
            irad += (xmesh[k] - point[k])**2
        irad     = np.sqrt(irad)
        sel      = irad < radius[i]
        img[sel] = 1
    
    indices = [tuple(index) for index in indices]
    return img, indices, radius            
        

def rectangle(npixels = 250,
              xlength = 100,
              ylength = 100):

    ndim    = 2
    bins    = [np.linspace(0, npixels, npixels + 1).astype(int) for i in range(ndim)]
    
    i0      = int(npixels/2)
    sx, sy  = int(xlength/2), int(ylength/2)

    img = np.zeros((npixels, npixels))
    img[ i0 - sx : i0 + sx, i0 - sy : i0 + sy] = 1
    return img, bins
      


def line(xline, eline, ts,  nbins, ranges):
    
    ndim   = len(xline)
    shape  = ndim *(nbins,) if type(nbins) == int else nbins
    bins   = [np.linspace(*range, nbin +1) 
              for nbin, range in zip(shape, ranges)]
    
    x    = [xl(ts) for xl in xline]
    xi   = [np.digitize(xi, bin) -1 for xi, bin in zip(x, bins)]
    ene  = eline(ts)

    img    = np.zeros(shape)    
    for i, xi in enumerate(zip(*xi)):
        img[tuple(xi)] += ene[i]
    
    return img, bins


#--- other



#--------------

# def generate_line(size, xline, eline, sigma = 0):
    
#     t    = np.random.uniform(size = size)
#     x    = [xl(t) for xl in xline]
#     for xi in x: xi += sigma * np.random.normal(size = size)
#     ene  = eline(t)
    
#     return x, ene, t


# def generate_points(npoints = 10, ndim = 2, npixels = 60):
#     """
    
#     Generate n random points image
    
  
#     Parameters
#     ----------
#     npoints : TYPE, optional
#         DESCRIPTION. The default is 10.
#     ndim : TYPE, optional
#         DESCRIPTION. The default is 2.
#     npixels : TYPE, optional
#         DESCRIPTION. The default is 60.

#     Returns
#     -------
#     img : TYPE
#         DESCRIPTION.
#     indices : TYPE
#         DESCRIPTION.

#     """
    
#     #width  = npixels  
#     img    = np.zeros(ndim*(npixels,))
#     indices = [np.random.choice(range(npixels), ndim) for i in range(npoints)]
#     for index in indices:
#         img[tuple(index)] = 1
#     return img, indices


# def generate_disks(npoints = 10,
#                    maxradius = 5,
#                    ndim = 2,
#                    gaus = False,
#                    npixels = 60):
    
#     minradius =  2  
#     width     = 2 * maxradius if ndim == 3 else npixels
#     maxradius = 10 if ndim == 3 else maxradius
    
#     bins    = [np.linspace(-width - 0.5, width + 0.5, 2 * width + 2) for i in range(ndim)]
#     centers = [(0.5 *(bin[1:] + bin[:-1])).astype(int)  for bin in bins]
                                 
#     xmesh    = np.meshgrid(*centers, indexing = 'ij')
#     length   = width -  2 * maxradius
#     xpoints  = [np.random.uniform(-length, length, npoints) for i in range(ndim)]
#     ipoints  = [np.digitize(xi, bin)-1 for xi, bin in zip(xpoints, bins)]
#     radius   =  np.random.uniform(minradius, maxradius, npoints)
    
#     shape  = [(2 * width + 1) for i in range(ndim)]
#     img    = np.zeros(shape)
#     for i in range(npoints):
#         iimg        = np.zeros(shape)
#         xpos, rad   = np.array([xp[i] for xp in xpoints]), radius[i]
#         #print('point ', xpos, ', radius ', rad)
#         mrad        = minradius if gaus is True else rad
#         irad        = 0
#         for i in range(ndim):
#             irad += (xmesh[i] - xpos[i])**2
#         sel         = np.sqrt(irad) <= mrad
#         #print('sel ', np.sum(sel), sel.shape)
#         intensity   = 1e3 if gaus is True else 1
#         iimg[sel]   = intensity
#         if (gaus):
#             iimg    = ndimg.gaussian_filter(iimg, rad)
#         img        += iimg 
        
#     return img, xmesh, ipoints, radius


# def generate_rectange(xlength = 20,
#                       ylength = 20,
#                       npixels = 60,
#                       sigma   = 0):

#     i0      = int(npixels/2)
#     sx, sy  = xlength, ylength

#     img = np.zeros((npixels, npixels))
#     img[ i0 - sx : i0 + sx, i0 - sy : i0 + sy] = 1
#     img = ndimg.gaussian_filter(img, sigma)
    
#     #img = np.zeros((npixels, npixels))
#     #img[square : -square, square : -square] = 1
#     #img = ndimg.gaussian_filter(img, sigma)
    
#     return img

