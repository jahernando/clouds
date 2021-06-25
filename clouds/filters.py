#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 17:03:42 2021

Filters 

@author: hernando
"""

import numpy         as np
import scipy.ndimage as ndimg

gradient                 = None
laplacian                = None
curvatures               = None
min_curvature            = None
min_transverse_curvature = None
transverse_curvatures    = None


def edge_filter(img   : np.array, 
                steps : tuple = None, 
                mask  : np.array = None,
                math_condition  : bool = True,
                perc  : float = 20,
                atol  : float = 0.05):
    """
    Edge Filter.

    Definition: 
        Select cells where the magnitude of the gradient is maximal
        in the gradien direction
        * Lvv: second derivative respect the gradient direction is null
        * Lvvv: third derivative respect the gradient direction is negative
    
    Parameters
    ----------
    img   : np.array, image
    steps : tuple, cells size, optional, default is None (ones)
    mask  : np.array, optional, mask some cells
    perc  : float, optional, percentaje of the filter. Default is 100.
    math_condition  : bool, optional, force curvature condition (Lvvv <0)
    atol  : float, absolute toleranza of lvv to be close to 0., DEfault 0.05

    Returns
    -------
    sel   : np.array(bool) filtered image
    lvv   : np.array, mangitude of the gradient
    
    """

    ndim         = img.ndim
    shape        = img.shape
    vgrad, edir  = gradient(img, steps)
    grad         = vgrad * edir
    
    #hess       = hessian(x, steps)
    lv             = vgrad
    vgrad2, edir2  = gradient(lv, steps)
    glv            = vgrad2 * edir2
    lvv            = np.zeros(shape, dtype = grad.dtype)
    for i in range(ndim): lvv += glv [i] * edir[i]
    #lvv            = np.abs(lvv) 

    vgrad3, edir3 = gradient(lvv, steps)
    glvv = vgrad3 * edir3
    lvvv = np.zeros(shape, dtype = grad.dtype)
    for i in range(ndim): lvvv += glvv[i] * edir[i]
    lvvv  = lvvv     
    
    mask = np.full(shape, True) if mask is None else mask
    
    cut  = np.percentile(vgrad[mask].flatten(), 100 - perc)
    sel1 = vgrad >= cut
    
    sel2 = mask
    if (math_condition):
        sel2  = lvvv < 0 if math_condition else mask
        sel2 = (sel2) & np.isclose(lvv, 0, atol = atol)
    
    sel  = (mask) & (sel1) & (sel2)
    
    return sel, lv
         



def ridge_filter(img        : np.array,
                 steps      : tuple = None,
                 mask       : np.array = None,
                 math_condition       : bool = True,
                 perc       : float = 20,
                 atol       : float = 0.05):
    
    ndim  = img.ndim
    shape = img.shape
    mask  = np.full(shape, True) if mask is None else mask

    grad, gdir = gradient(img, steps) 
    curv, edir = min_transverse_curvature(img, steps)

    # todo grad ane edir orthogonals
    xsel = mask
    if (math_condition):
        ls, _ = transverse_curvatures(img, steps, edir)
        for i in range(ndim - 1):
            xsel = (xsel) & (ls[i] < 0)
        xsel = (xsel) & (np.abs(ls[-2]) > ls[-1])
        cond = np.isclose(np.abs(np.sum(gdir * edir, axis = 0)), 1, atol = atol)
        xsel = (xsel) & (cond)
        
    cut0 = np.percentile(curv[xsel].flatten(), perc)
    xfil = curv <= cut0
    
    nfil = np.full(shape, False)
    
    sel  = (mask) & (xsel) & (xfil)
    nfil[sel] = True
    
    return nfil, curv
    

def ridge_lambda_filter(img        : np.array,
                        steps      : tuple = None,
                        mask       : np.array = None,
                        math_condition       : bool = False,
                        perc       : float = 100,
                        atol       : float = 0.05):
        
    
    shape = img.shape
    mask  = np.full(shape, True) if mask is None else mask

    grad, gdir = gradient(img, steps) 
    curv, edir = min_curvature(img, steps)

    # todo grad ane edir orthogonals
    xsel = mask
    if (math_condition):
        xsel = (xsel) & (curv < 0)
        xsel = (xsel) & (np.isclose(np.sum(gdir * edir, axis = 0), 0, atol = atol))
        
    cut0 = np.percentile(curv[xsel].flatten(), perc)
    xfil = curv <= cut0
    
    nfil = np.full(shape, False)
    
    sel  = mask & xfil
    nfil[sel] = True
    
    return nfil, curv
    

def node_filter(x          : np.array, 
                threshold  : float =  0):
    """
    
    find the local maxima or nodes

    Parameters
    ----------
    x         : np.array, image
    threshold : float, optional. The default is 0.

    Returns
    -------
    xfil : np.array(bool), fitered image

    """
    
    size = x.ndim * (3, )
    xm   = ndimg.maximum_filter(x, size = size)
    xfil = (xm == x) & (x > threshold )
    return xfil


def blob_filter(x     : np.array,
                steps : tuple = None):
    """
    
    returns the local maxima of the minus laplacian

    Parameters
    ----------
    x     : np.array, image
    steps : tuple, step sizes, optional, The default is None (ones).

    Returns
    -------
    xfil  : np.array(bool), filtered image 

    """
    
    lap = - laplacian(x, steps)
    return node_filter(lap)


def normal_laplacian(img   : np.array,
                     sigma : float = 1, 
                     steps : tuple = None):
    """
    
    Compute the minus normal laplacian

    Parameters
    ----------
    img   : np.array, image
    sigma : float, scale, optional. The default is 1.
    steps : tuple, step sizes, optional. The default is None (ones).

    Returns
    -------
    simg : np.array, smoothed image
    lap  : np.array, - normal laplacian

    """
    
    shape  = img.shape
    steps  = np.ones(img.ndim) if steps is None else np.array(steps)
    xsigma = sigma / steps
    
    simg   = ndimg.gaussian_filter(img, xsigma) if sigma > 0 else img
    
    #hess   = hessian(simg, steps)
    #vhess  = np.array([hess[i, i] for i in range(ndim)])
    #nlap   = np.array([xsigma[i] * vhess[i] for i in range(ndim)])
    #nlap   = np.sum(nlap, axis = 0)
    
    curvs  = curvatures(simg, steps)
    nlap   = np.zeros(shape)
    for curv in curvs: nlap += curv
    
    return -nlap, simg
    

def nlap_scan(img    : np.array, 
              sigmas : tuple = (1,),
              steps  : tuple = None,
              filter : bool  = False):
    """
    
    scan the normal laplacian

    Parameters
    ----------
    img    : np.array
    sigmas : tuple, scale values, optional. The default is (1,).
    steps  : tuple, step sizes, optional.  The default is None (ones.
    filter : bool, filter nodes, optional. The default is True.

    Returns
    -------
    sigmax : np.array, maximum value of the scale
    lapmax : np.array, maximum value of the minus normal laplacian
    laps   : np.array(img.shape, sigmas.shape), normal laplacian for each scale

    """
    
    
    
    ndim    = img.ndim
    nsigmas = len(sigmas)
    shape   = img.shape
    steps   = np.ones(ndim) if steps is None else np.array(steps)
    
    lapmax   = np.zeros(shape)
    sigmax   = np.zeros(shape)
    laps     = np.zeros( (nsigmas,) + shape)
    for i, sigma in enumerate(sigmas):
        lap, _     = normal_laplacian(img, sigma, steps)
        xfil       = node_filter(lap) if filter else np.full(shape, True)
        lap[~xfil] = 0
        sel  = (lap > lapmax) & (xfil)
        lapmax[sel]  = lap[sel]
        sigmax[sel]  = sigma
        laps[i]      = lap
    return sigmax, lapmax, laps

