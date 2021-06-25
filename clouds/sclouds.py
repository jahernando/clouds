#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Module to find features in smooth (diferentiable) images

Created on Tue Apr 27 15:24:01 2021

@author: hernando
"""

import numpy as np
import functools
import operator
   
import scipy.ndimage as ndimg     
   


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
         

# def ridge_filter(img        : np.array,
#                  steps      : tuple = None,
#                  mask       : np.array = None,
#                  math_condition       : bool = False,
#                  perc       : float = 100):
#     """
    
#     Ridge filter
    
#     Definition:
#         select cells where the curvature on the ortogonal directions 
#         of the gradient is minimal
#         * v-grad == e_n 
#         (the eigen-vector with the largest eigenvalue of the Hessian 
#          has the same direction as the gradient)
#         * |l_i| > |l_n| and l-i <0, 
#         (the eigenvalues are in magniture greather the maximum curvature, and
#          they are all negative, that is, with negative curvature)
                                     

#     Parameters
#     ----------
#     img     : np.array, image
#     steps   : tuple, cell sizes, detaul = None (ones)
#     mask    : np.array, filter cell, defaul = None
#     math_condition    : bool, consider all proyection 
#               or only the projection with minimal curvature
#     perc    : float,  percentaje of the filter, default = 100.

#     Returns
#     -------
#     sel     : np.array(bool), boolean array 
#               with the cells that pass the filter cndition
#     fu      : np.array(float), curvature in the ortoghonal direction(s) 
#               of the gradient
    
#     """
    
#     sels, fus = _ridge(img, steps)
    
#     mask = np.full(img.shape, True) if mask is None else mask

#     sel  = (sels[0]) & (mask)
#     if (math_condition):
#         for isel in sels[1: ]: sel = (sel) & isel
 
#     # Two options, either the orthogonal derection to the grad is null
#     # either the max-eigenvector is parallel to the gradient
    
#     fu    = fus[0] * fus[0]
#     if (math_condition):
#         for fui in fus[1 : -1]: fu += fui * fui
#     fu    = np.sqrt(fu)
#     cut   = np.percentile(fu[sel],  perc)
#     sel   = (sel) & (fu <= cut)
    
    
#     fu = np.abs(fus[-1])
#     cut   = np.percentile(fu[sel],  100 - perc)
#     sel   = (sel) & (fu >= cut)
    
#     return sel, fu

# def ridge_filter(img        : np.array,
#                  steps      : tuple = None,
#                  mask       : np.array = None,
#                  math_condition       : bool = False,
#                  perc       : float = 100):
#     """
    
#     Ridge filter
    
#     Definition:
#         select cells where the curvature on the ortogonal directions 
#         of the gradient is minimal
#         * v-grad == e_n 
#         (the eigen-vector with the largest eigenvalue of the Hessian 
#          has the same direction as the gradient)
#         * |l_i| > |l_n| and l-i <0, 
#         (the eigenvalues are in magniture greather the maximum curvature, and
#          they are all negative, that is, with negative curvature)
                                     

#     Parameters
#     ----------
#     img     : np.array, image
#     steps   : tuple, cell sizes, detaul = None (ones)
#     mask    : np.array, filter cell, defaul = None
#     math_condition    : bool, consider all proyection 
#               or only the projection with minimal curvature
#     perc    : float,  percentaje of the filter, default = 100.

#     Returns
#     -------
#     sel     : np.array(bool), boolean array 
#               with the cells that pass the filter cndition
#     fu      : np.array(float), curvature in the ortoghonal direction(s) 
#               of the gradient
    
#     """
    
#     sels, fus = _ridge(img, steps)
    
#     mask = np.full(img.shape, True) if mask is None else mask

#     sel  = (sels[0]) & (mask)
#     if (math_condition):
#         for isel in sels[1: ]: sel = (sel) & isel
 
#     # Two options, either the orthogonal derection to the grad is null
#     # either the max-eigenvector is parallel to the gradient
    
#     fu    = fus[0] * fus[0]
#     if (math_condition):
#         for fui in fus[1 : -1]: fu += fui * fui
#     fu    = np.sqrt(fu)
#     cut   = np.percentile(fu[sel],  perc)
#     sel   = (sel) & (fu <= cut)
    
    
#     fu = np.abs(fus[-1])
#     cut   = np.percentile(fu[sel],  100 - perc)
#     sel   = (sel) & (fu >= cut)
    
#     return sel, fu



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
        ls, _ = _hess_eigen(img, steps)
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



def features(x     : np.array,
             steps : tuple = None):
    """
    
    Return the features of the image

    Parameters
    ----------
    x     : np.array, image
    steps : tuple, cells sizes, optional, default is None (ones)

    Returns
    -------
    vgrad : np.array, gradient direction
    lap   : np.array, laplacian
    dhess : np.array, hessian determinant
    lmin  : np.array, min eigen-values of the hessian 
    
    """
    
    ndim       = x.ndim
    vgrad, _   = gradient(x, steps)
    hess       = hessian(x, steps)
    dhess      = det_hessian(x, steps)
    vhess      = _rev_matrix(hess)
    leig, _    = np.linalg.eigh(vhess)
    lmin       = leig[..., 0]
    lap        = np.zeros(x.shape)
    for i in range(ndim):
        lap += leig[..., i]
    
    return vgrad, lap, dhess, lmin


# utilities 


def vector_in_spherical(v: np.array):
    
    ndim   = v.shape[-1]
    vx, vy = v[..., 0], v[..., 1]
    vr     = np.sqrt(vx * vx + vy * vy)
    phi    = np.arctan2(vy, vx)/np.pi
    
    if (ndim == 2): 
        return (vr, phi)
    
    vz    = v[..., 2]
    theta = np.arctan2(vz, vr)/np.pi
    vg    = np.sqrt(vx * vx + vy * vy + vz * vz)
    return (vg, phi, theta)
            

def gradient(x : np.array, steps = None):
    """
    
    Compute ther gradient

    Parameters
    ----------
    x     : np.array, image
    steps : np.array or tuple, step in each direction. Default is None (ones)

    Returns
    -------
    grad :  np.array, gradient

    """
    
    ndim    = x.ndim
    steps   = np.ones(ndim) if steps is None else steps
    grad    = np.array(np.gradient(x, *steps))
    
    vgrad  = np.sqrt(np.sum(grad * grad, axis = 0))
    
    sel    = np.isclose(vgrad, 0)
    edir   = grad/vgrad
    for i in range(ndim):
        edir[i][sel] = 0
    
    return vgrad, edir
    

def hessian(x : np.array, steps = None):
    """
    
    Compute the hessian matrix

    Parameters
    ----------
    x     : np.array,
    steps : np.array or tuple, step in each direction. Default is None (ones)


    Returns
    -------
    hessian : np.array, hessian 

    """
    shape   = x.shape
    ndim    = x.ndim
    steps   = np.ones(ndim) if steps is None else steps
    x_grad  = np.gradient(x, *steps)
    #grad    = np.empty(shape + (ndim,), dtype = x.dtype)
    #for k in range(ndim): grad[..., k] = x_grad[k]
    hessian = np.empty((ndim, ndim) + shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k, *steps) 
        for l, grad_kl in enumerate(tmp_grad):
            #norma = steps[k] * steps[l]
            #norma = 1.
            hessian[k, l] = grad_kl 
    return hessian



def laplacian(img, steps = None):
    """
    
    Laplacian

    Parameters
    ----------
    img   : np.array, image
    steps : np.array or tuple, step in each direction. Default is None (ones)

    Returns
    -------
    lap   : laplacian

    """
    
    hess = hessian(img, steps)
    lap  = _laplacian(hess)
    return lap



def det_hessian(img, steps):
    """
    
    determinant of the hessian

    Parameters
    ----------
    img   : np.array,
    steps : np.array or tuple, step in each direction. Default is None (ones)

    Returns
    -------
    det   : np.array, hessian determinant

    """
    
    hess  = hessian(img, steps)
    hessv = _rev_matrix(hess)
    det   = np.linalg.det(hessv)
    return det


def min_curvature(img   : np.array,
                  steps : tuple = None):
    """
    
    compure the min curvature (min of the hessian eigen values)

    Parameters
    ----------
    img   : np.array
    steps : tuple, step sizes, optional. The default is None (ones).

    Returns
    -------
    lmin : np.array, minimum hessian eigenvalue or min curvature
    emin : np.array, eigenvector 

    """
    
    ndim       = img.ndim
    shape      = img.shape
    hess       = hessian(img, steps)
    vhess      = _rev_matrix(hess)
    leig, eeig = np.linalg.eigh(vhess)
    lmin       = leig[..., 0]
    emin       = np.zeros((ndim, ) + shape)
    for i in range(ndim): emin[i]       = eeig[..., i, 0]
    return lmin, emin


def min_transverse_curvature(x: np.array,
                             steps = None):
    
    ndim       = x.ndim
    shape      = x.shape
    vgrad, edir = gradient(x, steps)
    xgrad      = vgrad * edir
    grad       = np.zeros(shape + (ndim,))
    for i in range(ndim):
        grad[..., i] = xgrad[i]
    vgrad      = np.sqrt(np.sum(grad * grad, axis =ndim))
    hess       = hessian(x, steps)
    vhess      = _rev_matrix(hess)
    
    leig, eeig = np.linalg.eigh(vhess)
    
    curv  = np.zeros(shape)
    for i in range(ndim -1):
        curv += leig[..., i]
    edir0 = _rev_matrix(eeig[..., -1])  
 
    return curv, edir0


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
    
    ndim   = img.ndim
    steps  = np.ones(img.ndim) if steps is None else np.array(steps)
    xsigma = sigma / steps
    
    simg   = ndimg.gaussian_filter(img, xsigma) if sigma > 0 else img
    
    hess   = hessian(simg, steps)
    vhess  = np.array([hess[i, i] for i in range(ndim)])
    nlap   = np.array([xsigma[i] * vhess[i] for i in range(ndim)])
    nlap   = np.sum(nlap, axis = 0)
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


#----------------------
#     other
#-----------------------


# def min_transverse_curvature(x: np.array, steps = None):
    
#     ndim       = x.ndim
#     shape      = x.shape
#     vgrad, edir = gradient(x, steps)
#     xgrad      = vgrad * edir
#     grad       = np.zeros(shape + (ndim,))
#     for i in range(ndim):
#         grad[..., i] = xgrad[i]
#     vgrad      = np.sqrt(np.sum(grad * grad, axis =ndim))
#     hess       = hessian(x, steps)
#     vhess      = _rev_matrix(hess)
    
#     leig, eeig = np.linalg.eigh(vhess)
    
#     curv  = np.zeros(shape)
#     for i in range(ndim -1 ):
#         curv += leig[..., i]
#     edir0 = _rev_matrix(eeig[..., -1])  
 
#     return curv, edir0


def _hess_eigen(x: np.array, steps = None):
    
    ndim       = x.ndim
    shape      = x.shape
    vgrad, edir = gradient(x, steps)
    xgrad      = vgrad * edir
    grad       = np.zeros(shape + (ndim,))
    for i in range(ndim):
        grad[..., i] = xgrad[i]
    vgrad      = np.sqrt(np.sum(grad * grad, axis =ndim))
    hess       = hessian(x, steps)
    vhess      = _rev_matrix(hess)
    
    leig, eeig = np.linalg.eigh(vhess)
    
    ls   = [leig[..., i] for i in range(ndim)]
    xeis = [_rev_matrix(eeig[..., i]) for i in range(ndim)]
 
    return ls, xeis


# def _ridge(x: np.array, steps = None):
    
#     ndim       = x.ndim
#     shape      = x.shape
#     vgrad, edir = gradient(x, steps)
#     xgrad      = vgrad * edir
#     grad       = np.zeros(shape + (ndim,))
#     for i in range(ndim):
#         grad[..., i] = xgrad[i]
#     vgrad      = np.sqrt(np.sum(grad * grad, axis =ndim))
#     hess       = hessian(x, steps)
#     vhess      = _rev_matrix(hess)
    
#     leig, eeig = np.linalg.eigh(vhess)
    
    
#     ls    = [leig[..., i]                             for i in range(ndim)]
#     fus   = [np.sum(eeig[..., i] * grad   , axis = ndim) for i in range(ndim)]
#     fus   = [fu/vgrad for fu in fus]
#     # what to do with a vgrad close to 0?
#     for fu in fus: fu[np.isclose(vgrad, 0)] = 1.

#     l0    = ls[-1]
#     sels  = [(ls[i] < 0) & (np.abs(ls[i]) > np.abs(l0)) for i in range(ndim-1)]
 
#     return sels, fus


def _rev_matrix(h):
    
    ndim  = h.shape[1]
    shape = h.shape[2:]
    
    hrev  = np.zeros(shape + (ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            hrev[..., i, j] = h[i, j]
    return hrev


def _laplacian(hess):
    ndim = hess.shape[0]
    lap  = functools.reduce(operator.add, 
                            [hess[i, i] for i in range(ndim)])
    return lap
