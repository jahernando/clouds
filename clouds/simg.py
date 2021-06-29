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
   
import clouds.filters as filters

#import scipy.ndimage as ndimg     

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


def curvature(x     : np.array,
              edir  : np.array ,
              steps : tuple = None):
    
    shape  = x.shape 
    ndim   = x.ndim
    hess   = hessian(x, steps)
    curv   = np.zeros(shape)
    for i in range(ndim):
        for j in range(ndim):
            curv += hess[i, j] * edir[i] * edir[j]
    curv  /= 2
    #xhess  = hess if edir.ndim > 1 else hess.T
    #curv  = np.dot(np.dot(xhess, edir), edir)/2
    return curv  


def curvatures(img   : np.array,
               steps : tuple = None):
    
    ls, _  = _hess_eigen(img, steps)
    ls     = [li/2 for li in ls]
    return ls
    

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

    leig, eeig = _hess_eigen(img, steps)    
    
    return leig[0], eeig[0]
    

def min_transverse_curvature(x: np.array,
                             steps = None):
    
    ndim       = x.ndim
    shape      = x.shape
    #vgrad, edir = gradient(x, steps)
    #xgrad      = vgrad * edir
    #grad       = np.zeros(shape + (ndim,))
    #for i in range(ndim):
    #    grad[..., i] = xgrad[i]
    #vgrad      = np.sqrt(np.sum(grad * grad, axis =ndim))
    #hess       = hessian(x, steps)
    #vhess      = _rev_matrix(hess)
    
    leig, eeig = _hess_eigen(x, steps)
    
    curv  = np.zeros(shape)
    for i in range(ndim -1): curv += leig[i]
    edir0 = eeig[-1]  
 
    return curv, edir0


# def transverse_curvatures(x     : np.array,
#                           edir  : np.array,
#                           steps : tuple = None):
    
#     lap  = laplacian(x, steps)
#     curv = curvature(x, edir, steps)
#     return (lap - curv,)



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


# filters
#------------------------


edge_filter          = filters.get_edge_filter(gradient)
ridge_filter         = filters.get_ridge_filter(gradient, 
                                                min_transverse_curvature)
ridge_lambda_filter  = filters.get_ridge_lambda_filter(gradient,
                                                       min_curvature)
node_filter          = filters.node_filter
blob_filter          = filters.get_blob_filter(laplacian)
normal_laplacian     = filters.get_normal_laplacian(curvatures)
nlap_scan            = filters.get_nlap_scan(normal_laplacian)


#  internal functions
#-----------------------

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
