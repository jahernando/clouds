#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Module to Detect Ridges

Created on Tue Apr 27 15:24:01 2021

@author: hernando
"""

import numpy as np
import functools
import operator
    

def _filter(x: np.array, percentile = 80, binary = True):
    
    u0 = np.percentile(x, percentile)
    u  = np.copy(x)
     
    u[x < u0] = 0
    if (binary):
        u[x >= u0] = 1
    return u
    

# def ridge_filter(x : np.array, steps = None, percentile = 80):
    
#     uu = ridge(x, steps)
    
#     u0 = np.percentile(uu, percentile)
    
#     uu[uu < u0] = 0

#     return uu


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
        

# def features(x : np.array, steps = None):
    
#     #shape = x.shape
#     ndim  = x.ndim
    
#     grad     = gradient(x, steps)
#     grad_sph = vector_in_spherical(grad) 
#     hess     = hessian(x, steps)
#     lap    = laplacian(hess)    

#     #leig, eeig = hess_eigh(hess)
#     leig, eeig = np.linalg.eigh(hess)
    
#     #e0    = np.empty(shape + (ndim,), dtype = x.dtype)
#     #for k in range(ndim):
#     #    e0[..., k] = eeig[..., -1, k]  

#     # todo: what to do with zero gradient?
#     fus = []
#     for k in range(ndim):
#         ei = eeig[..., k]
#         fu = np.sum(grad * ei, axis = ndim)
#         fus.append(fu)
        
#     e0_sph = vector_in_spherical(eeig[..., -1])

    
#     return grad, hess, leig, eigh, vgrad, grad_esp, leig, fus


# def ridge_save(x : np.array, steps = None):
    
#     shape = x.shape
#     ndim  = x.ndim
    
#     grad  = gradient(x, steps)
#     vgrad = np.sqrt(np.sum(grad * grad, axis = ndim))    
#     #vgrad = np.round(vgrad, 5)
#     hess  = hessian(x, steps)
    
#     #leig, eeig = hess_eigh(hess)
#     leig, eeig, e0 = hessian_eigh(hess)
    
#     #e0    = np.empty(shape + (ndim,), dtype = x.dtype)
#     #for k in range(ndim):
#     #    e0[..., k] = eeig[..., -1, k]  

#     # todo: what to do with zero gradient?        
#     uu = np.abs(np.sum(grad * e0, axis = ndim))
#     non_zero = vgrad > 0
#     uu[non_zero]  = uu[non_zero]/vgrad[non_zero]
#     uu[~non_zero] = 0.
    
#     # check ridge condition
#     l0  = leig[..., -1]
#     sel = np.full(shape, True, dtype = bool)  
#     for i in range(ndim -1):
#         li = leig[..., i]
#         sel = sel & (li < 0) & (np.abs(li) > np.abs(l0))
    
#     #uu[~sel] = 0.
    
#     return uu, sel
    

def gradient(x : np.array, steps = None):
    
    shape   = x.shape
    ndim    = x.ndim
    steps   = np.ones(ndim) if steps is None else steps
    x_grad  = np.gradient(x, *steps) 
    grad    = np.empty(shape + (ndim,), dtype = x.dtype)
    for k in range(ndim):
        grad[..., k] = x_grad[k]   
    return grad
    

def hessian(x : np.array, steps = None):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    shape   = x.shape
    ndim    = x.ndim
    steps   = np.ones(ndim) if steps is None else steps
    x_grad  = np.gradient(x, *steps)
    #grad    = np.empty(shape + (ndim,), dtype = x.dtype)
    #for k in range(ndim): grad[..., k] = x_grad[k]
    hessian = np.empty(shape + (ndim, ndim), dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k, *steps) 
        for l, grad_kl in enumerate(tmp_grad):
            #norma = steps[k] * steps[l]
            #norma = 1.
            hessian[..., k, l] = grad_kl 
    return hessian


# def hessian_eigh(hess):
    
#     #leig, eeig = hess_eigh(hess)
#     shape = hess.shape[:-2]
#     ndim  = len(shape)
#     leig, eeig = np.linalg.eigh(hess)
    
#     for k in range(ndim):
#         e0 = np.empty(shape + (ndim,), dtype = hess.dtype)
#         e0[..., :] = eeig[..., :, k]  

#     return leig, eeig, e0
    

def laplacian(hess):
    ndim = hess.shape[-1]
    lap  = functools.reduce(operator.add, 
                            [hess[..., i, i] for i in range(ndim)])
    return lap



# def hess2d_eigvals(hess):
#     """
    
#     Compute the eigenvalues and rotation angle of a 2D hessian matrix

#     Parameters
#     ----------
#     hess : np.array (2, 2, :,)

#     Returns
#     -------
#     i1    : np.array, eigen-value
#     i2    : np.array, eigen-value
#     theta : np.array, theta angle
    
#         where:
#             U = [(ct, -st) (st, ct)] D = S^T H S, D is diagonal with (i1, i2)

#     """
#     a, b, c = hess[0, 0], hess[1, 1], hess[0, 1] 
#     delta  = a - b
#     delta2 = delta*delta
#     norm = np.sqrt(delta2 + 4 * c * c)
#     i1  = (a + b + norm)/2.
#     i2  = (a + b - norm)/2
#     s2t = 2 * c / norm
#     c2t = delta / norm
#     theta = np.arctan2(s2t, c2t)/(2*np.pi)
#     return i1, i2, theta


# def laplacian(hess):
#     n = hess.shape[0]
#     return functools.reduce(operator.add, [hess[..., i, i] for i in range(n)])
    


# def hess2d_v(hess, v):
#     lxx, lxy, lyy = hess[0, 0], hess[0, 1], hess[1, 1]
#     vx , vy       = v[0], v[1]
#     ux = lxx * vx + lxy * vy 
#     uy = lxy * vx + lyy * vy
#     uu = vx  * ux + vy  * uy 
#     uu = uu/(vx * vx + vy * vy)
#     return uu


# def hess3d_v(hess, v):
#     lxx, lxy, lxz = hess[0, 0], hess[0, 1], hess[1, 2]
#     lyy, lyz, lzz = hess[1, 1], hess[1, 2], hess[2, 2]
#     vx , vy , vz  = v[0], v[1], v[2]
#     ux = lxx * vx + lxy * vy + lxz * vz
#     uy = lxy * vx + lyy * vy + lyz * vz
#     uz = lxz * vz + lyz * vy + lzz * vz
#     uu = vx  * ux + vy  * uy + vz  * uz
#     uu = uu/(vx * vx + vy * vy + vz * vz)
#     return uu
    

# def hess2d_uv(lxx, lxy, lyy, phi):
#     cp, sp        = np.cos(phi), np.sin(phi)
#     lvv = cp * cp * lxx + 2 * cp * sp * lxy + sp * sp * lyy
#     luu = cp * cp * lxx - 2 * cp * sp * lxy + sp * sp * lyy
#     luv = cp * sp * (lxx - lyy) - lxy * (cp * cp - sp * sp)
#     return luu, luv, lvv
