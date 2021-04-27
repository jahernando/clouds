#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Module to Detect Ridges

Created on Tue Apr 27 15:24:01 2021

@author: hernando
"""

import numpy as np


def hessian(x : np.array):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


def hess2d_eigvals(hess):
    """
    
    Compute the eigenvalues and rotation angle of a 2D hessian matrix

    Parameters
    ----------
    hess : np.array (2, 2, :,)

    Returns
    -------
    i1    : np.array, eigen-value
    i2    : np.array, eigen-value
    theta : np.array, theta angle
    
        where:
            U = [(ct, -st) (st, ct)] D = S^T H S, D is diagonal with (i1, i2)

    """
    a, b, c = hess[0, 0], hess[1, 1], hess[0, 1] 
    delta  = a - b
    delta2 = delta*delta
    norm = np.sqrt(delta2 + 4 * c * c)
    i1  = (a + b + norm)/2.
    i2  = (a + b - norm)/2
    s2t = 2 * c / norm
    c2t = delta / norm
    theta = np.arctan2(s2t, c2t)/(2*np.pi)
    return i1, i2, theta