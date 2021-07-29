#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 17:03:42 2021

Filters 

@author: hernando
"""

import numpy         as np
import scipy.ndimage as ndimg

def get_edge_filter(gradient):

    gradient = gradient    

    def edge_filter(img   : np.array, 
                    steps : tuple = None, 
                    mask  : np.array = None,
                    math_condition  : bool = True,
                    perc  : float = 100,
                    atol  : float = 0.1):
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
    
    #print('mask ', np.sum(mask))
    #print('sel1 ', np.sum(sel1))
    #print('sel2 ', np.sum(sel2))
        sel  = (mask) & (sel1) & (sel2)
    
        return sel, lv
         
    return edge_filter


def get_ridge_filter(gradient, min_transverse_curvature):
    
    gradient                 = gradient
    min_transverse_curvature = min_transverse_curvature

    def ridge_filter(img        : np.array,
                      steps      : tuple = None,
                      mask       : np.array = None,
                      math_condition       : bool = True,
                      perc       : float = 100,
                      atol       : float = 0.05):
        """
        
        Ridge filter:

        A ridge requires:
            * the transverse minimum curvature must be negative
            * the gradient direction is aligned with the normal to the minimum
            curvature transverse

        Parameters
        ----------
        img : np.array
        steps : tuple, optional
        mask : np.array, optional
        math_condition : bool, optional. Default is True
        perc : float, optional. The default is 100.
        atol : float, optional. The default is 0.05.

        Returns
        -------
        nfil : np.array, filter
        curv : np.array, value of the minimum transverse curvature

        """
        
        shape = img.shape
        mask  = np.full(shape, True) if mask is None else mask
    
        grad, gdir = gradient(img, steps) 
        curv, edir = min_transverse_curvature(img, steps)
        
        #print('grad ', grad.shape, 'gdir ', gdir.shape)
        #print('curv ', grad.shape, 'edir ', edir.shape)

        xsel = curv < 0
        #print('curv neg ', np.sum(xsel))
        if (math_condition):
            # edir and gdir maybe not unitary
            gmod  = np.sqrt(np.sum(gdir * gdir, axis = 0))
            cmod  = np.sqrt(np.sum(edir * edir, axis = 0))
            vmod  = gmod * cmod
            amod  = np.sum(gdir * edir, axis = 0)
            #print('vmod ', vmod.shape)
            #print('amod ', amod.shape)
            cond = np.isclose(amod, vmod, atol = atol)
            #print('cond ', cond.shape, np.sum(cond))
            xsel = (xsel) & (cond)
        #print('orthog ', np.sum(xsel))
                   
        cut0 = np.percentile(curv[mask].flatten(), perc)
        xfil = curv <= cut0
        #print('perc ', np.sum(xfil))
        
        nfil = np.full(shape, False)
        
        sel  = (mask) & (xsel) & (xfil)
        nfil[sel] = True
        
        return nfil, curv
    
    return ridge_filter
    

def get_ridge_lambda_filter(gradient, min_curvature):
    
    gradient      = gradient
    min_curvature = min_curvature

    def ridge_lambda_filter(img        : np.array,
                            steps      : tuple = None,
                            mask       : np.array = None,
                            math_condition       : bool = True,
                            perc       : float = 100,
                            atol       : float = 0.05):
        """
        
        A Ridge simplified filter
        
        Requires:
            * The minimum curvature must be negative
            * the direction of the minimum curvature must be orthogonal to the gradient

        Parameters
        ----------
        img : np.array
        steps : tuple, optional
        mask : np.array, optional. The default is None.
        math_condition : bool, optional. The default is True.
        perc : float, optional. The default is 100.
        atol : float, optional. The default is 0.05.

        Returns
        -------
        nfil : np.array, filter
        curv : np.array, value of the minimum curvature

        """
            
        
        shape = img.shape
        mask  = np.full(shape, True) if mask is None else mask
    
        grad, gdir = gradient(img, steps) 
        curv, edir = min_curvature(img, steps)
        
        xsel = (curv < 0)
        if (math_condition):
            xsel = (xsel) & (np.isclose(np.sum(gdir * edir, axis = 0), 0, atol = atol))
            
        cut0 = np.percentile(curv[mask].flatten(), perc)
        xfil = curv <= cut0
        
        nfil = np.full(shape, False)
        
        #print('mask ', np.sum(mask))
        #print('xsel ', np.sum(xsel))
        #print('xfil ', np.sum(xsel))
        sel  = (xsel) & (mask) & (xfil)
        nfil[sel] = True
        
        return nfil, curv
    
    return ridge_lambda_filter
    

def node_filter(x     : np.array, 
                mask  : np.array = None):
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
    mask = np.full(x.shape, True) if mask is None else mask
    xfil = (xm == x) & (mask)
    return xfil

def get_blob_filter(laplacian):
    
    laplacian = laplacian

    def blob_filter(x     : np.array,
                    steps : tuple = None,
                    mask  : np.array = None):
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
        return node_filter(lap, mask = mask)

    return blob_filter


def get_normal_laplacian(curvatures):
    
    curvatures = curvatures

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
        
        nlap = - sigma * nlap 
        
        return nlap, simg
    
    return normal_laplacian
    

def get_nlap_scan(normal_laplacian):
    
    normal_laplacian = normal_laplacian

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

    return nlap_scan
    