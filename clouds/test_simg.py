#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 11:43:09 2021

@author: hernando
"""

import numpy         as np

import scipy.ndimage as ndimg

import clouds.utils     as cu
import clouds.sources   as sources
import clouds.simg      as simg

nborder = 5

def test_gradient(nbins  = 101,
                  ranges = ((-1, 1), (-1, 1)),
                  a0     = 0,
                  a      = (1, -1),
                  tol   = 1e-3):
    
    ndim   = len(a)  
    fun, grad0, _ = sources.taylor(a0 = a0, a = a)
    
    img, bins = sources.from_function(fun, nbins, ranges)
    steps     = cu.ut_steps(bins)
    
    vgrad, edir  = simg.gradient(img, steps) 
    grad  = vgrad * edir

    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    for i in range(ndim):
        print('dir ', i, np.mean(grad[i][center]), grad0[i])
        assert np.isclose(np.mean(grad[i][center]), grad0[i], atol = tol), \
            'not good enough gradient in coord ' + str(i)
        
    return
    

def test_hessian(nbins  = 101, 
                 ranges = ((-1, 1), (-1, 1)),
                 b      = (1, -1),
                 c      = (0,),
                 tol   = 2e-2):
    
    ndim          = len(b)  
    a0, a         = 0, np.zeros(ndim)
    fun, _, hess0 = sources.taylor(a0, a, b, c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    steps         = cu.ut_steps(bins)    
   
    hess          = simg.hessian(img, steps)
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    for i in range(ndim):
        for j in range(ndim):
            print('hess [', i, j,'] = ', hess0[i, j], ', mean ', np.mean(hess[i, j][center]))
            assert np.isclose(np.mean(hess[i, j][center]), hess0[i, j], atol = tol), \
                'not good enough hessian in coords [' + str(i) + ', ' + str(j) + ']'
        
    return
    

def test_laplacian(nbins  = 101, 
                   ranges = ((-1, 1), (-1, 1)),
                   b      = (1, -1),
                   c      = (0,),
                   tol   = 2e-2):
    
    ndim          = len(b)  
    a0, a         = 0, np.zeros(ndim)
    fun, _, hess0 = sources.taylor(a0, a, b, c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    steps         = cu.ut_steps(bins)    
   
    lap           = simg.laplacian(img, steps)
    ulap          = sum(hess0[i, i] for i in range(ndim))
    
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    print('laplacian = ', ulap, ', mean ', np.mean(lap[center]))
    assert np.isclose(np.mean(lap[center]), ulap, atol = tol), \
                'not good enough laplacian'
    
    return
    
def test_rev_matrix(h):
    
    ndim = h.shape[0] 
    hv   = simg._rev_matrix(h)
    
    for i in range(ndim):
        for j in range(ndim):
            #print(np.sum(h[i, j]), np.sum(hv[..., i, j]))
            assert np.isclose(np.sum(h[i, j]), np.sum(hv[..., i, j])), 'invalid rev matrix'
            
    return
    

def test_det_hessian(nbins  = 101, 
                     ranges = ((-1, 1), (-1, 1)),
                     b      = (1, -1),
                     c      = (0,),
                     tol   = 3e-2):

    ndim          = len(b)  
    a0, a         = 0, np.zeros(ndim)
    fun, _, hess0 = sources.taylor(a0, a, b, c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    steps         = cu.ut_steps(bins)    
   
    dhess         = simg.det_hessian(img, steps)  
    det           = np.linalg.det(hess0)
    
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    print('det hessian = ', det, ', mean ', np.mean(dhess[center]))
    assert np.isclose(np.mean(dhess[center]), det, atol = tol), \
                'not good enough hessian determinant'
            
    return
    

def test_curvature(nbins = 101,
                   ranges = ((-1, 1), (-1, 1)),
                   b      = (1, -1),
                   c      = (0,),
                   tol    = 3e-2):
    
    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = sources.taylor(b = b, c = c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    leig, eeig   = np.linalg.eigh(hess)

    for i in range(ndim):
        curv0 = leig[i]/2
        edir  = eeig[i]
        curv  = simg.curvature(img, edir, steps)
        print(' curvature i ', edir, ' curv0 ', curv0, 'mean curv', np.mean(curv[center]))
        assert np.isclose(np.mean(curv[center]), curv0, 0, atol = tol), \
            'not good enough curvature at dir ' + str(edir) 
        
    return


def test_curvatures(nbins = 101,
                    ranges = ((-1, 1), (-1, 1)),
                    b      = (1, -2),
                    c      = (0,),
                    tol    = 3e-2):
    
    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = sources.taylor(b = b, c = c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    leig, _ = np.linalg.eigh(hess)    
    lap0    = np.sum(leig)/2
    
    curvs   = simg.curvatures(img, steps)
    curv    = np.zeros(img.shape)
    for icurv in curvs: curv += icurv 
    
    print('sum curvatures ', np.mean(curv[center]), ', lap ', lap0)
    assert np.isclose(np.mean(curv[center]), lap0, atol = tol), \
        'Not good curvatures'
    
    return


def test_min_curvature(nbins = 101,
                       ranges = ((-1, 1), (-1, 1)),
                       b      = (1, -1),
                       c      = (0,),
                       tol    = 3e-2):
    
    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = sources.taylor(b = b, c = c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])

    leig, eeig   = np.linalg.eigh(hess)
    curv0, edir0 = leig[0], eeig[..., 0]
    curv , edir  = simg.min_curvature(img, steps)
    print(' min curv ', curv0, ', min curv ', np.mean(curv[center]))
    assert np.isclose(np.mean(curv[center]), curv0, atol = tol), \
        ' not good min curvature' 

    for i in range(ndim):
        print(' i-coor ', i, ' edir ', edir0[i], np.mean(edir[i][center]))
        assert np.isclose(np.mean(edir[i][center]), edir0[i], atol = tol), \
                ' not good min curvature dir ' + str(i) 


def test_min_transverse_curvature(nbins = 101,
                                  ranges = ((-1, 1), (-1, 1)),
                                  b      = (1, -1),
                                  c      = (0,),
                                  tol    = 3e-2):
    
    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = sources.taylor(b = b, c = c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])

    leig, eeig   = np.linalg.eigh(hess)
    curv0, edir0 = np.sum(leig[:-1]), eeig[..., -1]
    curv , edir  = simg.min_transverse_curvature(img, steps)
    print(' min transv curv0 ', curv0, ', min curv ', np.mean(curv[center]))
    assert np.isclose(np.mean(curv[center]), curv0, atol = tol), \
        ' not good min curvature' 

    mag = np.zeros(img.shape)
    for i in range(ndim): mag += edir[i] * edir0[i]
    print(' mag', np.mean(mag[center]))
    assert np.isclose(np.mean(mag[center]), 1, atol = tol), \
        ' not good min curvature dir '


def test_transverse_curvatures():
    assert True


def test_features(nbins = 101,
                  ranges = ((-1, 1), (-1, 1)),
                  a0     = 0,
                  a      = (1, -1),
                  b      = (1, -1),
                  c      = (0,),
                  tol    = 3e-2):
    
    ndim   = len(a)  
    def fun(x):
        y = a0
        for i in range(ndim): y += a[i] * x[i]
        return y
    
    img, bins = sources.from_function(fun, nbins, ranges)
    steps     = cu.ut_steps(bins)
    
    vgrad, _, _, _ = simg.features(img, steps)
    amod  = np.sqrt(sum([ai**2 for ai in a]))
    center = tuple([[nborder, -nborder] for i in range(ndim)])

    assert np.isclose(np.mean(vgrad[center]), amod, atol = tol), 'not good enough gradient modulus'
    
    ndim          = len(b)  
    a0, a         = 0, np.zeros(ndim)
    fun, _, hess0 = sources.taylor(a0, a, b, c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    steps         = cu.ut_steps(bins)    
   
    _, lap, dhess, lmin = simg.features(img, steps)
    
    ulap = np.sum([hess0[i, i] for i in range(ndim)])
    assert np.isclose(np.mean(lap[center]), ulap, atol = tol), 'not good enough laplacian'
 
    udhess = np.linalg.det(hess0)
    assert np.isclose(np.mean(dhess), udhess, atol = tol), 'not good enough hessian det'
 
    leig0, _ = np.linalg.eigh(hess0)
    assert np.isclose(np.mean(lmin), leig0[..., 0], atol = tol), 'not good min eigenvalue of hessian'

    return

#--- Filters

# def test_node_filter(npoints = 2, sigma = 1):
    
#     img, points = sources.points(npoints = npoints)
#     simg        = ndimg.gaussian_filter(img, sigma) if sigma > 0 else img 
    
#     xfil        = simg.node_filter(simg)
    
#     img = img.astype(bool)
    
#     assert np.all(img == xfil), 'not good node filter'
    

# def test_blob_filter(npoints = 2, sigma = 1):
    
#     img, points = sources.points(npoints = npoints)
#     simg        = ndimg.gaussian_filter(img, sigma) if sigma > 0 else img 
    
#     xfil        = simg.blob_filter(simg)
    
#     img = img.astype(bool)
    
#     assert np.all(img == xfil), 'not good node filter'
    
    
# def test_nlap_scan(npoints = 10, sigma = 1, maxradius = 10):
    
#     sigmas = np.linspace(0, 2 * maxradius, 40)
    
#     img, indices, radius = sources.disks(npoints = npoints, maxradius = maxradius)
#     simg          = ndimg.gaussian_filter(img, sigma) if sigma >0 else img
#     sigmax, _, _  = simg.nlap_scan(simg, sigmas = sigmas, filter = False)

#     radmu = [sigmax[index] for index in indices]
#     rat   = np.array(radius)/np.array(radmu)
#     print('mean ', np.mean(rat), 'std', np.std(rat))
#     assert np.isclose(np.mean(rat), 1.8, atol = 2 * np.std(rat))
#     return    