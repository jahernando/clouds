#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 11:43:09 2021

@author: hernando
"""

import numpy         as np
#import scipy.ndimage as ndimg

import clouds.utils     as cu
import clouds.sources   as sources
import clouds.simg      as simg

import clouds.test_filters as tfilters

from   clouds.pclouds   import fig, ax, efig, title, voxels, scatter, quiver

nborder = 3
debug   = False
plot    = False

def test_gradient(nbins  = 101,
                  width  = 10,
                  a0     = 0,
                  a      = (1, -1),
                  atol   = 1e-3):
    
    ndim   = len(a)  
    b      = np.zeros(ndim)
    c      = (0,) if ndim == 2 else np.zeros(ndim)
    fun, grad0, _ = sources.taylor(a0 = a0, a = a, b = b, c = c)
    
    ranges = ndim * ((a0 - width, a0 + width),)
    
    img, bins = sources.from_function(fun, nbins, ranges)
    steps     = cu.ut_steps(bins)
    
    vgrad, edir  = simg.gradient(img, steps) 
    grad  = vgrad * edir

    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    if (plot):
        fig(1, 3)
        draw = voxels if ndim == 2 else scatter
        ax(1, 3, 1, ndim); draw(img  , bins) ; title('img')
        ax(1, 3, 2, ndim); draw(vgrad, bins); title('grad')
        ax(1, 3, 3, ndim); quiver(vgrad, edir, bins); title('grad');
        efig();
    
    for i in range(ndim):
        if (debug):
            print('dir ', i, np.mean(grad[i][center]), grad0[i])
        assert np.isclose(np.mean(grad[i][center]), grad0[i], atol = atol), \
                'not good enough gradient in coord ' + str(i)
        
    return
    

def test_hessian(nbins  = 101, 
                 width  = 10,
                 b      = (1, -1),
                 c      = (0,),
                 atol   = 2e-2):
    
    ndim          = len(b)  
    a0, a         = 0, np.zeros(ndim)
    fun, _, hess0 = sources.taylor(a0, a, b, c)
    ranges        = ndim * ((a0 - width, a0 + width),)
    img, bins     = sources.from_function(fun, nbins, ranges)
    steps         = cu.ut_steps(bins)    
   
    hess          = simg.hessian(img, steps)
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    for i in range(ndim):
        for j in range(ndim):
            if (debug):
                print('hess [', i, j,'] = ', hess0[i, j], ', mean ', np.mean(hess[i, j][center]))
            assert np.isclose(np.mean(hess[i, j][center]), hess0[i, j], atol = atol), \
                'not good enough hessian in coords [' + str(i) + ', ' + str(j) + ']'
        
    return
    

def test_laplacian(nbins  = 101, 
                   width  = 10,
                   b      = (1, -1),
                   c      = (0,),
                   atol   = 2e-2):
    
    ndim          = len(b)  
    a0, a         = 0, np.zeros(ndim)
    fun, _, hess0 = sources.taylor(a0, a, b, c)
    ranges        = ndim * ((a0 - width, a0 + width),)
    img, bins     = sources.from_function(fun, nbins, ranges)
    steps         = cu.ut_steps(bins)    
   
    lap           = simg.laplacian(img, steps)
    ulap          = sum(hess0[i, i] for i in range(ndim))
    
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    if (plot):
        fig(1, 2)
        draw = voxels if ndim == 2 else scatter
        ax(1, 2, 1, ndim); draw(img, bins); title('img')
        ax(1, 2, 2, ndim); draw(lap, bins); title('lap')
        efig();
    
    if (debug):
        print('laplacian = ', ulap, ', mean ', np.mean(lap[center]))
    assert np.isclose(np.mean(lap[center]), ulap, atol = atol), \
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
                     width  = 10,
                     b      = (1, -1),
                     c      = (0,),
                     atol   = 3e-2):

    ndim          = len(b)  
    a0, a         = 0, np.zeros(ndim)
    fun, _, hess0 = sources.taylor(a0, a, b, c)
    ranges        = ndim * ((a0 - width, a0 + width),)
    img, bins     = sources.from_function(fun, nbins, ranges)
    steps         = cu.ut_steps(bins)    
   
    dhess         = simg.det_hessian(img, steps)  
    det           = np.linalg.det(hess0)
    
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    if (debug):
        print('det hessian = ', det, ', mean ', np.mean(dhess[center]))
    assert np.isclose(np.mean(dhess[center]), det, atol = atol), \
                'not good enough hessian determinant'
            
    return
    

def test_curvature(nbins = 101,
                   width = 10,
                   b     = (1, -1),
                   c     = (0,),
                   atol  = 3e-2):
        
    ndim          = len(b)
    a0, a         = 0, np.zeros(ndim)
    fun, _, hess  = sources.taylor(a0 = a0, a = a, b = b, c = c)
    ranges        = ndim * ((a0 - width, a0 + width),)
    
    img, bins     = sources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    leig, eeig   = np.linalg.eigh(hess)

    for i in range(ndim):
        curv0 = leig[i]/2
        edir  = eeig[i]
        curv  = simg.curvature(img, edir, steps)
        if (debug):
            print(' curvature i ', edir, ' curv0 ', curv0, 
                  'mean curv', np.mean(curv[center]))
            
        if (plot):
            fig(1, 2)
            draw = voxels if ndim == 2 else scatter
            ax(1, 2, 1, ndim); draw(img, bins) ; title('img')
            ax(1, 2, 2, ndim); draw(curv, bins); title('curv' + str(i))
            efig();

        assert np.isclose(np.mean(curv[center]), curv0, 0, atol = atol), \
            'not good enough curvature at dir ' + str(edir) 
        
    return


def test_curvatures(nbins = 101,
                    width = 10,
                    b     = (1, -1),
                    c     = (0,),
                    atol  = 3e-2):
    
    ndim          = len(b)
    a0, a         = 0, np.zeros(ndim)
    fun, _, hess  = sources.taylor(a0 = a0, a = a, b = b, c = c)
    ranges        = ndim * ((a0 - width, a0 + width), )
    img, bins     = sources.from_function(fun, nbins, ranges)
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    leig, _ = np.linalg.eigh(hess)    
    lap0    = np.sum(leig)/2
    
    curvs   = simg.curvatures(img, steps)
    curv    = np.zeros(img.shape)
    for icurv in curvs: curv += icurv 
    
    if (plot):
        fig(1, 2)
        draw = voxels if ndim == 2 else scatter
        ax(1, 2, 1, ndim = ndim); draw(img , bins); title('img')
        ax(1, 2, 2, ndim = ndim); draw(curv, bins); title('sum curv')
        efig();
    
    if (debug):
        print('sum curvatures ', np.mean(curv[center]), ', lap ', lap0)
        
    assert np.isclose(np.mean(curv[center]), lap0, atol = atol), \
        'Not good curvatures'
    
    return


def test_min_curvature(nbins = 101,
                       width = 10,
                       b     = (1, -1),
                       c     = (0,),
                       atol  = 5e-2):
    
    
    ndim          = len(b)
    a0, a         = 0, np.zeros(ndim)
    fun, _, hess  = sources.taylor(a0 = a0, a = a, b = b, c = c)
    ranges        = ndim * ((a0 - width, a0 + width), )
    img, bins     = sources.from_function(fun, nbins, ranges)
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])

    leig, eeig   = np.linalg.eigh(hess)
    curv0, edir0 = leig[0], eeig[..., 0]
    curv , edir  = simg.min_curvature(img, steps)
    
    if (debug):
        print(' min curv ', curv0, ', min curv ', np.mean(curv[center]))
        
    assert np.isclose(np.mean(curv[center]), curv0, atol = atol), \
        ' not good min curvature' 

    if (plot):
        fig(1, 3)
        draw = voxels if ndim == 2 else scatter
        ax(1, 3, 1, ndim); draw  (img , bins); title('img')
        ax(1, 3, 2, ndim); draw  (curv, bins); title('min curv')
        ax(1, 3, 3, ndim); quiver(curv, edir, bins); title('edir')
        efig()

    for i in range(ndim):
        if (debug):
            print(' i-coor ', i, ' edir ', 
                  edir0[i], np.mean(np.abs(edir[i][center])))
        assert np.isclose(np.mean(np.abs(edir[i][center])), edir0[i], atol = atol), \
                ' not good min curvature dir ' + str(i) 


def test_min_transverse_curvature(nbins = 101,
                                  width = 10,
                                  b     = (1, -1),
                                  c     = (0,),
                                  atol  = 3e-2):
    
    ndim          = len(b)
    a0, a         = 0, np.zeros(ndim)
    fun, _, hess  = sources.taylor(a0 = a0, a = a, b = b, c = c)
    ranges        = ndim * ((a0 - width, a0 + width), )
    img, bins     = sources.from_function(fun, nbins, ranges)
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])

    leig, eeig   = np.linalg.eigh(hess)
    curv0, edir0 = np.sum(leig[:-1]), eeig[..., -1]
    curv , edir  = simg.min_transverse_curvature(img, steps)

    if (debug):
        print(' min transv curv0 ', curv0, ', min curv ', np.mean(curv[center]))

    assert np.isclose(np.mean(curv[center]), curv0, atol = atol), \
        ' not good min curvature' 
    
    if (plot):
        fig(1, 3)
        draw = voxels if ndim == 2 else scatter
        ax(1, 3, 1, ndim); draw(img , bins); title('img')
        ax(1, 3, 2, ndim); draw(curv, bins); title('min trans curv')
        ax(1, 3, 3, ndim); quiver(curv, edir, bins); title('edir')
        efig()

    mag = np.zeros(img.shape)
    for i in range(ndim): mag += edir[i] * edir0[i]
    if (debug):
        print(' mag', np.mean(mag[center]))
    assert np.isclose(np.mean(mag[center]), 1, atol = atol), \
        ' not good min curvature dir '

    for i in range(ndim):
        if (debug):
            print(' i-coor ', i, ' edir ', 
                  edir0[i], np.mean(np.abs(edir[i][center])))
        assert np.isclose(np.mean(np.abs(edir[i][center])), edir0[i], atol = atol), \
                ' not good min tranvs curvature dir ' + str(i) 



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
    
test_edge_filter         = tfilters.get_test_edge_filter(simg.edge_filter)
test_ridge_lambda_filter = tfilters.get_test_ridge_lambda_filter(simg.ridge_lambda_filter)
test_ridge_filter        = tfilters.get_test_ridge_filter(simg.ridge_filter)
test_node_filter         = tfilters.get_test_node_filter(simg.node_filter)
test_blob_filter         = tfilters.get_test_blob_filter(simg.blob_filter)
test_nlap_scan           = tfilters.get_test_nlap_scan(simg.nlap_scan) 