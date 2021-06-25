#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 11:43:09 2021

@author: hernando
"""

import numpy         as np

import scipy.ndimage as ndimg

#import numpy.linalg as nplang

#import clouds.ridges as ridges
import clouds.utils   as cu
import clouds.sources as csources

#import clouds.sclouds as sclouds
import clouds.dclouds as dclouds

nborder = 5    

def test_gradient(nbins  = 101,
                  ranges = ((-1, 1), (-1, 1)),
                  a0     = 0,
                  a      = (1, 1),
                  tol    = 5e-2):

    ndim             = len(a)
    fun, grad, hess  = csources.taylor(a0 = a0, a = a)
    img, bins        = csources.from_function(fun, nbins, ranges)
    img             += 1 - np.min(img)
    steps            = cu.ut_steps(bins)

    adelta, adir     = _dgrad(a, steps)
 
    vgrad, ndir   = dclouds.gradient(img, steps)
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    #print(' vgrad ', np.mean(vgrad[center]), adelta)
    assert np.isclose(np.mean(vgrad[center]), adelta, atol = tol), \
            'not good enough gradient '
            
    for i in range(ndim):
        #print(' grad [', i,']', np.mean(ndir[i][center]), adir[i])
        assert np.isclose(np.mean(ndir[i][center]), adir[i], atol = tol), \
            'not good enough gradient in coord ' + str(i)
        
    return
    

def test_curvature(nbins  = 101,
                   ranges  = ((-1, 1), (-1, 1)),
                   b       = (1, 1),
                   c       = (0,),
                   tol     = 5e-2):

    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = csources.taylor(b = b, c = c)
    img, bins     = csources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])


    for i, move in enumerate(dclouds.moves_face(ndim)):
        curv0 = _dcurv   (hess, move, steps)
        curv  = dclouds.curvature(img , move, steps)
        #print('move ', move, ' curv0', curv0, ' mean curv ', np.mean(curv[center]))
        assert np.isclose(np.mean(curv[center]), curv0, tol), \
            ' not good enough curvature in direction ' + str(move)        
    return

        
def test_laplacian(nbins  = 101,
                   ranges  = ((-1, 1), (-1, 1)),
                   b       = (1, 1),
                   c       = (0,),
                   tol     = 5e-2):
    
    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = csources.taylor(b = b, c = c)
    img, bins     = csources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])

    for extended in (True, False):
        lap0  =  _dlap(hess, steps, extended = extended)
        lap   =  dclouds.laplacian(img, steps, extended = extended)
        #print('move ', move, ' curv0', curv0, ' mean curv ', np.mean(curv[center]))
        assert np.isclose(np.mean(lap[center]), lap0, atol = tol), \
            ' not good laplacian (extended)' + str(extended) 
    return
    
def test_curvature_transverse(nbins  = 101,
                              ranges  = ((-1, 1), (-1, 1)),
                              b       = (1, 1),
                              c       = (0,),
                              tol     = 5e-2):
    
    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = csources.taylor(b = b, c = c)
    img, bins     = csources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])

    moves = dclouds.moves_face(ndim)
    for move in moves:
        curt0  = _dcurv_trans(hess, move, steps)
        curt   = dclouds.curvature_transverse(img, move, steps)
        #print('edir ', move, ' curvt0 ', curt0, ' mean curv ', np.mean(curt[center]))
        assert np.isclose(np.mean(curt[center]), curt0, atol = tol), \
            ' not good transv curvature ' + str(move)       
    return


def test_min_curvature(nbins  = 101,
                       ranges  = ((-1, 1), (-1, 1)),
                       b       = (1, -2),
                       c       = (0,),
                       tol     = 5e-2):
    
    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = csources.taylor(b = b, c = c)
    img, bins     = csources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])

    curv0, edir0 = _dcurv_min(hess, steps)
    curv , edir  = dclouds.min_curvature(img, steps)
    print(' curv0 ', curv0, ', min curv ', np.mean(curv[center]))
    assert np.isclose(np.mean(curv[center]), curv0, atol = tol), \
        ' not good min curvature' 

    for i in range(ndim):
        print(' i-coor ', i, ' edir ', edir0[i], np.mean(edir[i][center]))
        assert np.isclose(np.mean(edir[i][center]), edir0[i], atol = tol), \
                ' not good min curvature dir ' + str(i) 

    
    return
    
        
def test_node_filter(npoints = 2, sigma = 1):
    
    img, points = csources.points(npoints = npoints)
    simg        = ndimg.gaussian_filter(img, sigma) if sigma > 0 else img 
    mask        = simg > 0
    xfil        = dclouds.node_filter(simg, mask = mask)
    
    img = img.astype(bool)
    
    assert np.all(img == xfil), 'not good node filter'
    

def test_blob_filter(npoints = 1, sigma = 1):
    
    img, points = csources.points(npoints = npoints)
    simg        = ndimg.gaussian_filter(img, sigma) if sigma > 0 else img 
    
    xfil        = dclouds.blob_filter(simg)

    img = img.astype(bool)
    
    assert np.all(img == xfil), 'not good node filter'
    
        

#---- Internal math
    
def _dgrad(a, steps = None):
    """
    non differencial gradient of gradient a and steps
    """
    a      = np.array(a)
    ndim   = len(a)
    steps  = np.ones(ndim) if steps is None else steps
    
    adelta = 0
    adir   = np.zeros(ndim)
    for move in dclouds.moves(ndim):
        vmove  = steps * move
        mmove  = np.sqrt(np.sum(vmove * vmove))
        vdelta = np.sum(a * vmove)/mmove
        if (vdelta > adelta):
            adelta = vdelta
            adir   = move
    return adelta, adir


def _dcurv(hess, edir, steps = None):
    """
    Non differencial curvature
    """
    ndim   = edir.ndim
    steps  = np.ones(ndim) if steps is None else steps 
    vdir   = steps * edir
    c0     = np.dot(vdir.T, np.dot(hess, vdir))/2
    norma2 = np.sum(vdir * vdir)
    return   c0/norma2
        

def _dlap(hess, steps = None, extended = False):

    ndim  = hess.shape[0]
    moves = dclouds.moves_face(ndim) if extended else dclouds.moves_axis(ndim)
    lap   = 0
    for move in moves:
        lap += _dcurv(hess, move, steps)
    return lap
                   
                   
def _dcurv_trans(hess, edir, steps = None):
    
    ndim  = hess.shape[0]
    moves = dclouds.moves_face(ndim)

    curvt = 0 # temporal holder to return all the curves
    for move in moves:
        mcur      = _dcurv(hess, edir, steps)
        vdot      = 0
        for i in range(ndim):
            vdot += edir[i] * move[i]
        if np.isclose(vdot, 0): curvt += mcur
    return curvt
    

def _dcurv_min(hess, steps = None):
    
    ndim  = hess.shape[0]
    moves = dclouds.moves_face(ndim)

    mcurv = 0
    mdir  = np.zeros(ndim)
    for move in moves:
        icur = _dcurv(hess, move, steps)
        if (icur < mcurv):
            mcurv = icur
            mdir  = move
    
    return mcurv, mdir


# def test_hessian(nbins  = 101, 
#                  ranges = ((-1, 1), (-1, 1)),
#                  b      = (1, -1),
#                  c      = (0,),
#                  rtol   = 2e-2):
    
#     ndim          = len(b)  
#     a0, a         = 0, np.zeros(ndim)
#     fun, _, hess0 = csources.get_taylor(a0, a, b, c)
#     img, bins     = csources.from_function(fun, nbins, ranges)
#     steps         = cu.ut_steps(bins)    
   
#     hess          = sclouds.hessian(img, steps)
#     center = np.array([np.arange(5, nbins - 5) for i in range(ndim)])
    
#     for i in range(ndim):
#         for j in range(ndim):
#             print('hess [', i, j,'] = ', hess0[i, j], ', mean ', np.mean(hess[i, j][center]))
#             assert np.isclose(np.mean(hess[i, j][center]), hess0[i, j], rtol = rtol), \
#                 'not good enough hessian in coords [' + str(i) + ', ' + str(j) + ']'
        
#     return
    

# def test_laplacian(nbins  = 101, 
#                    ranges = ((-1, 1), (-1, 1)),
#                    b      = (1, -1),
#                    c      = (0,),
#                    rtot   = 2e-2):
    
#     ndim          = len(b)  
#     a0, a         = 0, np.zeros(ndim)
#     fun, _, hess0 = csources.get_taylor(a0, a, b, c)
#     img, bins     = csources.from_function(fun, nbins, ranges)
#     steps         = cu.ut_steps(bins)    
   
#     lap           = sclouds.laplacian(img, steps)
#     ulap          = sum(hess0[i, i] for i in range(ndim))
    
#     center = np.array([np.arange(5, nbins - 5) for i in range(ndim)])
    
#     print('laplacian = ', ulap, ', mean ', np.mean(lap[center]))
#     assert np.isclose(np.mean(lap[center]), ulap, rtol = rtot), \
#                 'not good enough laplacian'
    
#     return
    
# def test_rev_matrix(h):
    
#     ndim = h.shape[0] 
#     hv   = sclouds._rev_matrix(h)
    
#     for i in range(ndim):
#         for j in range(ndim):
#             #print(np.sum(h[i, j]), np.sum(hv[..., i, j]))
#             assert np.isclose(np.sum(h[i, j]), np.sum(hv[..., i, j])), 'invalid rev matrix'
            
#     return
    

# def test_det_hessian(nbins  = 101, 
#                      ranges = ((-1, 1), (-1, 1)),
#                      b      = (1, -1),
#                      c      = (0,),
#                      rtot   = 3e-2):

#     ndim          = len(b)  
#     a0, a         = 0, np.zeros(ndim)
#     fun, _, hess0 = csources.get_taylor(a0, a, b, c)
#     img, bins     = csources.from_function(fun, nbins, ranges)
#     steps         = cu.ut_steps(bins)    
   
#     dhess         = sclouds.det_hessian(img, steps)  
#     det           = np.linalg.det(hess0)
    
#     center = np.array([np.arange(5, nbins - 5) for i in range(ndim)])
    
#     print('det hessian = ', det, ', mean ', np.mean(dhess[center]))
#     assert np.isclose(np.mean(dhess[center]), det, rtol = rtot), \
#                 'not good enough hessian determinant'
            
#     return
    

# def test_node_filter(npoints = 2, sigma = 1):
    
#     img, points = csources.points(npoints = npoints)
#     simg        = ndimg.gaussian_filter(img, sigma) if sigma > 0 else img 
    
#     xfil        = sclouds.node_filter(simg)
    
#     img = img.astype(bool)
    
#     assert np.all(img == xfil), 'not good node filter'
    

# def test_blob_filter(npoints = 2, sigma = 1):
    
#     img, points = csources.points(npoints = npoints)
#     simg        = ndimg.gaussian_filter(img, sigma) if sigma > 0 else img 
    
#     xfil        = sclouds.blob_filter(simg)
    
#     img = img.astype(bool)
    
#     assert np.all(img == xfil), 'not good node filter'
    
    
# def test_nlap_scan(npoints = 10, sigma = 1, maxradius = 10):
    
#     sigmas = np.linspace(0, 2 * maxradius, 40)
    
#     img, indices, radius = csources.disks(npoints = npoints, maxradius = maxradius)
#     simg          = ndimg.gaussian_filter(img, sigma) if sigma >0 else img
#     sigmax, _, _  = sclouds.nlap_scan(simg, sigmas = sigmas, filter = False)

#     radmu = [sigmax[index] for index in indices]
#     rat   = np.array(radius)/np.array(radmu)
#     print('mean ', np.mean(rat), 'std', np.std(rat))
#     assert np.isclose(np.mean(rat), 1.8, atol = 2 * np.std(rat))
#     return    