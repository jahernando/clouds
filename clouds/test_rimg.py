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
import clouds.sources as sources

#import clouds.sclouds as sclouds
import clouds.rimg    as rimg

nborder = 5    

def test_gradient(nbins  = 51,
                  ranges = ((-1, 1), (-1, 1)),
                  a0     = 0,
                  a      = (1, 1),
                  tol    = 5e-2):

    ndim             = len(a)
    fun, grad, hess  = sources.taylor(a0 = a0, a = a)
    img, bins        = sources.from_function(fun, nbins, ranges)
    img             += 1 - np.min(img)
    steps            = cu.ut_steps(bins)

    adelta, adir     = _dgrad(a, steps)
 
    vgrad, ndir   = rimg.gradient(img, steps)
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    print(' vgrad ', np.mean(vgrad[center]), adelta)
    assert np.isclose(np.mean(vgrad[center]), adelta, atol = tol), \
            'not good enough gradient '
            
    for i in range(ndim):
        print(' grad [', i,']', np.mean(ndir[i][center]), adir[i])
        assert np.isclose(np.mean(ndir[i][center]), adir[i], atol = tol), \
            'not good enough gradient in coord ' + str(i)
        
    return
    

def test_curvature(nbins  = 51,
                   ranges  = ((-1, 1), (-1, 1)),
                   b       = (1, 1),
                   c       = (0,),
                   tol     = 5e-2):

    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = sources.taylor(b = b, c = c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])


    for i, move in enumerate(rimg.moves_face(ndim)):
        curv0 = _dcurv   (hess, move, steps)
        curv  = rimg.curvature(img , move, steps)
        #print('move ', move, ' curv0', curv0, ' mean curv ', np.mean(curv[center]))
        assert np.isclose(np.mean(curv[center]), curv0, tol), \
            ' not good enough curvature in direction ' + str(move)        
    return


def test_curvatures(nbins   = 51,
                   ranges  = ((-1, 1), (-1, 1)),
                   b       = (1, 1),
                   c       = (0,),
                   tol     = 5e-2):
    
    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = sources.taylor(b = b, c = c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])
    
    
    for extended in (True, False):
        curvs0 = _dcurvs(hess, steps, extended = extended)
    
        curvs  = rimg.curvatures(img, steps, extended = extended)
    
        for c0, c in zip(curvs0, curvs):
            print('curve ', c0, ' mean ', np.mean(c[center]))
            assert np.isclose(np.mean(c[center]), c0, atol = tol), \
                'not good enough curvature ' + str(extended)
    return 
    

def test_laplacian(nbins   = 51,
                   ranges  = ((-1, 1), (-1, 1)),
                   b       = (1, 1),
                   c       = (0,),
                   tol     = 5e-2):
    
    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = sources.taylor(b = b, c = c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])

    for extended in (True, False):
        lap0  =  _dlap(hess, steps, extended = extended)
        lap   =  rimg.laplacian(img, steps, extended = extended)
        print('lap0', lap0, ' mean lap ', np.mean(lap[center]))
        assert np.isclose(np.mean(lap[center]), lap0, atol = tol), \
            ' not good laplacian (extended)' + str(extended) 
    return
    

def test_min_curvature(nbins  = 51,
                       ranges  = ((-1, 1), (-1, 1)),
                       b       = (1, -2),
                       c       = (0,),
                       tol     = 5e-2):
    
    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = sources.taylor(b = b, c = c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])

    curv0, edir0 = _dcurv_min(hess, steps)
    curv , edir  = rimg.min_curvature(img, steps)
    
    print(' curv0 ', curv0, ', min curv ', np.mean(curv[center]))
    assert np.isclose(np.mean(curv[center]), curv0, atol = tol), \
        ' not good min curvature' 
    for i in range(ndim):
        print(' i-coor ', i, ' edir ', edir0[i], np.mean(edir[i][center]))
        assert np.isclose(np.mean(edir[i][center]), edir0[i], atol = tol), \
                ' not good min curvature dir ' + str(i) 

    
    return       


def test_transverse_curvature(nbins  = 101,
                              ranges  = ((-1, 1), (-1, 1)),
                              b       = (1, 1),
                              c       = (0,),
                              tol     = 5e-2):
    
    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = sources.taylor(b = b, c = c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])

    moves = rimg.moves_face(ndim)
    for move in moves:
        curt0  = _dcurv_trans(hess, move, steps)
        curt   = rimg.transverse_curvature(img, move, steps)
        print('edir ', move, ' curvt0 ', curt0, ' mean curv ', np.mean(curt[center]))
        assert np.isclose(np.mean(curt[center]), curt0, atol = tol), \
            ' not good transv curvature ' + str(move)       
    return


def test_min_transverse_curvature(nbins  = 101,
                                  ranges  = ((-1, 1), (-1, 1)),
                                  b       = (-1, 1),
                                  c       = (0,),
                                  tol     = 5e-2):
    
    ndim          = len(b)  
    assert len(ranges) == ndim, 'not valid number of ranges'
    
    ndim          = len(b)
    fun, _, hess  = sources.taylor(b = b, c = c)
    img, bins     = sources.from_function(fun, nbins, ranges)
    img          +=  1 - np.min(img) 
    steps         = cu.ut_steps(bins)
    center = tuple([[nborder, -nborder] for i in range(ndim)])


    curv0, edir0 = _dcurv_trans_min(hess, steps)
    curv , edir  = rimg.min_transverse_curvature(img, steps)
    
    print(' curv0 ', curv0, ', min curv ', np.mean(curv[center]))
    assert np.isclose(np.mean(curv[center]), curv0, atol = tol), \
        ' not good min trans curvature' 
    for i in range(ndim):
        print(' i-coor ', i, ' edir ', edir0[i], np.mean(edir[i][center]))
        assert np.isclose(np.mean(edir[i][center]), edir0[i], atol = tol), \
                ' not good min trans curvature dir ' + str(i) 
    
    return
    
#
#  Test filters
#
        
def test_edge_filter(nbins = 81, sigma = 4, atol = 5e-1):
    

    img  = np.zeros((nbins, nbins))
    n0  = int(nbins/2)
    img[:, n0:] = 1

    img =  ndimg.gaussian_filter(img, sigma)


    for math in ('False', 'True'):
        xfil, rv  = rimg.edge_filter(img, math_condition = math,
                                     perc = 100, atol = atol)
        
        xi = [x[1] for x in np.argwhere(xfil == True)]
        print('edge: mean ', np.mean(xi), n0)
        assert np.isclose(np.mean(xi), n0, 1)
        
    return
    

def test_ridge_lambda_filter(nbins  = 101,
                              ranges = ((0, 10), (0, 10)),
                              y0     = 4,
                              atol   = 5e-2):
    
    fun    = lambda x : x[0] - (x[1] - y0)**2

    img, bins = sources.from_function(fun, nbins, ranges)
    steps     = [bin[1] - bin[0] for bin in bins]
    xmesh     = cu.ut_mesh(bins)

    xfil, rv   = rimg.ridge_lambda_filter(img, steps)

    ndim = img.ndim
    sel = np.full(xfil.shape, True)
    for i in range(ndim):
        sel = (sel) & (xmesh[i] > bins[i][nborder]) & (xmesh[i] < bins[i][-nborder])

    ys = xmesh[1][sel & xfil]
    
    print('ridge lambda : ', np.mean(ys), y0)
    assert np.isclose(np.mean(ys), y0, atol = atol), \
        'Not good ridge lambda'    

    return
    

def test_ridge_filter(nbins  = 101,
                      ranges = ((0, 10), (0, 10)),
                      y0     = 4,
                      atol   = 5e-2):
    
    fun    = lambda x : x[0] - (x[1] - y0)**2

    img, bins = sources.from_function(fun, nbins, ranges)
    steps     = [bin[1] - bin[0] for bin in bins]
    xmesh     = cu.ut_mesh(bins)

    xfil, rv  = rimg.ridge_filter(img, steps)

    ndim = img.ndim
    sel = np.full(xfil.shape, True)
    for i in range(ndim):
        sel = (sel) & (xmesh[i] > bins[i][nborder]) & (xmesh[i] < bins[i][-nborder])

    ys = xmesh[1][sel & xfil]
    
    print('ridge : ', np.mean(ys), y0)
    assert np.isclose(np.mean(ys), y0, atol = atol), \
        'Not good ridge'

    return
    

def test_node_filter(npoints = 2, sigma = 1):
    
    img, points = sources.points(npoints = npoints)
    ximg        = ndimg.gaussian_filter(img, sigma) if sigma > 0 else img 
    mask        = ximg > 0
    
    xfil        = rimg.node_filter(ximg, mask = mask)
    
    img = img.astype(bool)
    
    print('node filter :', npoints, np.sum(xfil))

    assert np.all(img == xfil), 'not good node filter'
    

def test_blob_filter(npoints = 2, sigma = 1):
    
    img, points = sources.points(npoints = npoints)
    ximg        = ndimg.gaussian_filter(img, sigma) if sigma > 0 else img 
    mask        = ximg > 0
    
    xfil        = rimg.blob_filter(ximg, mask = mask)
    
    img = img.astype(bool)
    
    print('blob filter :', npoints, np.sum(xfil))
    assert np.all(img == xfil), 'not good blob filter'
    

def test_nlap_scan(npoints = 10, sigma = 1, maxradius = 10):
    
    sigmas = np.linspace(0, 2 * maxradius, 40)
    
    img, indices, radius = sources.disks(npoints = npoints, maxradius = maxradius)
    ximg          = ndimg.gaussian_filter(img, sigma) if sigma >0 else img
    sigmax, _, _  = rimg.nlap_scan(ximg, sigmas = sigmas, filter = False)

    radmu = [sigmax[index] for index in indices]
    rat   = np.array(radius)/np.array(radmu)
    print('nlap scal: nmean ', np.mean(rat), 'std', np.std(rat))
    assert np.isclose(np.mean(rat), 1.8, atol = 2 * np.std(rat))
    return    

        
#
#  Internal functions
#
    
def _dgrad(a, steps = None):
    """
    non differencial gradient of gradient a and steps
    """
    a      = np.array(a)
    ndim   = len(a)
    steps  = np.ones(ndim) if steps is None else steps
    
    adelta = 0
    adir   = np.zeros(ndim)
    for move in rimg.moves(ndim):
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
        

def _dcurvs(hess, steps = None, extended = False):

    ndim  = hess.shape[0]
    moves = rimg.moves_face(ndim) if extended else rimg.moves_axis(ndim)
    curvs = [_dcurv(hess, move, steps) for move in moves]
    return curvs


def _dlap(hess, steps = None, extended = False):

    ndim  = hess.shape[0]
    moves = rimg.moves_face(ndim) if extended else rimg.moves_axis(ndim)
    lap   = 0
    for move in moves:
        lap += _dcurv(hess, move, steps)
    return lap
                   
                   
def _dcurv_trans(hess, edir, steps = None):
    
    ndim  = hess.shape[0]
    moves = rimg.moves_face(ndim)

    curvt = 0 # temporal holder to return all the curves
    for move in moves:
        mcur      = _dcurv(hess, move, steps)
        vdot      = 0
        for i in range(ndim):
            vdot += edir[i] * move[i]
        if np.isclose(vdot, 0): curvt += mcur
    return curvt
    

def _dcurv_min(hess, steps = None):
    
    ndim  = hess.shape[0]
    moves = rimg.moves_face(ndim)

    mcurv = 0
    mdir  = np.zeros(ndim)
    for move in moves:
        icur = _dcurv(hess, move, steps)
        if (icur < mcurv):
            mcurv = icur
            mdir  = move
    
    return mcurv, mdir


def _dcurv_trans_min(hess, steps = None):

    ndim  = hess.shape[0]
    moves = rimg.moves_face(ndim)

    mcurv = 0
    mdir  = np.zeros(ndim)
    for move in moves:
        curv = _dcurv_trans(hess, move, steps = steps)
        if (curv < mcurv):
            mcurv = curv
            mdir  = move
    
    return mcurv, mdir