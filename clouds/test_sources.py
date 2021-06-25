#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 10:52:53 2021

@author: hernando
"""

import numpy         as np

#import numpy.linalg as nplang

#import clouds.ridges as ridges
import clouds.utils   as cu
import clouds.sources as csources


def test_taylor(x, a0, a, b, c):
    
    ndim = len(a)
    
    fun, grad, hess  = csources.taylor(a0, a, b, c)
    tfun             = _tfun(a0, a, b, c)

    #print('x     :', x)
    #print('grad   ', grad)
    #print('hess   ', hess)
    #print('fun    ', tfun(x))
    #print('taylor ', fun(x))
    
    assert np.all(np.isclose(fun(x), tfun(x)))  , 'not same function value'
    assert np.all(np.isclose(np.array(a), grad)), 'not same gradient'
    assert np.all([hess[i, i] == b[i] for i in range(ndim)]), 'hess trace non valid'
    k = 0
    for i in range(ndim):
        for j in range(i + 1, ndim):
            assert hess[i, j] == hess[j, i], 'hess non symmetric'
            assert hess[i, j] == c[k]      , 'hess non valid diagonal element'
            k += 1
                
    return 


def test_img_function(img, bins, fun):
        
    xmesh   = cu.ut_mesh(bins)
    zs      = fun(xmesh)

    #print('img ', img)
    #print('fun ', zs)
    assert np.all(np.isclose(img, zs)), 'img is not the function'

    return    


def test_from_function(fun, nbins, ranges):
    
    img, bins = csources.from_function(fun, nbins, ranges)
    test_img_function(img, bins, fun)
    
    return
    

#---- Internal

def _tfun(a0, a, b, c):

    ndim = len(a)
    
    def _fun(x):
        y   = a0
        for i in range(ndim):
            y  += a[i] * x[i]
        k   = 0
        for i in range(ndim):
            y += (b[i] * x[i] * x[i]) / 2
            for j in range(i+1, ndim):
                y += c[k] * x[i] * x[j] 
                k += 1
        return y
    return _fun