#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:24:54 2021

@author: hernando
"""


import numpy as np

#
#--- Utilities
#

def ut_fscale(values, a = 0, b = 1):
    
    values = values.astype(int) if values.dtype == bool else values
    xmin, xmax = np.min(values), np.max(values)
    def _fun(val):
        scale  = (val - xmin)/(xmax - xmin)
        return scale
    return _fun


def ut_scale(values, a = 0, b = 1):
   
    values = values.astype(int) if values.dtype == bool else values
    xmin, xmax = np.min(values), np.max(values)
    scale  = (values - xmin)/(xmax - xmin)
    return scale


def ut_steps(bins):
    steps  = np.array([ibin[1] - ibin[0] for ibin in bins])
    return steps


def ut_centers(xs : np.array) -> np.array:
    return 0.5* ( xs[1: ] + xs[: -1])


def ut_mesh(bins):

    centers = [ut_centers(ibin) for ibin in bins]
    # indexing 'ij', ensures access via coordinates x[i, j, k]
    xmesh  = np.meshgrid(*centers, indexing = 'ij')
    return xmesh
    
def arstep(x, step, delta = False):
    delta = step/2 if delta else 0.
    return np.arange(np.min(x) - delta, np.max(x) + step + delta, step)


def to_coors(vs):
    ndim = len(vs[0])
    xs = [np.array([vi[i] for vi in vs]) for i in range(ndim)]
    return xs


def ut_sort(values, ids, reverse = True):
    
    vals_ = sorted(zip(values, ids), reverse = reverse)
    vals  = np.array([v[0] for v in vals_])
    kids  = np.array([v[1] for v in vals_])
    return vals, kids