#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Module to find features in non diferentiable clouds

Created on Tue Apr 27 15:24:01 2021

@author: hernando

"""

import numpy             as np

import clouds.utils      as cu
import clouds.filters    as filters

#import scipy.ndimage     as ndimg

#import scipy.sparse.csgraph as scgraph

#import collections
#import functools
#import operator

#import clouds.ridges     as ridges
#import tables            as tb

#from collections import namedtuple


#
#  get elements
#

    
#--- Filters



# def edge_filter(img        : np.array,
#                 steps      : tuple = None,
#                 mask       : np.array = None,
#                 hard       : bool = True,
#                 perc       : float = 100):

#     shape = img.shape
#     mask  = np.full(shape, True) if mask is None else mask
    
#     bins, cells, enes = _cells(img, steps, mask = mask)
#     egrad, edir, _    = _gradient(bins, mask, cells, enes)
#     efil   = mask.flatten()
#     if (hard):
#         efil              = _edge_filter(bins, mask, cells, egrad, edir)
    
#     nfil        = np.full(shape, False, dtype = bool)
#     ngrad       = np.zeros(shape)
#     nfil [mask] = efil
#     ngrad[mask] = egrad
        
#     cut0  = np.percentile(egrad[efil], 100 - perc)    
#     umask = ngrad > cut0
#     nfil [~umask] = False
    
#     #nfil, _ = np.histogramdd(cells, bins, weights = efil)
#     return nfil, ngrad
 
    
# def ridge_filter(img   : np.array,
#                  steps : tuple = None,
#                  perc  : float = 100,
#                  hard: bool = True,
#                  mask  : np.array = None):
    
#     shape = img.shape
#     mask  = np.full(shape, True) if mask is None else mask

#     bins, cells, enes = _cells(img, steps, mask = mask)
#     egrad, edir, _    = _gradient(bins, mask, cells, enes)

#     xfil        = mask.flatten()
#     rv, _, _, _ = _curvature_transverse(bins, mask, cells, enes, edir)
    
#     if (hard):
#         xfil, rv = _ridge_filter(bins, mask, cells, enes, edir)
    
#     xfil = xfil if hard else mask.flatten()
    
#     nfil       = np.full(shape, False, dtype = bool)
#     nval       = np.zeros(shape)
#     nfil[mask] = xfil
#     nval[mask] = rv
    
#     cut0  = np.percentile(rv, perc)    
#     umask = nval <= cut0
#     nfil [~umask] = False
    
#     return nfil, nval


# def ridge_lambda_filter(img   : np.array,
#                         steps : tuple = None,
#                         perc  : float = 100,
#                         hard : bool = True,
#                         mask  : np.array = None):
    
#     shape = img.shape
#     mask  = np.full(shape, True) if mask is None else mask

#     egrad, edir = gradient     (img, steps, mask)
#     mcur , mdir = min_curvature(img, steps, mask)

#     xfil = mask & (mcur < 0)
#     if (hard):
#         xfil = (xfil) & (np.isclose(np.sum(edir * mdir, axis = 0), 0))

#     if (np.sum(xfil) > 0):
#         cut0  = np.percentile(mcur[xfil], perc)    
#         xfil = (xfil) & (mcur < cut0)
    
#     nfil = np.full(shape, False)
#     nfil[xfil] = True
    
#     return nfil, mcur



# def node_filter(img   : np.array,
#                 steps : tuple = None,
#                 mask  : np.array = None):
    
#     shape = img.shape
#     mask  = np.full(shape, True) if mask is None else mask
#     bins, cells, enes = _cells(img, steps, mask = mask)
#     egrad, _, _       = _gradient(bins, mask, cells, enes)
    
#     isnode  = np.isclose(egrad, 0)
    
#     nfil, _ = np.histogramdd(cells, bins, weights = isnode)
    
#     return nfil.astype(bool)
    

# def blob_filter(img : np.array,
#                 steps = None,
#                 mask  = None,
#                 extended = False):
    
#     shape = img.shape
#     mask  = np.full(shape, True) if mask is None else mask
#     lap  = laplacian(img, steps, mask, extended)
#     umask = (mask) & (-lap >0)
#     nfil = node_filter(-lap, steps, mask = umask)
    
#     #bins, mask, cells, enes = mask(img, steps, mask = mask)
#     #lap                     = _laplacian(bins, mask, cells, enes, extended = extended)
    
#     #nlap, _  = np.histogramdd(cells, bins, weights = -lap)
#     #nfil     = node_filter(nlap, steps)

#     return nfil

    
#--- math objects


def gradient(img   : np.array,
             steps : tuple = None,
             mask  : np.array = None):
    
    ndim  = img.ndim
    shape = img.shape 
    mask  = np.full(shape, True) if mask is None else mask
    bins, cells, enes = _cells(img, steps, mask = mask)
        
    egrad, edir, _ =  _gradient(bins, mask, cells, enes)
    
    ngrad = np.zeros(shape)
    ngrad[mask] = egrad
    
    ndir = np.zeros((ndim,) + img.shape)
    for i in range(ndim):
        ndir[i, mask] = edir[:, i]
    
    return ngrad, ndir
    
    
    
def curvature(img, edir, steps = None, mask = None):
    
    mask  = np.full(img.shape, True) if mask is None else mask

    bins, cells, enes = _cells(img, steps, mask = mask)
        
    curv        =  _curvature(bins, mask, cells, enes, edir)
    #ncurv       = np.zeros(img.shape)
    #ncurv[mask] = curv
    ncurv, _ = np.histogramdd(cells, bins, weights = curv)
    
    return ncurv
    

def curvatures(img, steps = None, mask = None, extended = False):
    
    mask  = np.full(img.shape, True) if mask is None else mask

    bins, cells, enes = _cells(img, steps, mask = mask)
    curvs             = _curvatures(bins, mask, cells, enes, extended = extended)
    
    ncurvs = []
    for curv in curvs:
        ncurv       = np.zeros(img.shape)
        ncurv[mask] = curv
        #ncurv, _ = np.histogram(cells, bins, weights = curv)
        ncurvs.append(ncurv)
    
    return ncurvs


def min_curvature(img, steps = None, mask = None):
    
    mask = np.full(img.shape, True) if mask is None else mask

    bins, cells, enes = _cells(img, steps, mask = mask)
    #size = len(enes)
    #ndim = img.ndim
    #xdir = np.zeros((size, ndim))
    #for i in range(ndim): xdir[:, i] = edir[i, mask]
    curv, edir  = _min_curvature(bins, mask, cells, enes)

    ndim  = img.ndim
    shape = img.shape 
    ncurv = np.zeros(shape)
    ncurv[mask] = curv
    ndir  = np.zeros( (ndim,) + shape)
    for i in range(ndim): ndir[i][mask] = edir[i]

    return ncurv, ndir


def laplacian(img, steps = None, mask = None, extended = False):
    
    mask  = np.full(img.shape, True) if mask is None else mask

    bins, cells, enes = _cells(img, steps, mask = mask)
    lap               = _laplacian(bins, mask, cells, enes, extended = extended)
    
    nlap, _ = np.histogramdd(cells, bins, weights = lap)
    
    return nlap



def transverse_curvature(img, edir, steps = None, mask = None):
    
    
    mask = np.full(img.shape, True) if mask is None else mask

    bins, cells, enes = _cells(img, steps, mask = mask)
    curv = _transverse_curvature(bins, mask, cells, enes, edir)
    
    ncurv = np.zeros(img.shape)
    ncurv[mask] = curv
        
    return ncurv


def min_transverse_curvature(img, steps = None, mask = None):
    
    mask = np.full(img.shape, True) if mask is None else mask

    bins, cells, enes = _cells(img, steps, mask = mask)

    curv, edir = _min_transverse_curvature(bins, mask, cells, enes)
    
    ndim, shape = img.ndim, img.shape
    ncurv = np.zeros(shape)
    ndir  = np.zeros((ndim,)+ shape)
    ncurv[mask] = curv
    for i in range(ndim):
        ndir[i][mask] = edir[i]

    return ncurv, ndir


def transverse_curvatures(img, edir, steps = None, mask = None): 
    
    mask = np.full(img.shape, True) if mask is None else mask

    bins, cells, enes = _cells(img, steps, mask = mask)

    curvs = _transverse_curvatures(bins, mask, cells, enes, edir)
    
    ncurvs = []
    for curv in curvs:
        ncurv = np.zeros(img.shape)
        ncurv[mask] = curv
        ncurvs.append(ncurv)

    return ncurvs

    return




# filters
#------------------------


#edge_filter          = filters.get_edge_filter(gradient)
#ridge_filter         = filters.get_ridge_filter(gradient, 
#                                                min_transverse_curvature,
#                                                transverse_curvatures)
#ridge_lambda_filter  = filters.get_ridge_lambda_filter(gradient,
#                                                       min_curvature)
#node_filter          = filters.node_filter
#blob_filter          = filters.get_blob_filter(laplacian)
#normal_laplacian     = filters.get_normal_laplacian(curvatures)
#nlap_scan            = filters.get_nlap_scan(normal_laplacian)

#----- internal


def _bins(img, steps = None, x0 = None):
    
    ndim  = img.ndim
    shape = img.shape
    steps = np.ones(ndim)  if steps is None else steps
    x0s   = np.zeros(ndim) if x0    is None else x0

    ibins = [np.linspace(0, n, n + 1) for n in shape]
    bins  = [x0 + step * ibin for x0, step, ibin in zip(x0s, steps, ibins)]
    
    return bins


def _steps(bins):
    steps  = np.array([ibin[1] - ibin[0] for ibin in bins])
    return steps


def _scells(img, bins, mask):
    
    ndim      = img.ndim
    icells    = cu.to_coors(np.argwhere(mask))
    enes      = img[mask]
    centers   = [cu.ut_centers(ibin) for ibin in bins]
    cells     = [centers[i][icells[i]] for i in range(ndim)]
    
    return cells, enes
    


def _cells(img, steps = None, x0 = None, mask = None):
    
    ndim = img.ndim
    
    #mask      = img > mask
    mask      = np.full(img.shape, True) if mask is None else mask
    icells    = cu.to_coors(np.argwhere(mask))
    enes      = img[mask]
    #nsize     = len(enes)
    #kids      = np.arange(nsize).astype(int)

    bins      = _bins(img, steps, x0)
    centers   = [cu.ut_centers(ibin) for ibin in bins]
    cells     = [centers[i][icells[i]] for i in range(ndim)]

    # ISSUE: do we need icells, then ibins?
    return bins, cells, enes


    

# def _dene(bins, mask, cells, enes):

#     ndim, size   = len(cells), len(enes)
#     steps        = _steps(bins)
#     ids          = np.arange(size)
    
#     potential, _ = np.histogramdd(cells, bins, weights = enes)
#     kid, _       = np.histogramdd(cells, bins, weights = ids)

#     nn_dir       = np.zeros(potential.shape + (ndim,))
#     nn_potential = np.zeros(potential.shape)
#     nn_kid       = np.copy(kid)
    
#     #moves = get_moves_updown(ndim)
#     for move in moves(ndim):

#        #vmove = steps * move
#        # mmove = np.sqrt(np.sum(vmove * vmove))
        
#         coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
#         potential_next, _  = np.histogramdd(coors_next, bins, weights = enes)
#         kid_next, _        = np.histogramdd(coors_next, bins, weights = ids)

#         sel_next           = potential_next > nn_potential
#         sel                = (mask) & (sel_next)
        
#         nn_potential[sel]  = potential_next[sel]
#         nn_kid[sel]        = kid_next[sel]
#         for i in range(ndim):
#             nn_dir[sel, i] = -move[i]
    
#     epot  = nn_potential[mask] - potential[mask]
#     edir  = nn_dir[mask, :]
#     epath = nn_kid[mask]
    
#     return epot, edir, epath    


def _gradient(bins, mask, cells, enes):
    
    ndim, size   = len(cells), len(enes)
    steps        = _steps(bins)
    ids          = np.arange(size)
    
    potential, _ = np.histogramdd(cells, bins, weights = enes)
    kid, _       = np.histogramdd(cells, bins, weights = ids)

    nn_dir       = np.zeros(potential.shape + (ndim,))
    nn_potential = np.copy(potential)
    nn_egrad     = np.zeros(potential.shape)
    nn_kid       = np.copy(kid)
    
    #moves = get_moves_updown(ndim)
    for move in moves(ndim):

        vmove = steps * move
        mmove = np.sqrt(np.sum(vmove * vmove))
        
        coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
        potential_next, _  = np.histogramdd(coors_next, bins, weights = enes)
        kid_next, _        = np.histogramdd(coors_next, bins, weights = ids)
        egrad_next         = (potential_next - potential)/mmove


        sel_next           = potential_next > nn_potential
        sel                = (mask) & (sel_next)
        
        nn_potential[sel]  = potential_next[sel]
        nn_egrad[sel]      = egrad_next[sel]
        nn_kid[sel]        = kid_next[sel]
        for i in range(ndim):
            nn_dir[sel, i] = -move[i]
    
    egrad = nn_egrad[mask]
    edir  = nn_dir[mask, :]
    epath = nn_kid[mask].astype(int)
        
    return egrad, edir, epath
    
    

# def _gradient_bin(bins, mask, cells, enes):
    
#     ndim  = len(cells)
#     size  = len(cells[0])
#     steps = _steps(bins)
#     print('steps ', steps)
#     ids   = np.arange(size)
    
#     potential, _ = np.histogramdd(cells, bins, weights = enes)
#     kids, _      = np.histogramdd(cells, bins, weights = ids)

#     nn_potential = np.copy(potential)
#     nn_grad      = np.zeros(potential.shape)
#     nn_dir       = np.zeros(potential.shape + (ndim,))
#     nn_kids      = np.copy(kids) .astype(int)
    
#     #moves = get_moves_updown(ndim)
#     for i, move in enumerate(moves(ndim)):

#         vmove  = np.array([steps[i] * move[i] for i in range(ndim)])
#         vmode  = np.sqrt(np.sum(vmove * vmove))
#         coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
#         potential_next, _  = np.histogramdd(coors_next, bins, weights = enes)
#         kids_next, _       = np.histogramdd(coors_next, bins, weights = ids)

#         sel_pot_next       = potential_next > nn_potential
#         sel                = (mask) & (sel_pot_next)
        
#         nn_potential[sel]  = potential_next [sel]
#         #nn_move     [sel]  = i
#         nn_grad     [sel]  = (potential_next[sel] - potential[sel])/vmode
#         nn_kids     [sel]  = kids_next      [sel]
#         for i in range(ndim):
#             nn_dir[sel, i] = -move[i]



#     egrad = nn_grad[mask]
#     epath = nn_kids[mask]
#     edir  = nn_dir[mask, :]
#     #emove = nn_move[mask]
    
#     return egrad, edir, epath
    
    
def _curvature(bins, mask, cells, enes, edir):
    
    ndim         = len(cells)
    steps        = _steps(bins)
    
    vmove        = steps * edir
    vmag2        = np.sum(vmove * vmove)

    potential, _ = np.histogramdd(cells, bins, weights = enes)
    shape        = potential.shape
    mcurve       = np.zeros(shape)

    for sign in (-1, +1):
        coors_next        = [cells[i] + sign * vmove[i] for i in range(ndim)]
        potential_next, _ = np.histogramdd(coors_next, bins, weights = enes)
            
        sel             = (mask)
        mcurve   [sel] +=  (potential_next[sel] - potential[sel])/vmag2

    curve = mcurve[mask]/2
    return curve


def _curvatures(bins, mask, cells, enes, extended = False):
    
    ndim  = len(cells)
    moves = moves_face(ndim) if extended else moves_axis(ndim)
    print(extended, len(moves))
    curvs = [_curvature(bins, mask, cells, enes, move) for move in moves]
    return curvs


def _laplacian(bins, mask, cells, enes, extended = False):
    
    ndim, size = len(cells), len(cells[0])
    lap        = np.zeros(size)
    moves      = moves_face(ndim) if extended else moves_axis(ndim)
    for move in moves:
        icur = _curvature(bins, mask, cells, enes, move)
        lap += icur
    return lap


# def _curvature_transverse(bins, mask, cells, enes, edir):    
    
#     ndim,size    = len(cells), len(cells[0])

#     potential, _ = np.histogramdd(cells, bins, weights = enes)
#     shape        = potential.shape
#     nn_curvemin  = np.zeros(shape)
#     nn_curvemax  = np.zeros(shape)
#     nn_curve     = np.zeros(shape)

#     edir  = edir * np.ones ((size, ndim)) if (edir.ndim == 1) else edir
#     #print(edir.shape)
#     direction = [np.histogramdd(cells, bins, weights = edir[:, i])[0] \
#                  for i in range(ndim)]

#     curves = [] # temporal holder to return all the curves
#     for move in moves_face(ndim):
        
#         mcur      = _curvature(bins, mask, cells, enes, move)
#         mcurve, _ = np.histogramdd(cells, bins, weights = mcur)
        
#         vdot              = np.zeros(shape)
#         for i in range(ndim):
#             vdot += direction[i] * move[i]
#         sel_transv = np.isclose(vdot, 0)
        
#         mcurvesel = (mask) & (sel_transv)
        
#         nn_curve[mcurvesel] += mcurve[mcurvesel]
#         curves.append(mcurve)
                
#         sel        = (mcurvesel) & (mcurve < nn_curvemin)
#         nn_curvemin[sel] = mcurve[sel]
        
#         sel        = (mcurvesel) & (mcurve > nn_curvemax)
#         nn_curvemax[sel] = mcurve[sel]

#     curve    = nn_curve   [mask]
#     curvemin = nn_curvemin[mask]
#     curvemax = nn_curvemax[mask]

#     return curve, curvemin, curvemax, curves


def _min_curvature(bins, mask, cells, enes):
    
    ndim         = len(cells)
    potential, _ = np.histogramdd(cells, bins, weights = enes)
    shape        = potential.shape
    nn_mincurve  = np.zeros(shape)
    nn_edir      = np.zeros((ndim,) + shape)
    
    
    for move in moves_face(ndim):
        
        mcur      = _curvature(bins, mask, cells, enes, move)
        mcurve, _ = np.histogramdd(cells, bins, weights = mcur)
        
        sel_mincv = mcurve < nn_mincurve
        sel       = (mask) & (sel_mincv)
        
        nn_mincurve[sel]  = mcurve[sel]
        
        for i in range(ndim):
            nn_edir[i, sel] = move[i]
        
    mincurve = nn_mincurve[mask]
    edir     = [nn_edir[i][mask] for i in range(ndim)]

    return mincurve, edir



def _transverse_curvature(bins, mask, cells, enes,  edir):
    
    ndim, size = len(cells), len(cells[0])
    edir      = edir * np.ones ((size, ndim)) if (edir.ndim == 1) else edir
    direction = [np.histogramdd(cells, bins, weights = edir[:, i])[0] \
                 for i in range(ndim)]
    shape     = direction[0].shape
    mincurve  = np.zeros(size)
        
    for move in moves_face(ndim):
        
        vdot      = np.zeros(shape)
        for i in range(ndim): vdot += direction[i] * move[i]
        sel_orth  = np.isclose(vdot, 0)

        mcurve    = _curvature(bins, mask, cells, enes, move)

        sel       = (mask) & (sel_orth)
        sel       = sel.flatten()
        if (np.sum(sel) > 0):
            mincurve[sel] += mcurve[sel]
        
    return mincurve


def _min_transverse_curvature(bins, mask, cells, enes):
    
    ndim         = len(cells)
    potential, _ = np.histogramdd(cells, bins, weights = enes)
    shape        = potential.shape
    nn_mincurve  = np.zeros(shape)
    nn_edir      = np.zeros((ndim,) + shape)
    
    
    for move in moves_face(ndim):
        
        mcur      = _transverse_curvature(bins, mask, cells, enes, move)
        mcurve, _ = np.histogramdd(cells, bins, weights = mcur)
        
        sel_mincv = mcurve < nn_mincurve
        sel       = (mask) & (sel_mincv)
        
        nn_mincurve[sel]  = mcurve[sel]
        
        for i in range(ndim):
            nn_edir[i, sel] = move[i]
        
    mincurve = nn_mincurve[mask]
    edir     = [nn_edir[i][mask] for i in range(ndim)]

    return mincurve, edir


def _transverse_curvatures(bins, mask, cells, enes, edir):
    
    #TOTHINK
    ndim, size = len(cells), len(cells[0])
    edir      = edir * np.ones ((size, ndim)) if (edir.ndim == 1) else edir
    direction = [np.histogramdd(cells, bins, weights = edir[:, i])[0] \
                 for i in range(ndim)]
    shape     = direction[0].shape
    
    curvs = []        
    for move in moves_face(ndim):
        curv      = np.zeros(size)    
        vdot      = np.zeros(shape)
        for i in range(ndim): vdot += direction[i] * move[i]
        sel_orth  = np.isclose(vdot, 0)

        mcurve    = _curvature(bins, mask, cells, enes, move)

        sel       = (mask) & (sel_orth)
        sel       = sel.flatten()
        if (np.sum(sel) > 0):
            curv[sel] += mcurve[sel]
        curvs.append(curv)
        
    return curvs

    

# def _transverse_curvature(bins, mask, cells, ene, edir):
    
#     ndim, size   = len(cells), len(cells[0])
#     potential, _ = np.histogramdd(cells, bins, weights = enes)
#     mincurve  = np.zeros(size)
        
#     for move in moves_face(ndim):
        
#         if not moves_orthogonal(move, edir): continue
#         mcur         = _curvature(bins, mask, cells, enes, move)
#         mincurve    += mcurve
        
#     return mincurve
    
#  Internal filters        

def _edge_filter(bins, mask, cells, egrad, edir):
    
    ndim         = len(cells)
    steps        = [ibin[1] - ibin[0] for ibin in bins]

    vgrad, _     =  np.histogramdd(cells, bins, weights = egrad)
    vdir         = [edir[:, i] for i in range(ndim)]

    sel          = np.full(vgrad.shape, True)

    for sign in (-1, 1):
        cells_next   = [cells[i] + sign * steps[i] * vdir[i] for i in range(ndim)]
        vgrad_next,_ = np.histogramdd(cells_next, bins, weights = egrad)
        isel         = vgrad_next < vgrad
        sel          = (sel) & (isel)

    edge = sel[mask]
    
    return edge



def _ridge_filter(bins, mask, cells, enes, edir):

    ndim, size = len(cells), len(cells[0])
    
    vcur, _, _, _ = _curvature_transverse(bins, mask, cells, enes, edir)
    
    ridge   = np.full(size, True, dtype = bool)
    ncurv   = np.full(size, 0)
    ncurv   = vcur

    for move in moves_face(ndim):
        #if (np.sum(move) <= 0): continue
        #print(move)
        dmove = np.zeros((size, ndim), dtype = int)
        for i in range(ndim):
            dmove[:, i] = move[i]
        #print(dmove.shape)
        acur, _, _, _ = _curvature_transverse(bins, mask, cells, enes, dmove)
        
        sel         = acur < vcur 
        ridge[sel]  = False
        ncurv[sel]  = ncurv[sel]
    
    sel        = vcur >= 0
    ridge[sel] = False
    
    return ridge, ncurv

# def normal_laplacian(bins, mask, cells, enes, sigma = 1, steps = None, extended = True):
    
#     ndim   = len(cells)
#     steps  = np.ones(ndim) if steps is None else np.array(steps)
#     sigmas = sigma / steps
#     #print(sigmas)
#     img, _  = np.histogramdd(cells, bins, weights = enes) 
#     simg   = ndimg.gaussian_filter(img, sigmas)
#     wenes  = simg[mask]
#     #lap  = sigma * ndimg.laplace(simg)
#     lap    = sigma * laplace(bins, mask, cells, wenes, extended = extended)
#     return lap


# xsigmas = np.linspace(0.2, 2, 10)

# def blob_filter(bins, mask, cells, weights, sigmas = (1,), extended = True):

#     size    = len(cells[0])
#     nsigmas = len(sigmas)
#     steps   = _steps(bins)
    
#     img, _  = np.histogramdd(cells, bins, weights = weights)     
    
#     nlaps = np.zeros((size, nsigmas))
#     msigmas = np.zeros(size)
#     mlap    = np.zeros(size)
#     hasmax  = np.full(size, False)
#     for i, sigma in enumerate(sigmas):
#         nlap   = - normal_laplacian(bins, mask, cells, weights, sigma, steps, extended)
#         sel    = nlap > mlap
#         mlap   [sel] = nlap[sel]
#         usel   = (nlap > 0) & (nlap < mlap)
#         hasmax[usel] = True
#         msigmas[sel] = 1.8 * sigma
#         nlaps[:, i]  = nlap
        
#     return msigmas, mlap, hasmax, nlaps




#
#--- Utilities
#

# def ut_scale(values, a = 0, b = 1):
   
#     xmin, xmax = np.min(values), np.max(values)
#     scale  = (values - xmin)/(xmax - xmin)
#     return scale

# def ut_centers(xs : np.array) -> np.array:
#     return 0.5* ( xs[1: ] + xs[: -1])


# def arstep(x, step, delta = False):
#     delta = step/2 if delta else 0.
#     return np.arange(np.min(x) - delta, np.max(x) + step + delta, step)


# def to_coors(vs):
#     ndim = len(vs[0])
#     xs = [np.array([vi[i] for vi in vs]) for i in range(ndim)]
#     return xs


# def ut_sort(values, ids, reverse = True):
    
#     vals_ = sorted(zip(values, ids), reverse = reverse)
#     vals  = np.array([v[0] for v in vals_])
#     kids  = np.array([v[1] for v in vals_])
#     return vals, kids


#   Moves
#   Todo: convert them into kernels?
#--------------

def movei(i, ndim):
    ei    = np.zeros(ndim, dtype = int)
    ei[i] = 1
    return ei


def moves(ndim):

    u0 = np.zeros(ndim)
    def u1(idim):
        ui1 = np.zeros(ndim)
        ui1[idim] = 1
        return ui1.astype(int)

    vs = (u0, u1(0), -1 * u1(0))
    for idim in range(1, ndim):
        us = (u0, u1(idim), -1 * u1(idim))
        vs = [(vi + ui).astype(int) for vi in vs for ui in us]
    vs.pop(0)

    return vs


def moves_face(ndim):
    
    #for i in range(ndim):
    vis = moves(ndim)
    vos = []
    for i in range(ndim):
        for vi in vis:
            if (np.sum(vi[:i] * vi[:i]) == 0) & (vi[i] == 1): vos.append(vi)
    return vos


def moves_pos(ndim):
    movs = moves(ndim)
    _ok  = lambda x: np.sum([vx >= 0 for vx in x]) == ndim
    movs = [mov for mov in movs if _ok(mov)] 
    return movs

def moves_axis(ndim):
    movs = moves(ndim)
    _ok = lambda x: (np.sum(x*x) <= 1) & (np.sum(x) == 1) 
    movs = [mov for mov in movs if _ok(mov)] 
    return movs


def moves_orthogonal(mov0, mov1):
    ndim   = mov0.shape[0]
    vdot   = np.zeros(mov0.shape[-1:])
    for i in range(ndim):
        vdot += mov0[i] * mov1[i]
    sel  = np.isclose(vdot, 0)
    return sel
    

#
#  Path Utilites
#

def get_path(kid, epath):
    path = []
    while epath[kid] != kid:
        path.append(kid)
        kid = epath[kid]
    path.append(kid)
    return path


def get_path_from_link(kid0, kid1, epath):
    path1 = get_path(kid0, epath)
    path2 = get_path(kid1, epath)
    path1.reverse()
    path = path1 + path2
    return path



def get_path_to_path(kid, epath, path):
    ipath  = []
    while not np.isin(kid, path):
        ipath.append(kid)
        kid_next = epath[kid]
        if (kid_next == kid): return []
        kid = kid_next 
    ipath.append(kid)
    return ipath
    

#
#  Topology
#


def cells_selection(cells, sel):
    return [cell[sel] for cell in cells]


def get_segment(cells, kids):
    """ Fron a list of local IDs returns a segment to plot
    inputs:
        cells: tuple(array), m-dim tuple with n-size array with the cells' cordinates positions
        kids: tuple(int), list of the ID to generate the segment
    """
    ndim = len(cells)
    segment = [np.array([float(cells[i][kid]) for kid in kids]) for i in range(ndim)]
    return segment



#  debugging 
#----------

def _edge_filter_hard(img       : np.array,
                      steps     : tuple = None,
                      perc      : float = 80,
                      mask      : np.array = None):

    shape = img.shape
    mask  = np.full(shape, True) if mask is None else mask
    
    bins, cells, enes  = _cells(img, steps, mask = mask)
    egrad, edir, epath = _gradient(bins, mask, cells, enes)
    
    size = len(enes)
    kid  = np.arange(size).astype(int)
    xfil = egrad[epath] < egrad[kid]
    for k in np.argwhere(kid == True):
        ok = np.max(egrad[epath == k]) < egrad[k]
        xfil[k] = ok
    
    nfil        = np.full(shape, False, dtype = bool)
    ngrad       = np.zeros(shape)
    nfil [mask] = xfil
    ngrad[mask] = egrad
    
    #nfil, _ = np.histogramdd(cells, bins, weights = efil)
    return nfil, ngrad

#---- Analysis

# def analysis(df, name = 'e'):
    
#     true = df.istrue.values
#     ext  = df.isext .values
#     cells_types = (name + 'isnode', name +'isborder',
#                    name + 'ispass', name +'isridge', 'iscore')
    
#     dat = {}
#     for itype in cells_types:
#         vals  = df[itype].values  
#         ntot  = np.sum(vals)
#         yes   = vals & true
#         noes  = vals & (~true)
#         nyes  = np.sum(yes)
#         nnoes = np.sum(noes)
#         isext = np.sum(vals & ext)
#         eff   = float(nyes/ntot) if ntot >0 else -1
#         dat[name+itype+'_success']  = nyes
#         dat[name+itype+'_extreme']  = isext
#         dat[name+itype+'_failures'] = nnoes
#         dat[name+itype+'_eff']      = eff
#     return dat       
        
    

