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


# filters
#------------------------

edge_filter          = filters.get_edge_filter(gradient)
ridge_filter         = filters.get_ridge_filter(gradient, 
                                                min_transverse_curvature)
ridge_lambda_filter  = filters.get_ridge_lambda_filter(gradient,
                                                       min_curvature)
node_filter          = filters.node_filter
blob_filter          = filters.get_blob_filter(laplacian)
normal_laplacian     = filters.get_normal_laplacian(curvatures)
nlap_scan            = filters.get_nlap_scan(normal_laplacian)


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
    #print(extended, len(moves))
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
#  Path Utilites - TODO export to clouds
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




