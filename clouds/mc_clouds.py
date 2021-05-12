#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:02:21 2021

@author: hernando
"""

import numpy  as np
#import pandas as pd

import clouds.clouds as clouds
#import clouds.graphs as graphs

#
#  MC-clouds

def mctrue(bins, mask, cells, df,  
             mccoors, mcene, 
             mctime = None, mcid = None):
    
    mctime = mcene if mctime is None else mctime
    mcid   = mcid  if mcid   is None else mcid
    ones   = np.ones(len(mcene))

    ene    = clouds.cells_value(bins, mask, mccoors, mcene)
    counts = clouds.cells_value(bins, mask, mccoors, ones)
    time   = clouds.cells_value(bins, mask, mccoors, mctime)
    pid    = clouds.cells_value(bins, mask, mccoors, mcid)
    time   = time/counts
    pid    = pid/counts
    
    df['mcene']  = ene
    sel  = ene > 0.
    #time[~sel] = -1
    df['mctime'] = time
    df['mc']     = sel
    df['mcid']   = pid
    
    #print('number of true cells ', np.sum(sel), np.sum(time >= 0), np.sum(ene >0))    
    
    df['mcextreme'] = _mcextreme(bins, mask, cells, mccoors, mctime, mcid)
    
    return df


def _mcextreme(bins, mask, cells, mccells, mcene, mctime, mcid):
        
    
    size    = len(cells[0])
    extreme = np.full(size, -1, dtype = int)
    
    usize = len(mcid)
    upids  = np.arange(usize)
    def _set(pid, extr_type):
        sel      = mcid == pid
        kid      = upids[sel]
        fsel     = np.argmax if extr_type > 0 else np.argmin
        i        = int(kid[fsel(mctime[sel])])
        utime    = np.zeros(usize)
        utime[i] = mctime[i]
        xcell = clouds.cells_value(bins, mask, mccells, utime)
        #upos = [ucell[i] for ucell in mccells]
        #print('pid ', pid, 'index ', i, 'time ', utime[i], 'pos ', upos)
        extreme[xcell > 0] = extr_type
    
    for pid in   np.unique(mcid): _set(pid, 2)
    for pid in (1, 2): _set(pid, 1)
    for pid in (1, 2): _set(pid, 0)
    
    return extreme


# def _mcextreme(bins, mask, cells, ene):

#     size   = len(ene)
#     extreme  = np.full(size, -1, dtype = int)

#     node, epath, isnode  = graphs._emap(bins, mask, cells, ene)
        
#     # select the nodes and order then by energy
#     # array by node-index (nsize)
#     enode, nnode  = graphs._nodes(ene, node, isnode)
    
#     sel = enode > 0
#     nnode = nnode[sel]
        
#     _, nlink = graphs._links(bins, mask, cells, ene, node, nnode)
    
#     degrees = graphs._degrees(nlink)
#     extreme[nnode] = degrees
    
#     return extreme
    

# def mcextreme(bins, mask, cells, df):
    
#     for name in ('mcene', 'mctime'):      
#         vals = df[name].values
#         df[name + 'extr'] = _mcextreme(bins, mask, cells, vals)
    
#     time   = df.mctime.values
#     istrue = df.mc    .values
#     df['mcextr'] = _mcextreme_time(bins, mask, cells, time, istrue)
    
#     return df


# def _mcextreme_time(bins, mask, cells, time, istrue):
    
#     # find the time extremes
#     # compute the links,
#     # extremes are: single - no 
        
#     size   = len(time)
#     kid   = np.arange(size)
#     c0 = kid[istrue][np.argmin(time[istrue])] # minimum time 
#     c1 = kid[istrue][np.argmax(time[istrue])] # maximum time
#     print('time extremes ', c0, c1, time[c0], time[c1])
    
#      # computhe the nodes an the path to the node
#     # arrays by cell-index (size)
#     node, epath, isnode  = graphs._emap(bins, mask, cells, time)
        
#     # select the nodes and order then by energy
#     # array by node-index (nsize)
#     tnode, nnode  = graphs._nodes(time, node, isnode)
    
#     sel = tnode > 0
#     nnode = nnode[sel]
#     print('nodes ', nnode)
#     print('times ', tnode)
#     if (c0 not in nnode):
#         nnode = list(nnode)
#         nnode.append(c0)
#         nnode = np.array(nnode, dtype = int)
#         node[c0]  = c0
#         #enode.append(time[c0])
        
#     if (c1 not in nnode):
#         print('warning max time cell not in time nodes!')
    
#     print('mcextr time nodes', nnode)
#     _, nlink = graphs._links(bins, mask, cells, time, node, nnode)
    
#     degrees = graphs._degrees(nlink)
#     print('mcextr time degrees', degrees)

    
#     extreme  = np.full(size, -1, dtype = int)
#     extreme[nnode] = degrees
    
#     return extreme
    
    
#---- Analysis

def analysis(df, name = 'e'):
    
    true = df.istrue.values
    ext  = df.isext .values
    cells_types = (name + 'isnode', name +'isborder',
                   name + 'ispass', name +'isridge', 'iscore')
    
    dat = {}
    for itype in cells_types:
        vals  = df[itype].values  
        ntot  = np.sum(vals)
        yes   = vals & true
        noes  = vals & (~true)
        nyes  = np.sum(yes)
        nnoes = np.sum(noes)
        isext = np.sum(vals & ext)
        eff   = float(nyes/ntot) if ntot >0 else -1
        dat[name+itype+'_success']  = nyes
        dat[name+itype+'_extreme']  = isext
        dat[name+itype+'_failures'] = nnoes
        dat[name+itype+'_eff']      = eff
    return dat       