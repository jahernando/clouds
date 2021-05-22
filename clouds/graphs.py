#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:40:54 2021

@author: hernando
"""

import numpy             as np
#import pandas            as pd

#import scipy.sparse.csgraph as scgraph

import collections
#import functools
#import operator

#import clouds.ridges     as ridges
#import tables            as tb

import clouds.clouds as clouds

# 
#   Graphs
#
 
Graph = collections.namedtuple('Graph', 
                               ('enode', 'nnode', 'elink', 'nlink', 
                                'node', 'epath'))


def _graph(bins, mask, cells, ene):
        
    # computhe the nodes an the path to the node
    # arrays by cell-index (size)
    node, epath, isnode  = _emap(bins, mask, cells, ene)
    
    # select the nodes and order then by energy
    # array by node-index (nsize)
    enode, nnode  = _nodes(ene, node, isnode)
    
    # generate the links between nodes
    # array (nsize, size)
    elink, nlink  = _links(bins, mask, cells, ene, node, nnode)
    
    return Graph(enode, nnode, elink, nlink, node, epath)
    

def _emap(bins, mask, cells, ene, max_increase = True):

    #egrad, epath  = clouds.gradient(bins, mask, cells, ene)
    egrad, epath  = _path(bins, mask, cells, ene, max_increase = max_increase)
    isnode        = (egrad == 0) 
    node          = clouds.set_node(epath)
    
    return node, epath, isnode


def _nodes(ene, node, isnode):
    
    nnode   = np.unique(node[isnode])
    enode   = [np.sum(ene[node == inode]) for inode in nnode]

    enode, nnode = clouds.ut_sort(enode, nnode)
   
    return enode, nnode
   

def _links(bins, mask, cells, ene, node, nodes):
    # compute the link matrix - return matrix with IDs and strength

    # number of nodes
    ndim    = len(cells)
    nsize   = len(nodes)
    steps  = [ibin[1] - ibin[0] for ibin in bins]
    
    elink   = np.full((nsize, nsize),  0, dtype = float)
    nlink   = np.full((nsize, nsize), -1, dtype = int)
    
    # number of cells
    msize          = len(ene)
    kid            = np.arange(msize)

    # add 1 to node, as nodes starts at 0,
    n_potential, _ = np.histogramdd(cells, bins, weights = ene)
    n_node, _      = np.histogramdd(cells, bins, weights = 1 + node)
    n_kid, _       = np.histogramdd(cells, bins, weights = kid)
    shape          = n_potential.shape  
   
    def next_(move):
        
        nn_potential = np.copy(n_potential)
        nn_node      = np.full(shape, -1, int)
        nn_kid       = np.copy(n_kid)

        cells_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
        potential_next, _  = np.histogramdd(cells_next, bins, weights = ene)
        node_next, _       = np.histogramdd(cells_next, bins, weights = 1 + node)
        kid_next, _        = np.histogramdd(cells_next, bins, weights = kid)

        # select cells where the node has changes and its a valid node
        # select cells where the sum of the potential has changed (all!)
        sel_node           = (node_next != n_node) & (node_next > 0)
        #sel_pot_next       = n_potential + potential_next > nn_potential
        sel                = (mask) & (sel_node)
        
        nn_potential[sel]  = n_potential[sel] + potential_next[sel]
        nn_kid      [sel]  = kid_next [sel]
        nn_node     [sel]  = node_next[sel]
        
        cpot  = nn_potential[mask]
        ckid  = nn_kid      [mask].astype(int)
        cnod  = nn_node     [mask].astype(int) - 1
        
        #usel = (cnod != node) & (cnod >= 0)
        #print(' move ', move)
        #print(' number of changes ', np.sum(usel))
        #print(' nodes ', list(zip(node[usel], cnod[usel])))
        
        return cpot, ckid, cnod


    def next_link_(cpot, ckid, cnode):
        
        for i, inode in enumerate(nodes):
            for ii, jnode in enumerate(nodes[i+1 : ]):
                j     = i + 1 + ii
                sel   = (node == inode) & (cnode == jnode)
                if (np.sum(sel) <= 0): continue
                #print(' next link ', i, j, ', knodes, ', inode, jnode)
                xk0s  =  kid[sel]
                xpot  = cpot[sel]
                xpot, xk0s = clouds.ut_sort(xpot, xk0s)
                if (xpot[0] <= elink[i, j]): continue
        
                k0   = int(xk0s[0])
                k1   = int(ckid[k0])

                nlink[i, j] = k0
                nlink[j, i] = k1
                elink[i, j] = xpot[0]
                elink[j, i] = xpot[0]
                
                #print(' update link id  ', k0, k1)
                #print(' update link id  ', nlink[i, j], nlink[j, i])
                #print(' update link ene ', elink[i, j], elink[j, i])

    #moves = get_moves_updown(ndim)
    for move in clouds.moves(ndim):
        
        cpot, ckid, cnode = next_(move)
        next_link_(cpot, ckid, cnode)
    
    return elink, nlink


def _graph_links(nlinks):
    
    size  = len(nlinks[0])
    links = []
    for i in range(size):
        for j in range(size):
            if (j <= i): continue
            if (nlinks[i, j] >= 0):
                i0 = nlinks[i, j]
                j0 = nlinks[j, i]
                links.append((i0, j0))
    return links


def _degrees(nlink):
    return np.sum(nlink > 0, axis = 0)



def _path(bins, mask, cells, weights, 
          absolute = True, max_increase = True):
    
    ndim, size   = len(cells), len(weights)
    steps        = [ibin[1] - ibin[0] for ibin in bins]
    
    enes             = np.copy(weights)
    ids              = np.arange(size)
    
    potential, _ = np.histogramdd(cells, bins, weights = enes)
    kids, _      = np.histogramdd(cells, bins, weights = ids)

    factor       = 1      if absolute     else 0
    factor       = factor if max_increase else 100
    nn_potential = factor * np.copy(potential)
    nn_kids      =          np.copy(kids) .astype(int)
    
    fmax = lambda enext, ecurr, eref: (enext > ecurr) & (enext > eref)
    fmin = lambda enext, ecurr, eref: (enext < ecurr) & (enext > eref)
    _condition = fmax if max_increase else fmin
        
    #moves = get_moves_updown(ndim)
    for move in clouds.moves(ndim):

        coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
        potential_next, _  = np.histogramdd(coors_next, bins, weights = enes)
        kids_next, _       = np.histogramdd(coors_next, bins, weights = ids)

        sel_pot_next       = _condition(potential_next, nn_potential, potential)
        sel                = (mask) & (sel_pot_next)
        
        nn_potential[sel]  = potential_next[sel]
        nn_kids     [sel]  = kids_next     [sel]


    egrad = nn_potential[mask] - potential[mask]
    epath = nn_kids     [mask]
    
    return egrad, epath


# def _select_link(inode, jnode, node, lgrad, lpath):

#     size       = len(node)
#     kid        = np.arange(size)
#     sel_dir    = (kid  == lpath[ lpath ])
#     sel_border = (node == inode) & (node[lpath] == jnode)
#     if (np.sum(sel_border) <= 0): return (-1, -1)
    
#     sel = (sel_border) # & (sel_dir) 
#     sel = sel if np.sum(sel)>0 else (sel_border) 
#     if (np.sum(sel) <= 0): return (-1, -1)
    
#     ipos   = np.argmax(lgrad[sel])
#     k0     = kid[sel][ipos]            
#     k1     = lpath[k0]
#     return (k0, k1)
    

# def _graph_from_cloud(enes, node, ispass, lpath):
    
#     nnodes = np.unique(node)
#     enodes = [np.sum(enes[node == inode]) for inode in nnodes]

#     enodes, nnodes = clouds.ut_sort(enodes, nnodes)
    
#     nsize  = len(nnodes)
#     nlink  = np.full((nsize, nsize), -1, dtype = int) 
#     elink  = np.full((nsize, nsize), 0, dtype = float)

#     # the link
#     #size   = len(enes)
#     #condition = np.full(size, True, dtype = bool) if condition is None else condition
    
#     #print('nodes ', nids)
    
#     for i, n0 in enumerate(nnodes):
#         for j, n1 in enumerate(nnodes):
#             if (j <= i): continue
#             sel   = (ispass) & (node == n0)  & (node[lpath] == n1)
#             #print('nodes ', n0, n1, ', sel ', np.sum(sel))
#             if (np.sum(sel) != 1): continue
#             k0 = int(np.argwhere(sel))
#             k1 = lpath[k0]
#             nlink[i, j] = k0 
#             nlink[j, i] = k1
#             elink[i, j] = enes[k0]
#             elink[j, i] = enes[k1]
                
#     return Graph(enodes, nnodes, elink, nlink)
    
        

# def _graph_simple(bins, mask, cells, enes, node, condition = None):
    
#     nnodes = np.unique(node)
#     enodes = [np.sum(enes[node == inode]) for inode in nnodes]

#     enodes, nnodes = clouds.ut_sort(enodes, nnodes)
    
#     nsize  = len(nnodes)
#     nlink  = np.full((nsize, nsize), -1, dtype = int) 
#     elink  = np.full((nsize, nsize), 0, dtype = float)

#     # the links
#     isborder, idborder = \
#         clouds.find_borders(bins, mask, cells, node) 
  
#     lgrad, lpath       = \
#         clouds.gradient_between_nodes(bins, mask, cells, enes, node)

#     #size   = len(enes)
#     #condition = np.full(size, True, dtype = bool) if condition is None else condition
    
#     #print('nodes ', nids)
    
#     for i, n0 in enumerate(nnodes):
#         for j, n1 in enumerate(nnodes):
#             if (j <= i): continue
#             k0, k1 = _select_link(n0, n1, node, lgrad, lpath)
#             if (k0 < 0) or (k1 < 0): continue
#             nlink[i, j] = k0 
#             nlink[j, i] = k1
#             elink[i, j] = enes[k0]
#             elink[j, i] = enes[k1]
                
#     return Graph(enodes, nnodes, elink, nlink)
        

# def _graph_new(bins, mask, cells, cenes, cnode):
    
#     # The nodes
#     nodes  = np.unique(cnode)
#     enodes = [np.sum(cenes[cnode == inode]) for inode in nodes]

#     enodes, nodes = clouds.ut_sort(enodes, nodes)
#     #print('nodes  ', nodes)
#     #print('enodes ', enodes)

    
#     # the links
#     isborder, idborder = \
#         clouds.find_borders(bins, mask, cells, cnode) 
  
#     lgrad, lpath       = \
#         clouds.gradient_between_nodes(bins, mask, cells, cenes, cnode)

#     def _has(ar):
#         return (np.sum(ar) > 0)
    
#     is_multiborder   = idborder == -3 # id of cells as multiborder
    

#     #print(' num border cells '      , np.sum(isborder))
#     #print(' num multi-border cells ', np.sum(is_multiborder), np.sum(idborder != -1))
#     #print(' nodes with multiborder ', np.unique(cnode[is_multiborder]))

  
#     # prepare output    
#     size   = len(nodes)
#     nlink  = np.full((size, size), -1, dtype = int) 
#     elink  = np.full((size, size), 0, dtype = float)
    
            
#     for i, inode in enumerate(nodes):

#         iborder  = (cnode == inode) & isborder
#         #print(' border cells of node ', i, inode, ' : ', np.sum(iborder))
#         if (not _has(iborder)): continue
    
#         for j, jnode in enumerate(nodes):
#             if (j <= i): continue # ordered nodes
   
#             jborder = (cnode == jnode) & isborder
#             #print(' border cells of node ', j, jnode, ': ', np.sum(jborder))

#             if (not _has(jborder)): continue

#             ij_lgrad, ij_lpath = np.copy(lgrad), np.copy(lpath)
#             ij_border = np.logical_or((cnode == inode) & (idborder == jnode),
#                                       (cnode == jnode) & (idborder == inode))
            
#             #print(' num of single border cells ', np.sum(ij_border))

#             mborder = np.logical_or(iborder, jborder) & is_multiborder
#             #print(' are multi-border? ', np.sum(mborder))
            
#             if _has(mborder):
                
#                 sel = np.logical_or(iborder, jborder)
#                 ij_border, _ =  clouds.find_borders(bins, mask, cells, cnode, condition = sel)
#                 #print(' num of multi border cells ', np.sum(ij_border))
#                 #print('common border?', inode, jnode, np.sum(isborder_ij))
#                 if (not _has(ij_border)): continue
            
#                 ij_lgrad, ij_lpath = \
#                     clouds.gradient_between_nodes(bins, mask, cells,
#                                                   cenes, cnode, ij_border)
            
#             if (not _has(ij_border)): continue
                    
#             ij_lgrad[~ij_border] = 0 
            
            
#             k0, k1 = _select_link(inode, jnode, cnode, ij_lgrad, ij_lpath)
#             if ((k0 < 0) or (k1 < 0)): continue
            
#                 #id0 = np.argmax(ij_lgrad)
#                 #id1 = ij_lpath[id0]            
#                 #link = (id0, id1) if cnode[id0] == inode else (id1, id0)
#                 #id0, id1 = link[0], link[1] 
#             nlink[i, j] = k0 
#             nlink[j, i] = k1
#             elink[i, j] = cenes[k0]
#             elink[j, i] = cenes[k1]
#             #print(' link nodes ', inode, jnode,', n cells ', np.sum(ij_border))
#             #print(' link ids   ', k0, k1, ', enes ', cenes[k0], cenes[k1])

        
#     #print('graph stregnth ', link_strength)
#     #print('graph cells    ', link_cells)
#     return Graph(enodes, nodes, elink, nlink)
   
    

# def _graph(bins, mask, cells, cenes, cnode):
        
#     # set the axis 
#     nodes  = np.unique(cnode)
#     enodes = [np.sum(cenes[cnode == inode]) for inode in nodes]

#     enodes, nodes = clouds.ut_sort(enodes, nodes)
#     #print('nodes  ', nodes)
#     #print('enodes ', enodes)
        
#      # the links
#     cisborder, _ = \
#         clouds.find_borders(bins, mask, cells, cnode)
    
#     # make the link matrix
#     size   = len(nodes)
#     nlink  = np.full((size, size), -1, dtype = int) 
#     elink  = np.full((size, size), 0, dtype = float)
#     for i, inode in enumerate(nodes):
#         is_inode = (cnode == inode)
#         for j, jnode in enumerate(nodes):
#             if (j <= i): continue
#             is_jnode = (cnode == jnode)
#             sel         = (np.logical_or(is_inode, is_jnode)) & (cisborder)
#             #print('possible conection ?', inode, jnode, np.sum(sel))
#             if (np.sum(sel) == 0): continue
#             isborder_ij, _ =  clouds.find_borders(bins, mask, cells, cnode, condition = sel)
#             #print('common border?', inode, jnode, np.sum(isborder_ij))
#             if (np.sum(isborder_ij) == 0): continue
#             lgrad, lpath = clouds.gradient_between_nodes(bins, mask, cells,
#                                                          cenes, cnode, isborder_ij)
#             k0, k1 = _select_link(inode, jnode, cnode, lgrad, lpath)
#             if ((k0 < 0) or (k1 < 0)): continue
            
#                 #id0 = np.argmax(ij_lgrad)
#                 #id1 = ij_lpath[id0]            
#                 #link = (id0, id1) if cnode[id0] == inode else (id1, id0)
#                 #id0, id1 = link[0], link[1] 
#             nlink[i, j] = k0 
#             nlink[j, i] = k1
#             elink[i, j] = cenes[k0]
#             elink[j, i] = cenes[k1]
#             #print(' link nodes ', inode, jnode,', n cells ', np.sum(ij_border))
#             #print(' link ids   ', k0, k1, ', enes ', cenes[k0], cenes[k1])
#             #if (k0 >= 0)
#             #id0 = np.argmax(lgrad)
#             #id1 = lpath[id0]            
#             #link = (id0, id1) if cnode[id0] == inode else (id1, id0)
#             #print('link ', inode, jnode, lgrad[id0])
#             #id0, id1 = link[0], link[1] 
#             #nlink[i, j] = id0 
#             #nlink[j, i] = id1
#             #elink[i, j] = cenes[id0]
#             #elink[j, i] = cenes[id1]
        
#     #print('graph stregnth ', link_strength)
#     #print('graph cells    ', link_cells)
#     return Graph(enodes, nodes, elink, nlink)
        
    
# def _graph_mat(elinks):
#     sel = elinks > 0
#     ulinks      = np.zeros(elinks.shape)
#     ulinks[sel] = 1 + elinks[sel] - np.max(elinks)
#     ulinks += ulinks.T
#     #print(ulinks)
#     return ulinks


# def _graph_links_mstree(elinks, nlinks):

#     # symmetrice the link matrix    
#     ulinks = _graph_mat(elinks)
#     #print(ulinks)
    
#     data = scgraph.minimum_spanning_tree(ulinks)
#     mat  = data.toarray()
#     #print(mat)
    

#     pairs = [list(kid) for kid in np.argwhere(mat > 0)]    
#     #print(pairs)
#     links = [(nlinks[i, j], nlinks[j, i]) for i, j in pairs]
    
#     return links


# def _graph_links_shortestdist(elinks, nlinks):
        
#     ulinks = _graph_mat(elinks) 
#     #print(ulinks)
    
#     dist, pred = scgraph.shortest_path(ulinks, return_predecessors = True)
#     dist[np.isinf(dist)] = 0.
#     #print(dist)
#     #print(pred)
    
#     kid_best = np.argmax(dist)
#     ijbest   = np.unravel_index(kid_best, dist.shape)
#     #print('best ', ijbest)
#     #print('dist ', dist[ijbest])

#     def _get_path(pred, i, j):
#         path = [j]
#         k = j
#         while pred[i, k] != -9999:
#             path.append(pred[i, k])
#             k = pred[i, k]
#         return path[::-1]

#     npath = _get_path(pred, *ijbest)
#     #print(npath)

#     pairs = [(npath[i], npath[i+1]) for i in range(len(npath)-1)]
#     links = [(nlinks[i, j], nlinks[j, i]) for i, j in pairs]
        
#     return links
    
    
# def _graph_links(nlinks):
    
#     size  = len(nlinks[0])
#     links = []
#     for i in range(size):
#         for j in range(size):
#             if (j <= i): continue
#             if (nlinks[i, j] >= 0):
#                 i0 = nlinks[i, j]
#                 j0 = nlinks[j, i]
#                 links.append((i0, j0))
#     return links
    

# def _graph_paths(nodes, link_cells, epath):
    
#     paths = []
#     for i, inode in enumerate(nodes):
#         for j, jnode in enumerate(nodes):
#             #if (j <= i): continue        
#             kid0 = link_cells[i, j]
#             kid1 = link_cells[j, i]
#             if ((kid0 <= -1) & (kid1 <= -1)): continue
#             path = clouds.get_path_from_link(kid0, kid1, epath)
#             paths.append(path)
#     return paths

