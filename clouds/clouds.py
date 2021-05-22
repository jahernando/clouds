import numpy             as np
import pandas            as pd

#import scipy.sparse.csgraph as scgraph

import collections
import functools
import operator

import clouds.ridges     as ridges
#import tables            as tb

from collections import namedtuple


#
#  Main steps
#

def clouds(coors, bins, weights, 
           threshold = 0.):
    """

    Parameters
    ----------
    coors     : tuple(np.array) n-dim tuple of np.array with the coordinates
    bins      : tuple(int) or tuple(np.array), either a m-dim tuple with the bins sizes or a list of arrays with the bins
    weights   : np.array of weights of each coordinate
    threshold : TYPE, optional

    Returns
    -------
    bins : TYPE
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION.
    cells : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.

    """
    
    _check(coors, bins, weights)
    
    bins, mask, icells, cells, enes = \
        get_frame(coors, bins, weights, threshold)
    
    def _asdict(ndtup, name = 'e'):
        data = {}
        adic = ndtup._asdict()
        for key in adic.keys():
            data[name + key] = adic[key]
        return data
        
    data  = get_features(bins, mask, cells, enes)
    
    for i, cell in enumerate(cells):
        data['x'+str(i)] = cell    
    nbours = neighbours(bins, mask, cells, enes)
    data['energy']     = enes
    data['neighbours'] = nbours
    data['kid']        = np.arange(len(enes))
    
    size           = len(enes)
    truecondition  = np.full(size, True, bool)
    data['iscore'] = truecondition

    ecloud = get_cloud(bins, mask, cells, enes, truecondition)
    data.update(_asdict(ecloud, 'e'))
    
    lap   = data['lap']
    vv    = -lap + np.max(lap)
    pcloud = get_cloud(bins, mask, cells, vv, truecondition)
    data.update(_asdict(pcloud, 'p'))
    
    df    = pd.DataFrame(data)
   # frame = Frame(bins, mask, cells)
        
    return bins, mask, cells, df
    
Frame = namedtuple('Frame', ('bins', 'mask', 'cells'))

Cloud = namedtuple('Cloud', ('inten', 'grad', 'path', 'gradrel', 'pathrel',
                             'lgrad', 'link', 'node', 'isnode', 'isborder',
                             'idborder', 'ispass', 'isridge'))
    

def get_features(bins, mask, cells, weights):
    
    x, _       = np.histogramdd(cells, bins, weights = weights)
    steps      = [bin[1] - bin[0] for bin in bins]

    ndim       = x.ndim
    grad       = ridges.gradient(x, steps)
    vgrad      = np.sqrt(np.sum(grad * grad, axis = ndim))
    hess       = ridges.hessian(x, steps)
    leig, eeig = np.linalg.eigh(hess)
    lap        = ridges.laplacian(hess)    

    gradsph    = ridges.vector_in_spherical(grad)
    e0sph      = ridges.vector_in_spherical(eeig[..., -1])
    fus        = [np.sum(eeig[..., i] * grad, axis = ndim) for i in range(ndim)]
    
    data          = {}
    data['vgrad'] = vgrad[mask] # gradsph[0][mask] (check!)
    data['lap']   = lap[mask]
    data['l1']    = leig[..., 0][mask]
    data['vphi']  = gradsph[1][mask]
    data['l0']    = leig[..., -1][mask]
    data['e0phi'] = e0sph[1][mask]
    data['ge0']   = fus[-1][mask]
    data['ge1']   = fus[0][mask]

    
    if (ndim == 3):
        data['l2']      = leig[..., 1][mask]
        data['ge2']     = fus[1][mask]
        data['vtheta']  = gradsph[2] [mask]
        data['e0theta'] = e0sph  [2] [mask]

    return data


def get_cloud(bins, mask, cells, weights, condition = None):
    
    egrad, epath = gradient_to_neighbour(bins, mask, cells, weights,
                                         absolute = True)
    egradrel, epathrel = gradient_to_neighbour(bins, mask, cells, weights,
                                         absolute = False)

    isnode       = find_nodes(egrad)
    enode        = set_node(epath)
    
    isborder, idborder = find_borders(bins, mask, cells, enode) 
    lgrad, lpath       = gradient_between_nodes(bins, mask, cells, weights, enode)
    ispass             = find_passes(enode, lpath, lgrad, condition)
    
    isridge      = find_new_ridge(ispass, epath, lpath)
    
    cloud = Cloud(weights, egrad, epath, egradrel, epathrel,
                  lgrad, lpath, enode, isnode,
                  isborder, idborder, ispass, isridge)
    
    return cloud
    

def _check(coors, bins, weights):

    ndim         = len(coors)
    size         = len(coors[0])
    
    assert np.sum([len(coor) for coor in coors]) == ndim * size, \
        'not same dimension of coors'
    
    assert len(bins) == ndim, \
        'steps dimension must be equal to dimension of coors'
    
    return True
    

def get_frame(coors, bins, weights, threshold = 0.):
    
    ndim      = len(coors)
        
    # create the bins if bins are the step-size
    isbins    = isinstance(bins[0], np.ndarray) or isinstance(bins[0], collections.Sequence)
    #print('is bins?', isbins)
    bins      = bins if isbins else \
        [arstep(x, step, True) for x, step in zip(coors, bins)] 

    counts, _ = np.histogramdd(coors, bins = bins, weights = weights)

    mask      = counts > threshold
    icells    = to_coors(np.argwhere(mask))
    enes      = counts[mask]
    #nsize     = len(enes)
    #kids      = np.arange(nsize).astype(int)

    centers   = [ut_centers(ibin) for ibin in bins]
    cells     = [centers[i][icells[i]] for i in range(ndim)]

    # ISSUE: do we need icells, then ibins?
    return bins, mask, icells, cells, enes


def _steps(bins):
    steps    = [ibin[1] - ibin[0] for ibin in bins]
    return steps
     

def cells_value(bins, mask, cells, weights):
    
    potential, _ = np.histogramdd(cells, bins = bins, weights = weights)
    return potential[mask]
   
    
def neighbours(bins, mask, cells, weights):

     steps        = [ibin[1] - ibin[0] for ibin in bins]
     ndim, _      = len(cells), len(cells[0])

     counts, _       = np.histogramdd(cells, bins, weights = weights)

     counts[mask]   = 0

     for move in moves(ndim):
         coors_next      = [cells[i] + steps[i] * move[i] for i in range(ndim)]
         counts_next, _  = np.histogramdd(coors_next, bins, weights = weights)
         sel             = counts_next > 0
         counts[sel & mask] += 1

     nbours = counts[mask].astype(int)
     return nbours


def gradient_to_neighbour(bins, mask, cells, weights,
                          condition = None, absolute = True):
    
    ndim, size   = len(cells), len(weights)
    steps        = [ibin[1] - ibin[0] for ibin in bins]
    
    condition    = np.full(size, True, dtype = bool) if condition == None else condition

    enes             = np.copy(weights)
    enes[~condition] = 0.
    ids              = np.arange(size)
    
    potential, _ = np.histogramdd(cells, bins, weights = enes)
    cond, _      = np.histogramdd(cells, bins, weights = condition)
    kids, _      = np.histogramdd(cells, bins, weights = ids)

    factor       = 1 if absolute else 0
    nn_potential = factor * np.copy(potential)
    nn_grad      =     0  * np.copy(potential)
    nn_kids      =          np.copy(kids) .astype(int)
    sel_cond     = cond == True    
    
    #moves = get_moves_updown(ndim)
    for move in moves(ndim):

        vmove  = np.array([steps[i] * move[i] for i in range(ndim)])
        vmode  = np.sqrt(np.sum(vmove * vmove))
        coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
        potential_next, _  = np.histogramdd(coors_next, bins, weights = enes)
        kids_next, _       = np.histogramdd(coors_next, bins, weights = ids)

        sel_pot_next       = potential_next > nn_potential
        sel                = (mask) & (sel_cond) & (sel_pot_next)
        
        nn_potential[sel]  = potential_next[sel]
        nn_grad     [sel]  = (potential_next[sel] - potential[sel])/vmode
        nn_kids     [sel]  = kids_next     [sel]


    lgrad = nn_grad[mask]#nn_potential[mask] - potential[mask]
    lpath = nn_kids[mask]
    
    return lgrad, lpath


def find_nodes(egrad, condition = None):
    
    size = len(egrad)
    condition = np.full(size, True, dtype = bool) if condition is None else condition
   
    #print('find_nodes egrad ', np.sum(egrad == 0))
    isnode = (egrad == 0) & condition
    
    return isnode


def set_node(epath):

    # associate each cell to a node
    kids  = np.arange(len(epath))
    node  = [get_path(kid, epath)[-1] for kid in kids]
    node  = np.array(node).astype(int)

    return node


# def find_borders(bins, mask, cells, node, condition = None):

#     ndim   = len(cells)
#     steps  = [ibin[1] - ibin[0] for ibin in bins]

#     size      = len(node)           
#     condition = np.full(size, True, dtype = bool) \
#         if condition is None else condition
                
#     node_             = np.copy(node)
#     node_[~condition] = -1

#     nodes, _     = np.histogramdd(cells, bins, weights = node_)
#     cond, _      = np.histogramdd(cells, bins, weights = condition)
#     sel_cond     = np.logical_and(mask, cond > 0)
#     nn_border    = np.full(nodes.shape, False, dtype = bool)
#     nn_iborder   = np.copy(nodes)

#     for move in moves(ndim):
#         coors_next     = [cells[i] + steps[i] * move[i] for i in range(ndim)]
#         nodes_next, _  = np.histogramdd(coors_next, bins, weights = node_)

#         sel_nodes      = np.logical_and((nodes_next != nodes), (nodes_next != -1))

#         sel            = np.logical_and(sel_nodes, sel_cond)

#         nn_border[sel] = np.logical_or(nn_border[sel], nodes_next[sel])
#         #nn_border[sel] = True
        
#         #usel_new       = (nn_iborder == nodes) & (sel)
#         #usel_mul       = (nn_iborder != nodes) & (sel) & (nn_iborder != nodes_next)
#         #nn_iborder[usel_new] = nodes_next[usel_new]
#         #nn_iborder[usel_mul] = -2


#     isborder = nn_border [mask].astype(bool)
#     idborder = nn_iborder[mask].astype(int)
#     return isborder, idborder



def find_borders(bins, mask, cells, node, condition = None):

    ndim   = len(cells)
    steps  = [ibin[1] - ibin[0] for ibin in bins]

    size      = len(node)           
    condition = np.full(size, True, dtype = bool) \
        if condition is None else condition
                
    node_             = 1 + np.copy(node)
    node_[~condition] = -1

    nodes, _     = np.histogramdd(cells, bins, weights = node_)
    cond, _      = np.histogramdd(cells, bins, weights = condition)
    sel_cond     = cond > 0
    nn_nborder   = np.full(nodes.shape, 0, int)
    nn_iborder   = np.full(nodes.shape, -1, int)

    for move in moves(ndim):
        coors_next     = [cells[i] + steps[i] * move[i] for i in range(ndim)]
        nodes_next, _  = np.histogramdd(coors_next, bins, weights = node_)

        sel_nodes      = (nodes_next != nodes) & (nodes_next > 0)
                          
        sel            = mask & sel_cond & sel_nodes
        
        nn_nborder[sel] += 1

        usel            = sel & (nn_iborder == -1)
        nsel            = sel & (nn_iborder != nodes_next) & (nn_iborder != -1)
        nn_iborder[usel] = nodes_next[usel]
        nn_iborder[nsel] = -2
    

    isborder = (nn_nborder[mask] > 0).astype(bool)
    #isborder = nn_nborder[mask].astype(int)
    idborder = nn_iborder[mask].astype(int) - 1
    #print('is border ', np.sum(isborder))
    #print('id border ', np.sum(idborder == -3), np.sum(idborder >= 0))
    return isborder, idborder


def gradient_between_nodes(bins, mask, cells, weights, node,
                           condition = None, absolute = True):
    
    ndim, size   = len(cells), len(weights)
    steps        = [ibin[1] - ibin[0] for ibin in bins]

    condition        = np.full(size, True, dtype = bool) \
        if condition is None else condition
    enes             = np.copy(weights)
    enes[~condition] = 0.
    ids              = np.arange(size)

    potential, _ = np.histogramdd(cells, bins, weights = enes)
    nodes, _     = np.histogramdd(cells, bins, weights = node)
    cond, _      = np.histogramdd(cells, bins, weights = condition)
    kids, _      = np.histogramdd(cells, bins, weights = ids)

    factor       = 1 if absolute else 0
    nn_potential = factor * np.copy(potential)
    nn_kids      =          np.copy(kids) .astype(int)
    sel_cond     = cond == True    
    
    #moves = get_moves_updown(ndim)
    for move in moves(ndim):

        coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
        potential_next, _  = np.histogramdd(coors_next, bins, weights = enes)
        nodes_next, _      = np.histogramdd(coors_next, bins, weights = node)
        kids_next, _       = np.histogramdd(coors_next, bins, weights = ids)

        sel_node           = nodes_next != nodes
        sel_pot_next       = potential + potential_next > nn_potential
        sel                = (mask) & (sel_cond) & (sel_node) & (sel_pot_next)
        
        nn_potential[sel]  = potential[sel] + potential_next[sel]
        nn_kids     [sel]  = kids_next     [sel]


    lgrad = nn_potential[mask]
    lpath = nn_kids     [mask]
    
    return lgrad, lpath


# def find_passes(node, lpath, condition = None):

#     size      = len(node)
#     condition = np.full(size, True, dtype = bool) if condition is None else condition
#     kid       = np.arange(size)
#     ispass    = np.full(size, False, dtype = bool)

#     sel_dir   = (kid       == lpath[ lpath[ kid ] ])
#     sel_nodes = (node[kid] != node [ lpath[ kid ] ])
#     sel       = (sel_dir) & (sel_nodes) & (condition)
    
#     ispass[sel] = True
    
#     return ispass    


def find_passes(node, lpath, lgrad, condition = None):
    
    size      = len(node)
    condition = np.full(size, True, dtype = bool) if condition is None else condition
    kid       = np.arange(size)
    is_pass   = np.full(size, False, dtype = bool)    
    
    sel_dir    = (kid  == lpath[ lpath ])
    sel_border = (node != node[ lpath ])
    nids       = np.unique(node[sel_border]) 
    #print('nodes ', nids)
    
    for i, n0 in enumerate(nids):
        for j, n1 in enumerate(nids[ i+1 : ]):
            sel_border    = (node == n0) & (node[lpath] == n1)
            if (np.sum(sel_border) <= 0): continue # No contiguous node
            sel_condition = condition & condition[lpath] 
            #print('passes? ', n0, n1, np.sum(sel_border))
            sel = (sel_border) & (sel_dir) & (sel_condition) 
            #print('passes? ', n0, n1, np.sum(sel_border), np.sum(sel))
            sel = sel if np.sum(sel)>0 else (sel_border) & (sel_condition)
            if (np.sum(sel) > 0):
                ipos   = np.argmax(lgrad[sel])
                k0     = kid[sel][ipos]            
                is_pass[k0]        = True
                is_pass[lpath[k0]] = True
                
            
    return is_pass
        
        
def find_new_ridge(ispass, epath, elink):
    
    paths = get_new_ridges(ispass, epath, elink)

    size    = len(ispass)
    isridge = np.full(size, False, dtype = bool)

    if (len(paths) == 0):return isridge
    
    path = functools.reduce(operator.add, paths)
    path  = np.unique(path)
    isridge[path] = True
    
    return isridge
    

def energy_in_ridge(enes, ridge, epath):
    size     = len(enes)
    kids     = np.arange(size)
    eridge   = np.zeros(size, dtype = float)
    path     = np.argwhere(ridge)
    for kid in kids:
        kpath = get_path_to_path(kid, epath, path)
        #print('kid ', kid, 'path ', kpath)
        if (len(kpath) <= 0): continue
        jid   = kpath[-1]  
        eridge[jid] += enes[kid]
        #print('kid ', kid, 'path ', kpath, 'jid', jid, 'enes ', eridge[jid])
    return eridge

    
#
#--- Links
#


def get_links(ispass, elink):
            
    ids   = list(np.argwhere(ispass))
    links = [(i, elink[i]) for i in ids] 
        
    return links


def get_new_ridges(ispass, epath, elink, condition = None):
        
    links = get_links(ispass, elink)
    
    size      = len(ispass)
    condition = np.full(size, True, dtype = bool) if condition is None else condition

    
    links = [(i0, i1) for i0,i1 in links if (condition[i0]) & (condition[i1])]
    
    paths = [get_path_from_link(*link, epath) for link in links]
    
    return paths
 
#
#--- Utilities
#

def ut_scale(values, a = 0, b = 1):
   
    xmin, xmax = np.min(values), np.max(values)
    scale  = (values - xmin)/(xmax - xmin)
    return scale

def ut_centers(xs : np.array) -> np.array:
    return 0.5* ( xs[1: ] + xs[: -1])


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
   

def cells_select(cells, sel):
    return [cell[sel] for cell in cells]


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
    return [val[sel] for val in cells]


def get_segment(cells, kids):
    """ Fron a list of local IDs returns a segment to plot
    inputs:
        cells: tuple(array), m-dim tuple with n-size array with the cells' cordinates positions
        kids: tuple(int), list of the ID to generate the segment
    """
    ndim = len(cells)
    segment = [np.array([float(cells[i][kid]) for kid in kids]) for i in range(ndim)]
    return segment



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
        
    


# Link = namedtuple('Link', ('scale', 'cells', 'nodes'))

# def get_links_save(enes, node, lpath, ispass):

#     # issue maybe we can order by the energy of the nodes!    
#     kids   = list(np.argwhere(ispass == True))
#     #print('kids ', kids)    
    
#     links = []
#     while len(kids) > 0:
#         k0   = int(kids[0])
#         k1   = int(lpath[k0])
#         n0   = int(node[k0])
#         n1   = int(node[k1])
#         ene  = float(enes[k0] + enes[k1])
#         links.append((ene, (k0, k1), (n0, n1)))
#         kids.remove(k0)
#         kids.remove(k1)

#     links = sorted(links, reverse = True)
#     links = [Link(*link) for link in links]
    
#     return links


# def get_nodes_in_links(links):
#     nodes = []
#     for link in links:
#         for ni in link.nodes:
#             if ni not in nodes: nodes.append(ni)
#     return nodes


# def get_chains_of_links(links, mode = 'loop'):

#     tree    = True  if mode == 'tree'   else False
#     branch  = True  if mode == 'branch' else False
    
#     def valid_link_(link, nodes_in):
#         n0, n1 = link.nodes
#         nodes_ = np.array((n0, n1)).astype(int)
#         sel    = np.isin(nodes_, nodes_in)
#         nsel   = np.sum(sel)
#         if (tree): 
#             return nsel == 1
#         if (branch):
#             nlast = nodes_in[-1]
#             return (nsel == 1) and ((nlast == n0) or (nlast == n1))
#         return nsel >= 1

#     def get_chain_(links):
#         nodes_in = list(links[0].nodes)
#         links_in = [links[0]]
#         links    = list(links[1:])
#         nlinks   = len(links) +1
#         while (len(links) < nlinks):
#             nlinks = len(links)
#             sels = [valid_link_(link, nodes_in) for link in links]
#             if (np.sum(sels) > 0):
#                 ii   = sels.index(True)
#                 link = links[ii]
#                 links   .remove(link)
#                 links_in.append(link)
#                 for ni in link.nodes:
#                     if ni not in nodes_in: nodes_in.append(ni)
#         return links_in, links
    
#     chains = []
#     while (len(links) > 0):
#         ichain, links = get_chain_(links)
#         chains.append(ichain)
    
#     return chains


#-----   CLEAN UP
#---------------------------------

# def clouds(coors, steps, weights):
#     """
#     inputs:
#         coors: tuple(array), m-dim tuple with k-size arrays with the coordinates of the hits
#         steps: tuple(float), m-dim tuple with the size in each coordinate of the cells
#         weights: array, k-size array with the energy/weight of the hits
#     returns:
#         pd: a Pandas DataFrame with a list of columns:
#     """

#     clouds_check(coors, steps, weights)

#     bins, icells, cells, cells_ene, \
#     cells_kid                   = clouds_potential(coors, steps, weights)
    
#     ndim, nsize                 = clouds_size(cells, cells_ene)
    
#     #print(' clouds size ', ndim, nsize)
#     cells_neighbours            = clouds_neighbours(bins, cells, cells_ene)
    
#     cells_egrad, cells_epath    = clouds_gradient (bins, cells, cells_ene,
#                                                    cells_kid)
    
#     cells_ngradbours            = clouds_neighbours_grad(bins, cells, 
#                                                          cells_kid,
#                                                          cells_epath)
   
#     cells_vgrad                 = clouds_vgrad(bins, cells, cells_ene)
#     cells_laplacian             = clouds_laplacian(bins, cells, cells_ene)
    
#     cells_node, cells_enode, \
#     cells_nodesize              = clouds_nodes(cells_ene, cells_kid, cells_epath)

#     cells_lgrad, cells_lnode, \
#     cells_lpath                 = clouds_gradient_link(bins, cells, cells_ene,
#                                                        cells_node, cells_kid)

#     cells_epass                 = clouds_passes(cells_ene, cells_node,
#                                                 cells_enode, cells_lnode,
#                                                 cells_kid, cells_lgrad,
#                                                 cells_lpath)

#     cells_track, cells_tnode, \
#     cells_tpass                 = clouds_tracks(cells_node, cells_enode,
#                                                 cells_epass, cells_lpath,
#                                                 cells_kid)

#     cells_crest, cells_ecrest = clouds_crests(cells_ene,
#                                                cells_tnode, cells_tpass,
#                                                cells_epath, cells_lpath,
#                                                cells_kid)

#     dat = {}
#     for i in range(ndim):
#         dat['x'+str(i)] = cells[i]            # positions of the cells
#     for i in range(ndim):
#         dat['k'+str(i)] = icells[i]           # index of the cells
#     dat['ene']          = cells_ene           # energy of the cells
#     dat['kid']          = cells_kid           # local-ID of the cells

#     dat['egrad']        = cells_egrad         # energy grandient of the cell
#     dat['epath']        = cells_epath         # local-ID cell that has the largest energy gradient to this cell
#     dat['neighbours']   = cells_neighbours    # number of neighbours cells
    
#     dat['ngradbours']   = cells_ngradbours    # number of neighbouts whose gradients point to this cells

#     det['vgrad']        = cells_vgrad         # module of the gradient in this cell
#     dat['laplacian']    = cells_laplacian     # laplacian of the cell

#     dat['enode']        = cells_enode         # energy of the node (sum of the energies) only for node-cells
#     dat['node']         = cells_node          # local-ID of the node-cell of this cell
#     #dat['inode']        = cells_inode         # indices of the nodes, sorted by energy (decreasing)
#     dat['sizenode']     = cells_nodesize      # number of cells in the node (only for node-cells)

#     dat['lgrad']        = cells_lgrad         # energy gradient with cells of different nodes
#     dat['lpath']        = cells_lpath         # local-ID of the cells with the largest energy gradient and different node
#     dat['lnode']        = cells_lnode         # local-ID of the node to which this cell is a border and it is linked to
#     dat['epass']        = cells_epass         # energy of the link between two cells of different nodes

#     dat['track']        = cells_track         # ID of the most energetic cells in the track
#     dat['tnode']        = cells_tnode         # ID of the most energetic cell-node for nodes in the track
#     dat['tpass']        = cells_tpass         # ID of the most energetic cell-node for passes in the track
#     #dat['ipass']        = cells_ipass        # indeces of the links, sorted by energy (decreasing)

#     dat['crest']       = cells_crest       # cell-ID of the most energy cell in the crest
#     dat['ecrest']      = cells_crest       # sum-energy of the cells that are associate to this cell-crest

#     return pd.DataFrame(dat)


# def clouds_mc(coors, steps, ene, coorsmc, enemc):

#     # clouds
#     dfclouds = clouds(coors, steps, ene)

#     # mc-ene
#     in_cells = get_values_in_cells(coors, steps, ene)
#     xmcene, _, _ = in_cells(coorsmc, enemc)
#     dfclouds['mcene'] = xmcene

#     #mcpaths # THINK: can put paths into cells?
#     mcpaths = get_mcpaths(coorsmc, enemc, in_cells)

#     return dfclouds, mcpaths


# #------- INTERNAL


# #--- utilities

# #arstep = ut.arstep

# def to_indices(cells, bins, dirs = None):
#     """ converts the cells x,y,z positions into indices (ix, iy, iz)
#     inputs:
#         cells: tuple(array) m-dim tuple with n-size arrays with the i-coordenate
#         bins : tuple(array) m-dim tuple with the bins for each i-coordinate
#         dirs : tuple(array) m-dim tuple with n-size arrays with the increase in the i-coordenate
#     """
#     xcells = cells if dirs is None else [cell + idir for cell, idir in zip(cells, dirs)]
#     icells =  [np.digitize(icell, ibin) - 1 for icell, ibin in zip(xcells, bins)]
#     return icells


# def get_moves_updown(ndim):

#     def u1(idim):
#         ui1 = np.zeros(ndim)
#         ui1[idim] = 1
#         return ui1.astype(int)
#     vs = []
#     for i in range(ndim):
#         vs.append(u1(i))
#         vs.append(-u1(i))
#     vs.pop(0)
#     return vs


# def get_moves(ndim):
#     """ returns movelments of combination of 1-unit in each direction
#     i.e for ndim =2 returns [(1, 0), (1, 1), (0, 1,) (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
#     """

#     u0 = np.zeros(ndim)
#     def u1(idim):
#         ui1 = np.zeros(ndim)
#         ui1[idim] = 1
#         return ui1.astype(int)

#     vs = (u0, u1(0), -1 * u1(0))
#     for idim in range(1, ndim):
#         us = (u0, u1(idim), -1 * u1(idim))
#         vs = [(vi + ui).astype(int) for vi in vs for ui in us]
#     vs.pop(0)

#     return vs


# #----------------------
# #     clouds
# #----------------------




# def get_values_in_cells(coors, steps, weights):
#     """ returns a function to locate values in cells
#     inputs: (same as clouds)
#         coors: tuple(array), m-dim tuple with k-size arrays with the coordinates of the hits
#         steps: tuple(float), m-dim tuple with the size in each coordinate of the cells
#         weights: array, k-size array with the energy/weight of the hits
#     returns:
#         in_cells(xcoors, values):
#             a function that locate that give xcoors (a m-dim tuple with p-size arrays with coordinates)
#             locate the values (a p-size array) into the cells.
#             That function returns an  n-size (n = number of cells) with the values in the cells
#             and a float with the value of sum of the values that are not in the pre-determined cells

#     """

#     ndim         = len(coors)
#     bins         = [arstep(x, step, True) for x, step in zip(coors, steps)]
#     potential, _ = np.histogramdd(coors, bins, weights = weights)

#     sel          = potential > 0
#     icells       = to_coors(np.argwhere(sel))

#     def in_cells(xcoors, values):
#         """ return the values in predefined cells
#         inputs:
#             xcoors: a m-dim tuple of p-size arrays with the coordinates
#             values: a p-size array with the values associated to the coordinates
#         returns:
#             vals  : a n-size array with teh values located in the pre-defined cells (there are n-cells)
#         """
#         hvals, _  = np.histogramdd(xcoors, bins, weights = values)
#         vals      = hvals[icells]
#         vsel      = hvals > 0.
#         outvals   = np.sum(hvals[vsel & ~sel])
#         outscope  = np.sum(values) - outvals - np.sum(vals)
#         #if (np.sum(vsel & sel) != np.sum(vsel)):
#         #    print('in_cells ', np.sum(vsel), np.sum(sel), np.sum(vsel & sel))
#         #print('hvals not in cells ', hvals[vsel & ~sel])
#         #assert (np.sum(vsel & sel) != np.sum(vsel)) # selected cells must be pre-defined
#         return vals, outvals, outscope

#     return in_cells



# #
# #   Clouds main pieces
# #-------------------------


# def clouds_check(coors, steps, weights):

#     ndim  = len(coors)
#     nsize = len(coors[0])

#     assert len(steps)   == ndim
#     for i in range(ndim): assert len(coors[i]) == nsize

#     assert len(weights) == nsize

#     return ndim, nsize


# def clouds_size(coors, weights = None):

#     ndim  = len(coors)
#     nsize = len(coors[0])

#     for i in range(ndim): assert len(coors[i]) == nsize
#     if (weights is not None):
#         assert len(weights) == nsize

#     return ndim, nsize


# def clouds_potential(coors, steps, weights):

#     ndim         = len(coors)
#     bins         = [arstep(x, step, True) for x, step in zip(coors, steps)]
#     potential, _ = np.histogramdd(coors, bins, weights = weights)

#     sel          = potential > 0
#     icells       = to_coors(np.argwhere(sel))
#     enes         = potential[icells]
#     nsize        = len(enes)
#     kids         = np.arange(nsize)

#     centers      = [ut_centers(ibin) for ibin in bins]
#     cells        = [centers[i][icells[i]] for i in range(ndim)]

#     return bins, icells, cells, enes, kids.astype(int)


# def clouds_neighbours(bins, cells, cells_ene):

#     steps        = [ibin[1] - ibin[0] for ibin in bins]

#     ndim, _      = len(cells), len(cells[0])

#     counts, _       = np.histogramdd(cells, bins, weights = cells_ene)

#     sel = counts > 0
#     counts[sel]   = 1

#     moves = get_moves(ndim)
#     for move in moves:
#         coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
#         potential_next, _  = np.histogramdd(coors_next, bins, weights = cells_ene)
#         isel                = potential_next > 0
#         counts[sel & isel] += 1

#     nbours = counts[sel].astype(int)

#     return nbours


# def clouds_gradient(bins, cells, cells_enes, cells_kids):
#     """ returns the maximum gradient of energy and the cell which this gradient points to for each cell
#     inputs:
#         bins : tupe(array), m-dim tuple with the bins in each coordinate
#         cells: tuple(array), m-dim tuple with n-size arrays with the cells coordinates
#         enes : array,        n-size array with the energy of the cells
#         cells_kid  :
#     returns:
#         cells_grad : array, n-size with the gradient to the link with respect to this cell
#         cells_kid  : array, n-size with the id of the gradient cell to this cell
#         cells_node : array, n-size with the id of the linked node to this cell
#     """

#     ndim, nsize  = clouds_size(cells, cells_enes)
#     steps        = [ibin[1] - ibin[0] for ibin in bins]

#     potential, _ = np.histogramdd(cells, bins, weights = cells_enes)
#     kids, _      = np.histogramdd(cells, bins, weights = cells_kids)

#     sel_cells    = potential > 0

#     nn_potential = np.copy(potential)
#     nn_kids      = np.copy(kids) .astype(int)

#     #moves = get_moves_updown(ndim)
#     moves = get_moves(ndim)
#     for move in moves:

#         coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
#         potential_next, _  = np.histogramdd(coors_next, bins, weights = cells_enes)
#         kids_next, _       = np.histogramdd(coors_next, bins, weights = cells_kids)

#         sel_pot            = potential_next > nn_potential

#         sel                = (sel_cells) & (sel_pot)
#         if (np.sum(sel) > 0):
#             nn_potential[sel]  = potential_next[sel]
#             nn_kids     [sel]  = kids_next     [sel]

#     vgrad  = nn_potential[sel_cells] - potential[sel_cells]
#     vkids  = nn_kids     [sel_cells].astype(int)

#     return vgrad, vkids


# def clouds_neighbours_grad(bins, cells, cells_kids, cells_epath):
#     """

#     """
    
#     ndim, nsize  = clouds_size(cells, cells_kids)
#     steps        = [ibin[1] - ibin[0] for ibin in bins]

#     counts, _    = np.histogramdd(cells, bins, weights = 1 + cells_kids)
#     nncounts     = np.full(counts.shape, 0, dtype = int) 
#     sel          = counts > 0
#     nncounts[sel]  = 1 
    
#     counts, _    = np.histogramdd(cells, bins, weights = cells_kids)
#     kids_center  = counts.astype(int)
#     kids_center[~sel] = -1
   
#     moves = get_moves(ndim)
#     for move in moves:
#         coors_next    = [cells[i] + steps[i] * move[i] for i in range(ndim)]
#         kids_next, _  = np.histogramdd(coors_next, bins, weights = cells_epath)
#         isel          = (kids_next == kids_center)
#         nncounts[sel & isel] += 1

#     nbours = nncounts[sel].astype(int)
#     return nbours


# def clouds_vgrad(bins, cells, cells_ene):

#     #ndim      = len(bins)    
#     counts, _ = np.histogramdd(cells, bins, weights = cells_ene)
#     sel       = counts > 0
    
#     grad      = np.gradient(counts)
#     grad2     = [igrad * igrad for igrad in grad]
#     vgrad2    = functools.reduce(operator.add, grad2)
#     vgrad     = np.sqrt(vgrad2)
    
#     return vgrad[sel]


# def clouds_nodes(cells_ene, cells_kid, cells_epath):

#     nsize = len(cells_ene)

#     # associate each cell to a node
#     cells_node  = [get_path(kid, cells_epath)[-1] for kid in cells_kid]
#     cells_node  = np.array(cells_node).astype(int)

#     # compute the energy of a node and assoticate to the cell enode
#     cells_enode    = np.zeros(nsize)
#     nodes = np.unique(cells_node)
#     for node in nodes:
#         sel   = cells_node == node
#         ene   = np.sum(cells_ene[sel])
#         cells_enode[node]    = ene

#     return cells_node, cells_enode



# def clouds_laplacian(bins, cells, cells_ene):
    
   
#     counts, _ = np.histogramdd(cells, bins, weights = cells_ene)
#     sel       = counts > 0
    
#     hess      = ridges.hessian(counts)
#     lapl      = ridges.laplacian(hess)
    
#     return lapl[sel]



# def clouds_borders(bins, cells, cells_nodes):
    
#     ndim         = len(cells)
#     steps        = [ibin[1] - ibin[0] for ibin in bins]

#     nodes, _     = np.histogramdd(cells, bins, weights = cells_nodes)
#     nn_border    = np.full(nodes.shape, False, dtype = bool)

#     sel_cells    = nodes > 0

#     #moves = get_moves_updown(ndim)
#     moves = get_moves(ndim)
#     for move in moves:

#         coors_next     = [cells[i] + steps[i] * move[i] for i in range(ndim)]
#         nodes_next, _  = np.histogramdd(coors_next, bins, weights = cells_nodes)

#         sel_nodes      = nodes_next != nodes

#         sel            = (sel_cells) & (sel_nodes)
#         nn_border[sel] = np.logical_or(nn_border[sel], nodes_next[sel])

#     return nn_border[sel_cells]
    
    
# def clouds_borders_gradient(bins, cells, cells_enes, cells_kids, cells_borders):
    
#     ndim         = len(cells)
#     steps        = [ibin[1] - ibin[0] for ibin in bins]

#     counts, _    = np.histogramdd(cells, bins, weights = cells_borders)
#     potential, _ = np.histogramdd(cells, bins, weights = cells_enes)
#     sel_cells    = potential > 0
#     borders, _   = np.histogramdd(cells, bins, weights = cells_borders)
#     sel_borders  = borders == True
#     kids, _      = np.histogramdd(cells, bins, weights = cells_kids)

#     enes                 = np.copy(cells_enes)
#     enes[~cells_borders] = 0.

#     nn_potential = 0 * np.copy(potential)
#     nn_kids      = np.copy(kids) .astype(int)
    
#     #moves = get_moves_updown(ndim)
#     moves = get_moves(ndim)
#     for move in moves:

#         coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
#         potential_next, _  = np.histogramdd(coors_next, bins, weights = enes)
#         kids_next, _       = np.histogramdd(coors_next, bins, weights = cells_kids)

#         sel_pot_next       = potential_next > nn_potential
#         sel                = (sel_cells) & (sel_borders) &(sel_pot_next)
        
#         nn_potential[sel]  = potential_next[sel]
#         nn_kids     [sel]  = kids_next     [sel]


#     cells_lgrad = nn_potential[sel_cells]
#     cells_lpath = nn_kids     [sel_cells]
    
#     return cells_lgrad, cells_lpath

    
# def clouds_gradient_link(bins, cells, cells_enes, cells_nodes, cells_kids):

#     ndim         = len(cells)

#     steps        = [ibin[1] - ibin[0] for ibin in bins]

#     nodes, _     = np.histogramdd(cells, bins, weights = cells_nodes)
#     potential, _ = np.histogramdd(cells, bins, weights = cells_enes)
#     kids, _      = np.histogramdd(cells, bins, weights = cells_kids)

#     sel_cells    = potential > 0

#     nn_potential = np.copy(potential)
#     nn_nodes     = np.copy(nodes).astype(int)
#     nn_kids      = np.copy(kids) .astype(int)

#     #moves = get_moves_updown(ndim)
#     moves = get_moves(ndim)

#     for move in moves:

#         coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
#         potential_next, _  = np.histogramdd(coors_next, bins, weights = cells_enes)
#         nodes_next, _      = np.histogramdd(coors_next, bins, weights = cells_nodes)
#         kids_next, _       = np.histogramdd(coors_next, bins, weights = cells_kids)

#         sel_nodes          = nodes_next != nodes
#         sel_pot_next       = potential + potential_next > nn_potential

#         sel = (sel_cells) & (sel_nodes) & (sel_pot_next)
#         nn_potential[sel] = potential[sel] + potential_next[sel]
#         nn_nodes    [sel] = nodes_next    [sel]
#         nn_kids     [sel] = kids_next     [sel]

#     link_grad  = nn_potential[sel_cells] #- potential[sel_cells]
#     link_nodes = nn_nodes    [sel_cells]
#     link_kids  = nn_kids     [sel_cells]

#     return link_grad, link_nodes, link_kids


# def clouds_passes_save(cells_ene, cells_node, cells_enode, cells_lnode,
#                        cells_kid, cells_lgrad, cells_lpath):

#     nsize  = len(cells_node)
#     cells_epass = np.zeros(nsize)
#     cells_ipass = np.full(nsize, -1).astype(int)

#     nodes_kid, _ = sorted_by_energy(cells_node, cells_enode)
#     #print('nodes ', nodes_kid)

#     sel_passes = (cells_kid == cells_lpath[cells_lpath[cells_kid]])
#     #print('possible passes ', np.sum(sel_passes))

#     for i, inode in enumerate(nodes_kid):
#         for jnode in nodes_kid[ i +1 :]:
#             sel  = np.logical_and(((cells_node == inode) & (cells_lnode == jnode)), sel_passes)
#             #print(' passes? ', inode, jnode, np.sum(sel))
#             if (np.sum(sel) > 0) :
#                 isel = np.argmax(cells_lgrad[sel])
#                 id1  = cells_kid [sel][isel]
#                 #print('index 1 ', id1)
#                 cells_epass[id1] = cells_ene[id1] + cells_ene[cells_lpath[id1]]

#     return cells_epass, cells_ipass


# def clouds_find_passes(cells_node , cells_kid, cells_lpath, cells_laplacian):

#     nsize        = len(cells_node)
#     cells_passes = np.full(nsize, False, dtype = bool)

#     sel_dir   = (cells_kid == cells_lpath[cells_lpath[cells_kid]])
#     sel_nodes = (cells_node[cells_kid] != cells_node[cells_lpath[cells_kid]])
#     sel_lap   = cells_laplacian < 0
#     sel       = (sel_dir) & (sel_nodes) & (sel_lap)
    
#     cells_passes[sel] = True
    
#     return cells_passes
    
    
# def clouds_passes(cells_ene, cells_node, cells_enode, cells_lnode,
#                   cells_kid, cells_lgrad, cells_lpath):

#     nsize  = len(cells_node)
#     cells_epass = np.zeros(nsize)
#     #cells_ipass = np.full(nsize, -1).astype(int)

#     nodes_kid, _ = sorted_by_energy(cells_node, cells_enode)
#     #print('nodes ', nodes_kid)

#     sel_passes = (cells_kid == cells_lpath[cells_lpath[cells_kid]])
#     #print('possible passes ', np.sum(sel_passes))

#     for i, inode in enumerate(nodes_kid):
#         for jnode in nodes_kid[ i +1 : ]:
#             sel  = np.logical_and(((cells_node == inode) & (cells_lnode == jnode)), sel_passes)
#             if (np.sum(sel) == 0):
#                 sel = np.logical_and((cells_node == inode), (cells_lnode == jnode))
#             #print(' passes? ', inode, jnode, np.sum(sel))
#             if (np.sum(sel) > 0) :
#                 isel = np.argmax(cells_lgrad[sel])
#                 id1  = cells_kid [sel][isel]
#                 #print('index 1 ', id1)
#                 cells_epass[id1] = cells_ene[id1] + cells_ene[cells_lpath[id1]]

#     return cells_epass #, cells_ipass

# #
# #--- Links  (temptative)
# #

# Link = namedtuple('Link', ('scale', 'cells', 'nodes'))

# def get_links_(cells_ene, cells_kid, cells_node, cells_lpath, cells_passes):

#     # issue maybe we can order by the energy of the nodes!    
#     kids   = list(cells_kid[cells_passes])
#     #print('kids ', kids)    
    
#     links = []
#     while len(kids) > 0:
#         k0 = kids[0]
#         k1 = cells_lpath[k0]
#         n0 = cells_node[k0]
#         n1 = cells_node[k1]
#         ene  = cells_ene[k0] + cells_ene[k1]
#         links.append((ene, (k0, k1), (n0, n1)))
#         kids.remove(k0)
#         kids.remove(k1)

#     links = sorted(links, reverse = True)
#     links = [Link(*link) for link in links]
    
#     return links

# def get_links_nodes_(links):
#     nodes = []
#     for link in links:
#         for ni in link.nodes:
#             if ni not in nodes: nodes.append(ni)
#     return nodes

# def get_links_chains_(links, mode = 'loop'):

#     tree    = True  if mode == 'tree'   else False
#     branch  = True  if mode == 'branch' else False
    
#     def valid_link_(link, nodes_in):
#         n0, n1 = link.nodes
#         nodes_ = np.array((n0, n1)).astype(int)
#         sel    = np.isin(nodes_, nodes_in)
#         nsel   = np.sum(sel)
#         if (tree): 
#             return nsel == 1
#         if (branch):
#             nlast = nodes_in[-1]
#             return (nsel == 1) and ((nlast == n0) or (nlast == n1))
#         return nsel >= 1

#     def get_chain_(links):
#         nodes_in = list(links[0].nodes)
#         links_in = [links[0]]
#         links    = list(links[1:])
#         nlinks   = len(links) +1
#         while (len(links) < nlinks):
#             nlinks = len(links)
#             sels = [valid_link_(link, nodes_in) for link in links]
#             if (np.sum(sels) > 0):
#                 ii   = sels.index(True)
#                 link = links[ii]
#                 links   .remove(link)
#                 links_in.append(link)
#                 for ni in link.nodes:
#                     if ni not in nodes_in: nodes_in.append(ni)
#         return links_in, links
    
#     chains = []
#     while (len(links) > 0):
#         ichain, links = get_chain_(links)
#         chains.append(ichain)
    
#     return chains


# def clouds_tracks_new(cells_ene, cells_kid, cells_node,
#                       cells_epath, cells_lpath, cells_passes):
    
#     ndim = len(cells_ene)
#     ctrack  = np.full(ndim, -1, dtype = int)
#     cpass   = np.full(ndim, -1, dtype = int)
    
#     links  = get_links_(cells_ene, cells_kid, cells_node, cells_lpath, cells_passes)
#     tracks = get_links_chains_(links)
#     #print('tracks ', len(tracks))
        
#     def track_ene(track):
#         nodes = get_links_nodes_(track)
#         enes  = [cells_ene[cells_node == n] for n in nodes]
#         return enes
    
#     enes =  [track_ene(track) for track in tracks]
#     vals = sorted(zip(enes, tracks))

#     tracks = [v[1] for v in vals]    

#     for i, track in enumerate(tracks):
#         #print('track ', i, track)
#         for j, link in enumerate(track):
#             for ni in link.nodes:
#                 ctrack[cells_node == ni] = i
#             for ki in link.cells:
#                 cpass[ki] = j
                    
#     return ctrack, cpass


# def clouds_extreme_nodes(cells_enode, cells_node, cells_passes):
    
#     ndim       = len(cells_node)
#     cnodes_ext = np.full(ndim, False, dtype = bool)
    
#     nodes = np.unique(np.argwhere(cells_enode > 0))
#     for node in nodes:
#         npass = np.sum(cells_passes[cells_node == node])
#         cnodes_ext[node] = (npass == 1)
    
#     return cnodes_ext


# def clouds_outer(cells, bins, cells_ene, cells_region):
    
#     ndim    = len(cells)
#     size    = len(cells[0])
#     steps   = [ibin[1] - ibin[0] for ibin in bins]
#     couter  = np.full(size, False, dtype = bool)     
    
#     potential, _ = np.histogramdd(cells, bins, weights = cells_ene)
#     sel_cells    = potential > 0
#     potential, _ = np.histogramdd(cells, bins, weights = cells_region)
#     sel_region   = potential > 0
    
#     nn_neighbours = np.full(potential.shape, 0, dtype = int)
#     moves = get_moves(ndim)
#     for move in moves:

#         coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
#         potential_next, _  = np.histogramdd(coors_next, bins,
#                                             weights = cells_region)

#         sel_pot_next = potential_next > 0
#         sel = (sel_cells) & (sel_region) & (sel_pot_next)

#         nn_neighbours[sel] +=1 
        
#     nn            = 3 ** ndim - 1 
#     nn_neighbours = nn_neighbours < nn
#     nn_neighbours[~sel_region] = False
#     couter        = nn_neighbours[sel_cells]

#     return couter


# def clouds_extreme_cells(cells, bins, cells_ene, cells_node,
#                          cells_epath, cells_dispersive, 
#                          cells_nodes_ext):
    
#     ndim         = len(cells_ene)
#     cells_ext     = np.full(ndim, False, dtype = bool)
    
#     nodes = np.unique(np.argwhere(cells_nodes_ext == True))
    
#     for node in nodes:
        
#         region      = (cells_node == node) & cells_dispersive
#         print('region ', node, np.sum(region))
#         cells_outer = clouds_outer(cells, bins, cells_ene, region)
#         kids        = np.argwhere(cells_outer == True)
#         print('kids ', kids)
#         print('enes ', cells_ene[kids])
#         paths       = [get_path(kid, cells_epath) for kid in kids]
#         enes        = [np.sum([cells_ene[i] for i in path]) for path in paths]
#         print('enes path ', enes)
#         ii          = np.argmax(enes)
#         kid         = kids[ii]
#         print('extreme ', kid)
        
#         cells_ext[kid] = True
        
#     return cells_ext
        
        
        
        

# #def get_tracks_paths(cells_trk, cells_tpass, cells_ends):

# #def get_tracks_ids(cells_trk, cells_trktime):
# #    ntrks   = np.max(cells_trk) + 1
# #    trks    = []
# #    for itrk in range(ntrks):
# #        trktime = cells_trktime >= itrk 
# #        sel     = (cells_trktime >= 0) & (cells_trk == 0)
# #        ids     = np.argwhere(sel)
# #        vals    = sorted(zip(trktime, ids))
# #        ids     = [int(v[1]) for v in vals]
# #        trks.append(ids)
# #    return trks


# #--- clouds again

# def clouds_tracks(cnode, enodes, epasses, lpaths, kids):

#     sel         = enodes > 0
#     knodes, _   = sorted_by_energy(kids[sel], enodes[sel])
#     sel         = epasses  > 0
#     kpasses,  _ = sorted_by_energy(kids[sel], epasses[sel])
#     #print('passes', kpasses, _)

#     kstaples = [tuple((cnode[kid], cnode[lpaths[kid]], kid)) for kid in kpasses]

#     def valid_pass_(staple, nodes_in):
#         nodes_ = np.array((staple[0], staple[1])).astype(int)
#         sel    = np.isin(nodes_, nodes_in)
#         return (np.sum(sel) == 1)


#     def new_track(xnodes, xstaples):
#         nodes      = list(xnodes[1:] )
#         nodes_in   = [xnodes[0],]
#         staples    = list(xstaples)
#         staples_in = []
#         nstaples   = len(staples) +1
#         while (len(staples) < nstaples):
#             nstaples = len(staples)
#             sels = [valid_pass_(staple, nodes_in) for staple in staples]
#             if (np.sum(sels) > 0):
#                 staple = np.array(staples)[sels][0]
#                 staples_in.append(tuple(staple))
#                 ii    = sels.index(True)
#                 staples.pop(ii)
#                 inode1, inode2 = staple[0], staple[1]
#                 for inode in [inode1, inode2]:
#                     if inode not in nodes_in:
#                         nodes_in.append(inode)
#                         nodes.remove(inode)
#         return nodes_in, staples_in, nodes, staples

#     nsize    = len(cnode)
#     tracks  = np.full(nsize, -1).astype(int)
#     tnodes  = np.full(nsize, -1).astype(int)
#     tpasses = np.full(nsize, -1).astype(int)
#     while (len(knodes) > 0):
#         track_nodes, track_staples, knodes, kstaples = new_track(knodes, kstaples)
#         #print('track nodes   ! ', track_nodes)
#         #print('track staples   ', track_staples)
#         main_node = track_nodes[0]
#         for inode in track_nodes:
#             tracks[cnode == inode]      = main_node
#             tnodes[inode]               = main_node
#         for staple in track_staples:
#             kid = staple[2]
#             tpasses[kid]                = main_node
#              #print('tpass ', kid)

#     return tracks, tnodes, tpasses


# def clouds_crests(enes, tnode, tpass, epath, lpath, ckids):

#     nsize  = len(tpass)
#     trange = np.full(nsize, -1).astype(int)

#     trange[ckids[tnode > -1]] = ckids[tnode > -1]

#     tracks  = np.unique(tpass[tpass > -1])
#     for track in tracks:
#         #print('track ', track)
#         kids  = ckids[tpass  == track]
#         #print('passes ', kids)
#         paths = [get_pass_path(kid, epath, lpath) for kid in kids]
#         #print('paths ', paths)
#         path  = paths[0]
#         for ipath in paths[0:]: path += ipath
#         path  = np.unique(path)
#         #print('path ', path)
#         trange[path] = track


#     erange = np.zeros(nsize)
#     tkids = ckids[trange > -1]
#     #print(tkids)
#     def _irange(kid):
#         if kid in tkids: return kid
#         kid  = epath[kid]
#         return _irange(kid)

#     for kid in ckids:
#         erange[_irange(kid)] += enes[kid]


#     return trange, erange

# #----------------------
# # MC Clouds
# #----------------------


# # def clouds_mc(coors, steps, ene, coorsmc, enemc):

# #     # clouds
# #     dfclouds = clouds(coors, steps, ene)

# #     # mc-ene
# #     in_cells = get_values_in_cells(coors, steps, ene)
# #     xmcene, _, _ = in_cells(coorsmc, enemc)
# #     dfclouds['mcene'] = xmcene

# #     #mcpaths # THINK: can put paths into cells?
# #     mcpaths = get_mcpaths(coorsmc, enemc, in_cells)

# #     return dfclouds, mcpaths


# def get_mcpaths(xcoors, enemc, in_cells):

#     def dis(i):
#         if (i == 0): return 0.
#         dd2 = np.sqrt(np.sum([(xcoor[i] - xcoor[i-1]) * (xcoor[i] - xcoor[i-1]) for xcoor in xcoors]))
#         return dd2

#     ndim, nsize  = len(xcoors), len(xcoors[0])
#     dds = [dis(i) for i in range(nsize)]
#     def locate_in_cell(xcoor, val):
#         vals, _, _ = in_cells(xcoor, val)
#         return np.argwhere(vals > 0)

#     xpos = [locate_in_cell([[xcoor[i],] for xcoor in xcoors], [enemc[i],]) for i in range(nsize)]

#     paths   = []
#     path    = []
#     prevkid = -1
#     for i, dd in enumerate(dds):
#         ckid = int(xpos[i][0]) if len(xpos[i]) == 1 else -1
#         if (ckid != prevkid): # to a different cells
#             if (ckid != -1):
#                 if (dd > 2.):
#                     if (len(path) > 0): paths.append(path)
#                     path = [ckid]
#                 else:
#                     path.append(ckid)
#             #else:
#             #    print('empty!', i, ckid, prevkid)
#         prevkid = ckid
#     paths.append(path)
#     #print(paths)
#     return paths

# # SAVE THIS CODE
# #
# # ncells  = len(dfclouds.ene)
# # print(ncells)
# # mcinit  = np.full(ncells, -1)
# # mcpath  = np.full(ncells, -1)
# # mcpath[mcene > 0] = dfclouds.kid[mcene > 0]
# # mccells = [-1,]
# # paths = []
# # path  = []
# # prevkid = -1
# # lastkid = -1
# # for i, dd in enumerate(dds):
# #     ckid = int(xpos[i][0]) if len(xpos[i]) == 1 else -1
# #     if (ckid != prevkid): # to a different cells
# #         if (ckid not in mccells):
# #             mccells.append(ckid)
# #             if (dd > 2.):
# #                 # new init
# #                 if (len(path) > 0): paths.append(path)
# #                 path = [ckid]
# #                 mcinit[ckid] = ckid
# #                 lastkid      = ckid
# #             else:
# #                 # continue
# #                 if (len(path) == 0):
# #                     mcinit[ckid] = ckid
# #                 path.append(ckid)
# #                 if (lastkid != -1): mcpath[lastkid] = ckid
# #                 lastkid = ckid
# #         else:
# #             # new empty
# #             if (len(path) > 0): paths.append(path)
# #             path = []
# #             lastkid = -1
# #     prevkid = ckid
# # paths.append(path)
# # print(mccells)
# # for path in paths:
# #     print('path', path[0], path[-1], ', nodes: ', path)
# # npaths = len(paths)
# # #for i in range(npaths):
# # #    for j in range(i +1, npaths):
# # #        print(np.sum(np.isin(paths[j], paths[i])))
# #
# # #print(mcpath)
# # for path in paths:
# #     print('init ', path[0], np.isin(path[0], mcinit[mcinit > -1]))
# # ipaths = [path[0] for path in paths]
# # print('inits ', sorted(mcinit[mcinit > -1]), 'len ', len(mcinit[mcinit > -1]))
# # print('inits ', sorted(ipaths), 'len ', len(ipaths))
# #
# #
# # #for path in paths:
# # #    print('init ', path[-1],  mcpath[path[-1]] == path[-1]))
# # lpaths = [path[-1] for path in paths]
# # for path in paths:
# #     print('end ', path[-1], np.isin(path[-1], mcpath[mcpath == dfclouds.kid]))
# # print('ends ', sorted(mcpath[mcpath == dfclouds.kid]), 'len ', len(mcpath[mcpath == dfclouds.kid]))
# # print('ends ', sorted(lpaths), 'len ', len(lpaths))
# #

# #HERE



# #
# #   Post-cloud utils
# #-------------------------------


# def sorted_by_energy(kids, enes):
#     """ return the ids ordered by energy
#     inputs:
#         kids : array(int), array wih the IDs
#         enes : array(float), array with the energies
#     returs:
#         xkids: array(int), array with the IDsordeed by energy (decreasing)
#         enes : array(float), array with the energies in decreasing order
#     """
#     sel = enes > 0.
#     nodes = sorted(zip(enes[sel], kids[sel]))
#     nodes.reverse()
#     xenes, xkids = [node[0] for node in nodes], [node[1] for node in nodes]
#     return xkids, xenes



# def get_path(kid, next_kid):
#     """ given a kid, a local ID and the link array next_kid return the path of
#     consecutive kids
#     inputs:
#         kid: int, local ID of the array to get the path
#         next_kid: array, n-size dimension with the associated local ID cells to a give cell ID
#     """
#     path = []
#     while next_kid[kid] != kid:
#         path.append(kid)
#         kid = next_kid[kid]
#     path.append(kid)
#     return path


# def get_link_path(link, epath):
#     kid0, kid1 = link
#     path1 = get_path(kid0       , epath)
#     path2 = get_path(kid1, epath)
#     path1.reverse()
#     path = path1 + path2
#     return path


# def get_pass_path(kid, epath, lpath):
#     """ return the path (list of cells IDs) given an ID of a pass-cell
#     inputs:
#         kid   : int, ID of the pass-cell
#         epath : array(int), array with the IDs of the gradients
#         lpath : array(int), array with the IDs of the links
#     returns:
#         path  : array(int), array with the IDs of the cells in the path of this pass
#     """
#     path1 = get_path(kid       , epath)
#     path2 = get_path(lpath[kid], epath)
#     path1.reverse()
#     path = path1 + path2
#     return path


# def get_segment(cells, kids):
#     """ Fron a list of local IDs returns a segment to plot
#     inputs:
#         cells: tuple(array), m-dim tuple with n-size array with the cells' cordinates positions
#         kids: tuple(int), list of the ID to generate the segment
#     """
#     ndim = len(cells)
#     segment = [np.array([float(cells[i][kid]) for kid in kids]) for i in range(ndim)]
#     return segment


# #--- Function with passes

# def trim(evt : pd.DataFrame, 
#          emin: float = 0.04):
#     """
    
#     trim the extreme nodes with low energy

#     Parameters
#     ----------
#     evt  : pd.DataFrame, cloulds dF
#     emin : float, optional. Minimum value of the extreme node energy
#             The default is 0.04.

#     Returns
#     -------
#     evt : pd.DataFrame, clouds DF with the trimmed off extreme weak nodes
    
#     TODO: revisit!
#     """
    
#     kids  = evt.kid  .values
#     enode = evt.enode.values
#     tpass = evt.tpass.values
#     node  = evt.node .values
#     lnode = evt.lnode.values
    
#     passes  = get_passes(tpass, node, lnode)
#     dpasses = get_passes_dict(passes)
    
#     knodes = [kid for kid in dpasses.keys() if len(dpasses[kid]) == 1]
#     enes   = [float(enode[kids == kid]) for kid in knodes]
#     #print(knodes)
#     #print(enes)
    

#     knodes = [kid for kid, ene in zip(knodes, enes) if ene < emin]
#     #print('removing ', knodes)
    
#     for kid in knodes:
#         sel = (evt.node == kid).values
#         #print(np.sum(sel))
#         evt['tpass'][sel] = -1
#         evt['epass'][sel] = -1
#         evt['track'][sel] = -1
        
#         sel = (evt.lnode == kid).values
#         #print(np.sum(sel))
#         evt['tpass'][sel] = -1
#         evt['epass'][sel] = -1
#         evt['lnode'][sel] = -1
        
#     return evt


# def get_passes(epass, node, lnode):
#     """
    
#     return the list of pair (node0, node1) with the passes between nodes

#     Parameters
#     ----------
#     epass : np.array, potential of the pass for this cell
#     node  : np.array(int), k-index of the node for this cell
#     lnode : np.array(int), k-index of the lined node for this cell

#     Returns
#     -------
#     passes : list( (int, int) ), list of passes, pairs of (node0, node1)
#     """
    
#     ksel   = epass > 0.
#     passes = list(zip(node[ksel], lnode[ksel]))
    
#     #passes = list(zip(node[ksel], lnode[ksel]))
#     #unodes = list(node[ksel]) + list(lnode[ksel])
#     #knodes = np.unique(unodes)
    
#     #dpasses = {}
#     #for knode in knodes:
#     #    dpasses[knode] = [ipass for ipass in passes if np.isin(knode, ipass)]
    
#     return passes

# def get_passes_dict(passes):
#     """
    
#     from the list of passes (node0, node1) returns a dictionary that for each node key
#     returns the passes of that node

#     Parameters
#     ----------
#     passes : list( (int, int)), list of passes (node0, node1)
#         DESCRIPTION.

#     Returns
#     -------
#     dpasses : dict(int) = list( (node0, node1)), dictionary, for each node-id returns the list of passes

#     """
    
#     knodes = np.unique(np.concatenate(list(passes)))
#     dpasses = {}
#     for knode in knodes:
#         dpasses[knode] = [ipass for ipass in passes if np.isin(knode, ipass)]
        
#     return dpasses
    

# def nodes_idistance(passes):
#     """
    
#     From the list of passes compute the step distance of each node to an extreme of the track
    
#     Parameters
#     ----------
#     passes : list( (int, int)), list of passes, each pass is a tuple (node0, node1)

#     Returns
#     -------
#     udist : dic(int) = int, returns the number of node steps of the current node to an extreme

#     """
    
    
#     passes  = list(passes)
#     dpasses = get_passes_dict(passes)
#     knodes  = dpasses.keys()

#     end_kids = np.array([k for k in knodes if (len(dpasses[k]) == 1)])
#     #print(end_kids)

#     udist = {}
#     i = 1
#     for kid in end_kids:
#         udist[kid] = 1
#     ok = True
#     while ok > 0:
#         i = i +1
#         upasses = [pair for pair in passes if np.sum(np.isin(pair, end_kids)) > 0]
#         uus = np.unique(np.concatenate(upasses))
#         end_kids = uus[~np.isin(uus, end_kids)]
#         for kid in end_kids:
#             udist[kid] = i
#         #print(i, end_kids)
#         for upass in upasses:
#             passes.pop(passes.index(upass))
#         ok = (len(passes) > 0) and (len(end_kids) >0)
        
#     return udist


# def get_function_branches(passes):
#     """
    
#     returns a function, branches, that branches(int) where int is the id of the node
#     returns the list of branches starting from this node, each branch is a list of consecutive nodes

#     Parameters
#     ----------
#     passes : tuple( (int, int)), list of passes, each pass is (node0, node1)

#     Returns
#     -------
#     branches; callable, branches(int) return the list of branches starting in that node

#     """
    
#     dpasses = get_passes_dict(passes)
    
#     def grow_branch(path):
#         ik = path[-1]
#         if (ik not in dpasses.keys()):
#             return []
#         kpasses = dpasses[ik]
#         paths   = []
#         for i, ipass in enumerate(kpasses):
#             ipass = np.array(ipass, int)
#             ik0, ik1 = ipass
#             ksel = np.isin(ipass, path)
#             #print(ipass, path, ksel)
#             ipath = list(path)
#             if (np.sum(ksel) == 2): continue
#             ikn   = ipass[~ksel][0] 
#             ipath = list(path)
#             ipath.append(ikn)
#             paths.append(ipath)        
#         return paths

#     def get_branches(kid):    
#         branches = []
#         paths = [[kid],]
#         while len(paths) > 0:
#             npaths = [] 
#             for path in paths:
#                 nipaths = grow_branch(path)
#                 if (len(nipaths)  == 0): 
#                     branches.append(list(path))
#                     #print('branch ', path)
#                 else:
#                     npaths += nipaths
#                     #print('paths ', nipaths)
#                 paths = npaths
#         return branches
    
#     return get_branches
