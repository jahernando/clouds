import numpy             as np
import pandas            as pd
#import tables            as tb


def clouds(coors, steps, weights):
    """
    inputs:
        coors: tuple(array), m-dim tuple with k-size arrays with the coordinates of the hits
        steps: tuple(float), m-dim tuple with the size in each coordinate of the cells
        weights: array, k-size array with the energy/weight of the hits
    returns:
        pd: a Pandas DataFrame with a list of columns:
    """

    clouds_check(coors, steps, weights)

    bins, icells, cells, cells_ene, \
    cells_kid                   = clouds_potential(coors, steps, weights)
    ndim, nsize                 = clouds_size(cells, cells_ene)
    #print(' clouds size ', ndim, nsize)
    cells_neighbours            = clouds_neighbours(bins, cells, cells_ene)
    cells_egrad, cells_epath    = clouds_gradient (bins, cells, cells_ene,
                                                   cells_kid)
    cells_node, cells_enode, \
    cells_nodesize              = clouds_nodes(cells_ene, cells_kid, cells_epath)

    cells_lgrad, cells_lnode, \
    cells_lpath                 = clouds_gradient_link(bins, cells, cells_ene,
                                                       cells_node, cells_kid)

    cells_epass, cells_ipass    = clouds_passes(cells_ene, cells_node,
                                                cells_enode, cells_lnode,
                                                cells_kid, cells_lgrad,
                                                cells_lpath)

    cells_track, cells_tnode, \
    cells_tpass                 = clouds_tracks(cells_node, cells_enode,
                                                cells_epass, cells_lpath,
                                                cells_kid)

    cells_crest, cells_ecrest = clouds_crests(cells_ene,
                                               cells_tnode, cells_tpass,
                                               cells_epath, cells_lpath,
                                               cells_kid)

    dat = {}
    for i in range(ndim):
        dat['x'+str(i)] = cells[i]            # positions of the cells
    for i in range(ndim):
        dat['k'+str(i)] = icells[i]           # index of the cells
    dat['ene']          = cells_ene           # energy of the cells
    dat['kid']          = cells_kid           # local-ID of the cells

    dat['egrad']        = cells_egrad         # energy grandient of the cell
    dat['epath']        = cells_epath         # local-ID cell that has the largest energy gradient to this cell
    dat['neighbours']   = cells_neighbours    # number of neighbours cells

    dat['enode']        = cells_enode         # energy of the node (sum of the energies) only for node-cells
    dat['node']         = cells_node          # local-ID of the node-cell of this cell
    #dat['inode']        = cells_inode         # indices of the nodes, sorted by energy (decreasing)
    dat['sizenode']     = cells_nodesize      # number of cells in the node (only for node-cells)

    dat['lgrad']        = cells_lgrad         # energy gradient with cells of different nodes
    dat['lpath']        = cells_lpath         # local-ID of the cells with the largest energy gradient and different node
    dat['lnode']        = cells_lnode         # local-ID of the node to which this cell is a border and it is linked to
    dat['epass']        = cells_epass         # energy of the link between two cells of different nodes

    dat['track']        = cells_track         # ID of the most energetic cells in the track
    dat['tnode']        = cells_tnode         # ID of the most energetic cell-node for nodes in the track
    dat['tpass']        = cells_tpass         # ID of the most energetic cell-node for passes in the track
    #dat['ipass']        = cells_ipass        # indeces of the links, sorted by energy (decreasing)

    dat['crest']       = cells_crest       # cell-ID of the most energy cell in the crest
    dat['ecrest']      = cells_crest       # sum-energy of the cells that are associate to this cell-crest

    return pd.DataFrame(dat)


def clouds_mc(coors, steps, ene, coorsmc, enemc):

    # clouds
    dfclouds = clouds(coors, steps, ene)

    # mc-ene
    in_cells = get_values_in_cells(coors, steps, ene)
    xmcene, _, _ = in_cells(coorsmc, enemc)
    dfclouds['mcene'] = xmcene

    #mcpaths # THINK: can put paths into cells?
    mcpaths = get_mcpaths(coorsmc, enemc, in_cells)

    return dfclouds, mcpaths


#------- INTERNAL


#--- utilities

def ut_centers(xs : np.array) -> np.array:
    """ returns the center between the participn
    inputs:
        xs: np.array
    returns:
        np.array with the centers of xs (dimension len(xs)-1)
    """
    return 0.5* ( xs[1: ] + xs[: -1])

def arstep(x, step, delta = False):
    """ returns an array with bins of step size from x-min to x-max (inclusive)
    inputs:
        x    : np.array
        step : float, step-size
    returns:
        np.array with the bins with step size
    """
    delta = step/2 if delta else 0.
    return np.arange(np.min(x) - delta, np.max(x) + step + delta, step)


#arstep = ut.arstep

def to_indices(cells, bins, dirs = None):
    """ converts the cells x,y,z positions into indices (ix, iy, iz)
    inputs:
        cells: tuple(array) m-dim tuple with n-size arrays with the i-coordenate
        bins : tuple(array) m-dim tuple with the bins for each i-coordinate
        dirs : tuple(array) m-dim tuple with n-size arrays with the increase in the i-coordenate
    """
    xcells = cells if dirs is None else [cell + idir for cell, idir in zip(cells, dirs)]
    icells =  [np.digitize(icell, ibin) - 1 for icell, ibin in zip(xcells, bins)]
    return icells


def to_coors(vs):
    """ convert a list of m-size of vector of n-dim into n-dim list of coordinates eatchof m-dize (x1, x2,x)
    """
    ndim = len(vs[0])
    xs = [[vi[i] for vi in vs] for i in range(ndim)]
    return xs


# def to_vectors(vals):
#     """ convert a n-list of m-size into a list of m-size of n-vectors
#     """
#     ndim, nsize = len(vals), len(vals[0])
#     vvals = [np.array([val[k] for val in vals]) for k in range(nsize)]
#     return np.array(vvals)
#
# def to_ids(icoors, scale = 1000):
#     """ generate a unique id for coordinates (x1, x2, ...), xi
#     a m-size arrpy with the xi-components
#     icoor are always integer and positive indices!!
#     """
#     #ndim = len(icoors)
#     #icoors = icoors if (type(icoors[0]) == int) else [(ix,) for ix in icoors]
#
#     #if (type(icoors[0]) == int):
#     #    icoors = [(icoors[i],) for i in range(ndim)]
#
#     ndim, nsize = len(icoors), len(icoors[0])
#
#     #ndim  = len(icoors)
#     #nsize = len(icoors[0])
#     #for i in range(ndim): assert len(icoors[i]) == nsize
#
#     ids  = [np.sum([(scale**i) * icoors[i][k] for i in range(ndim)]) for k in range(nsize)]
#     ids  = np.array(ids).astype(int)
#
#     #if (nsize == 1): ids = ids[0]
#     return ids

def get_moves_updown(ndim):

    def u1(idim):
        ui1 = np.zeros(ndim)
        ui1[idim] = 1
        return ui1.astype(int)
    vs = []
    for i in range(ndim):
        vs.append(u1(i))
        vs.append(-u1(i))
    vs.pop(0)
    return vs


def get_moves(ndim):
    """ returns movelments of combination of 1-unit in each direction
    i.e for ndim =2 returns [(1, 0), (1, 1), (0, 1,) (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    """

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


#----------------------
#     clouds
#----------------------




def get_values_in_cells(coors, steps, weights):
    """ returns a function to locate values in cells
    inputs: (same as clouds)
        coors: tuple(array), m-dim tuple with k-size arrays with the coordinates of the hits
        steps: tuple(float), m-dim tuple with the size in each coordinate of the cells
        weights: array, k-size array with the energy/weight of the hits
    returns:
        in_cells(xcoors, values):
            a function that locate that give xcoors (a m-dim tuple with p-size arrays with coordinates)
            locate the values (a p-size array) into the cells.
            That function returns an  n-size (n = number of cells) with the values in the cells
            and a float with the value of sum of the values that are not in the pre-determined cells

    """

    ndim         = len(coors)
    bins         = [arstep(x, step, True) for x, step in zip(coors, steps)]
    potential, _ = np.histogramdd(coors, bins, weights = weights)

    sel          = potential > 0
    icells       = to_coors(np.argwhere(sel))

    def in_cells(xcoors, values):
        """ return the values in predefined cells
        inputs:
            xcoors: a m-dim tuple of p-size arrays with the coordinates
            values: a p-size array with the values associated to the coordinates
        returns:
            vals  : a n-size array with teh values located in the pre-defined cells (there are n-cells)
        """
        hvals, _  = np.histogramdd(xcoors, bins, weights = values)
        vals      = hvals[icells]
        vsel      = hvals > 0.
        outvals   = np.sum(hvals[vsel & ~sel])
        outscope  = np.sum(values) - outvals - np.sum(vals)
        #if (np.sum(vsel & sel) != np.sum(vsel)):
        #    print('in_cells ', np.sum(vsel), np.sum(sel), np.sum(vsel & sel))
        #print('hvals not in cells ', hvals[vsel & ~sel])
        #assert (np.sum(vsel & sel) != np.sum(vsel)) # selected cells must be pre-defined
        return vals, outvals, outscope

    return in_cells



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
#     cells_node, cells_enode, \
#     cells_nodesize              = clouds_nodes(cells_ene, cells_kid, cells_epath)

#     cells_lgrad, cells_lnode, \
#     cells_lpath                 = clouds_gradient_link(bins, cells, cells_ene,
#                                                        cells_node, cells_kid)

#     cells_epass, cells_ipass    = clouds_passes(cells_ene, cells_node,
#                                                 cells_enode, cells_lnode,
#                                                 cells_kid, cells_lgrad,
#                                                 cells_lpath)

#     cells_track, cells_tnode, \
#     cells_tpass                 = clouds_tracks(cells_node, cells_enode,
#                                                 cells_epass, cells_lpath,
#                                                 cells_kid)

#     cells_ranger, cells_eranger = clouds_rangers(cells_ene,
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

#     dat['ranger']       = cells_ranger        # cell-ID of the most energy cell in the range
#     dat['eranger']      = cells_eranger       # sum-energy of the cells that are associate to this cell-range

#     return pd.DataFrame(dat)


#
#   Clouds main pieces
#-------------------------


def clouds_check(coors, steps, weights):

    ndim  = len(coors)
    nsize = len(coors[0])

    assert len(steps)   == ndim
    for i in range(ndim): assert len(coors[i]) == nsize

    assert len(weights) == nsize

    return ndim, nsize


def clouds_size(coors, weights = None):

    ndim  = len(coors)
    nsize = len(coors[0])

    for i in range(ndim): assert len(coors[i]) == nsize
    if (weights is not None):
        assert len(weights) == nsize

    return ndim, nsize


def clouds_potential(coors, steps, weights):

    ndim         = len(coors)
    bins         = [arstep(x, step, True) for x, step in zip(coors, steps)]
    potential, _ = np.histogramdd(coors, bins, weights = weights)

    sel          = potential > 0
    icells       = to_coors(np.argwhere(sel))
    enes         = potential[icells]
    nsize        = len(enes)
    kids         = np.arange(nsize)

    centers      = [ut_centers(ibin) for ibin in bins]
    cells        = [centers[i][icells[i]] for i in range(ndim)]

    return bins, icells, cells, enes, kids.astype(int)


def clouds_neighbours(bins, cells, cells_ene):

    steps        = [ibin[1] - ibin[0] for ibin in bins]

    ndim, _      = len(cells), len(cells[0])

    counts, _       = np.histogramdd(cells, bins, weights = cells_ene)

    sel = counts > 0
    counts[sel]   = 1

    moves = get_moves(ndim)
    for move in moves:
        coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
        potential_next, _  = np.histogramdd(coors_next, bins, weights = cells_ene)
        isel                = potential_next > 0
        counts[sel & isel] += 1

    nbours = counts[sel].astype(int)

    return nbours


def clouds_gradient(bins, cells, cells_enes, cells_kids):
    """ returns the maximum gradient of energy and the cell which this gradient points to for each cell
    inputs:
        bins : tupe(array), m-dim tuple with the bins in each coordinate
        cells: tuple(array), m-dim tuple with n-size arrays with the cells coordinates
        enes : array,        n-size array with the energy of the cells
        cells_kid  :
    returns:
        cells_grad : array, n-size with the gradient to the link with respect to this cell
        cells_kid  : array, n-size with the id of the gradient cell to this cell
        cells_node : array, n-size with the id of the linked node to this cell
    """

    ndim, nsize  = clouds_size(cells, cells_enes)
    steps        = [ibin[1] - ibin[0] for ibin in bins]

    potential, _ = np.histogramdd(cells, bins, weights = cells_enes)
    kids, _      = np.histogramdd(cells, bins, weights = cells_kids)

    shape        = potential.shape
    sel_cells    = potential > 0

    nn_potential = np.copy(potential)
    nn_kids      = np.copy(kids) .astype(int)

    #moves = get_moves_updown(ndim)
    moves = get_moves(ndim)
    for move in moves:

        coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
        potential_next, _  = np.histogramdd(coors_next, bins, weights = cells_enes)
        kids_next, _       = np.histogramdd(coors_next, bins, weights = cells_kids)

        sel_pot            = potential_next > nn_potential

        sel                = (sel_cells) & (sel_pot)
        if (np.sum(sel) > 0):
            nn_potential[sel]  = potential_next[sel]
            nn_kids     [sel]  = kids_next     [sel]

    vgrad  = nn_potential[sel_cells] - potential[sel_cells]
    vkids  = nn_kids     [sel_cells].astype(int)

    return vgrad, vkids


def clouds_nodes(cells_ene, cells_kid, cells_epath):

    nsize = len(cells_ene)

    # associate each cell to a node
    cells_node  = [get_path(kid, cells_epath)[-1] for kid in cells_kid]
    cells_node  = np.array(cells_node).astype(int)

    # compute the energy of a node and assoticate to the cell enode
    cells_enode    = np.zeros(nsize)
    cells_nodesize = np.zeros(nsize).astype(int)
    nodes = np.unique(cells_node)
    for node in nodes:
        sel   = cells_node == node
        ene   = np.sum(cells_ene[sel])
        size  = np.sum(sel)
        cells_enode[node]    = ene
        cells_nodesize[node] = size

    return cells_node, cells_enode, cells_nodesize


def clouds_gradient_link(bins, cells, cells_enes, cells_nodes, cells_kids):

    ndim, nsize  = len(cells), len(cells[0])

    steps        = [ibin[1] - ibin[0] for ibin in bins]

    nodes, _     = np.histogramdd(cells, bins, weights = cells_nodes)
    potential, _ = np.histogramdd(cells, bins, weights = cells_enes)
    kids, _      = np.histogramdd(cells, bins, weights = cells_kids)

    shape        = potential.shape
    sel_cells    = potential > 0

    nn_potential = np.copy(potential)
    nn_nodes     = np.copy(nodes).astype(int)
    nn_kids      = np.copy(kids) .astype(int)

    #moves = get_moves_updown(ndim)
    moves = get_moves(ndim)

    for move in moves:

        coors_next         = [cells[i] + steps[i] * move[i] for i in range(ndim)]
        potential_next, _  = np.histogramdd(coors_next, bins, weights = cells_enes)
        nodes_next, _      = np.histogramdd(coors_next, bins, weights = cells_nodes)
        kids_next, _       = np.histogramdd(coors_next, bins, weights = cells_kids)

        sel_nodes          = nodes_next != nodes
        sel_pot_next       = potential + potential_next > nn_potential

        sel = (sel_cells) & (sel_nodes) & (sel_pot_next)
        nn_potential[sel] = potential[sel] + potential_next[sel]
        nn_nodes    [sel] = nodes_next    [sel]
        nn_kids     [sel] = kids_next     [sel]

    link_grad  = nn_potential[sel_cells] #- potential[sel_cells]
    link_nodes = nn_nodes    [sel_cells]
    link_kids  = nn_kids     [sel_cells]

    return link_grad, link_nodes, link_kids


def clouds_passes_save(cells_ene, cells_node, cells_enode, cells_lnode,
                       cells_kid, cells_lgrad, cells_lpath):

    nsize  = len(cells_node)
    cells_epass = np.zeros(nsize)
    cells_ipass = np.full(nsize, -1).astype(int)

    nodes_kid, _ = sorted_by_energy(cells_node, cells_enode)
    #print('nodes ', nodes_kid)

    sel_passes = (cells_kid == cells_lpath[cells_lpath[cells_kid]])
    #print('possible passes ', np.sum(sel_passes))

    for i, inode in enumerate(nodes_kid):
        for jnode in nodes_kid[ i +1 :]:
            sel  = np.logical_and(((cells_node == inode) & (cells_lnode == jnode)), sel_passes)
            #print(' passes? ', inode, jnode, np.sum(sel))
            if (np.sum(sel) > 0) :
                isel = np.argmax(cells_lgrad[sel])
                id1  = cells_kid [sel][isel]
                #print('index 1 ', id1)
                cells_epass[id1] = cells_ene[id1] + cells_ene[cells_lpath[id1]]

    return cells_epass, cells_ipass


def clouds_passes(cells_ene, cells_node, cells_enode, cells_lnode,
                  cells_kid, cells_lgrad, cells_lpath):

    nsize  = len(cells_node)
    cells_epass = np.zeros(nsize)
    cells_ipass = np.full(nsize, -1).astype(int)

    nodes_kid, _ = sorted_by_energy(cells_node, cells_enode)
    #print('nodes ', nodes_kid)

    sel_passes = (cells_kid == cells_lpath[cells_lpath[cells_kid]])
    #print('possible passes ', np.sum(sel_passes))

    for i, inode in enumerate(nodes_kid):
        for jnode in nodes_kid[ i +1 : ]:
            sel  = np.logical_and(((cells_node == inode) & (cells_lnode == jnode)), sel_passes)
            if (np.sum(sel) == 0):
                sel = np.logical_and((cells_node == inode), (cells_lnode == jnode))
            #print(' passes? ', inode, jnode, np.sum(sel))
            if (np.sum(sel) > 0) :
                isel = np.argmax(cells_lgrad[sel])
                id1  = cells_kid [sel][isel]
                #print('index 1 ', id1)
                cells_epass[id1] = cells_ene[id1] + cells_ene[cells_lpath[id1]]

    return cells_epass, cells_ipass



def clouds_tracks(cnode, enodes, epasses, lpaths, kids):

    sel         = enodes > 0
    knodes, _   = sorted_by_energy(kids[sel], enodes[sel])
    sel         = epasses  > 0
    kpasses,  _ = sorted_by_energy(kids[sel], epasses[sel])
    #print('passes', kpasses, _)

    kstaples = [tuple((cnode[kid], cnode[lpaths[kid]], kid)) for kid in kpasses]

    def valid_pass_(staple, nodes_in):
        nodes_ = np.array((staple[0], staple[1])).astype(int)
        sel    = np.isin(nodes_, nodes_in)
        return (np.sum(sel) == 1)


    def new_track(xnodes, xstaples):
        nodes      = list(xnodes[1:] )
        nodes_in   = [xnodes[0],]
        staples    = list(xstaples)
        staples_in = []
        nstaples   = len(staples) +1
        while (len(staples) < nstaples):
            nstaples = len(staples)
            sels = [valid_pass_(staple, nodes_in) for staple in staples]
            if (np.sum(sels) > 0):
                staple = np.array(staples)[sels][0]
                staples_in.append(tuple(staple))
                ii    = sels.index(True)
                staples.pop(ii)
                inode1, inode2 = staple[0], staple[1]
                for inode in [inode1, inode2]:
                    if inode not in nodes_in:
                        nodes_in.append(inode)
                        nodes.remove(inode)
        return nodes_in, staples_in, nodes, staples

    nsize       = len(cnode)
    main_nodes  = []
    tracks  = np.full(nsize, -1).astype(int)
    tnodes  = np.full(nsize, -1).astype(int)
    tpasses = np.full(nsize, -1).astype(int)
    while (len(knodes) > 0):
        track_nodes, track_staples, knodes, kstaples = new_track(knodes, kstaples)
        #print('track nodes   ! ', track_nodes)
        #print('track staples   ', track_staples)
        main_node = track_nodes[0]
        for inode in track_nodes:
            tracks[cnode == inode]      = main_node
            tnodes[inode]               = main_node
        for staple in track_staples:
            kid = staple[2]
            tpasses[kid]                = main_node
             #print('tpass ', kid)

    return tracks, tnodes, tpasses


def clouds_crests(enes, tnode, tpass, epath, lpath, ckids):

    nsize  = len(tpass)
    trange = np.full(nsize, -1).astype(int)

    trange[ckids[tnode > -1]] = ckids[tnode > -1]

    tracks  = np.unique(tpass[tpass > -1])
    for track in tracks:
        #print('track ', track)
        kids  = ckids[tpass  == track]
        #print('passes ', kids)
        paths = [get_pass_path(kid, epath, lpath) for kid in kids]
        #print('paths ', paths)
        path  = paths[0]
        for ipath in paths[0:]: path += ipath
        path  = np.unique(path)
        #print('path ', path)
        trange[path] = track


    erange = np.zeros(nsize)
    tkids = ckids[trange > -1]
    #print(tkids)
    def _irange(kid):
        if kid in tkids: return kid
        kid  = epath[kid]
        return _irange(kid)

    for kid in ckids:
        erange[_irange(kid)] += enes[kid]


    return trange, erange

#----------------------
# MC Clouds
#----------------------


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


def get_mcpaths(xcoors, enemc, in_cells):

    def dis(i):
        if (i == 0): return 0.
        dd2 = np.sqrt(np.sum([(xcoor[i] - xcoor[i-1]) * (xcoor[i] - xcoor[i-1]) for xcoor in xcoors]))
        return dd2

    ndim, nsize  = len(xcoors), len(xcoors[0])
    dds = [dis(i) for i in range(nsize)]
    def locate_in_cell(xcoor, val):
        vals, _, _ = in_cells(xcoor, val)
        return np.argwhere(vals > 0)

    xpos = [locate_in_cell([[xcoor[i],] for xcoor in xcoors], [enemc[i],]) for i in range(nsize)]

    paths   = []
    path    = []
    prevkid = -1
    for i, dd in enumerate(dds):
        ckid = int(xpos[i][0]) if len(xpos[i]) == 1 else -1
        if (ckid != prevkid): # to a different cells
            if (ckid != -1):
                if (dd > 2.):
                    if (len(path) > 0): paths.append(path)
                    path = [ckid]
                else:
                    path.append(ckid)
            #else:
            #    print('empty!', i, ckid, prevkid)
        prevkid = ckid
    paths.append(path)
    #print(paths)
    return paths

# SAVE THIS CODE
#
# ncells  = len(dfclouds.ene)
# print(ncells)
# mcinit  = np.full(ncells, -1)
# mcpath  = np.full(ncells, -1)
# mcpath[mcene > 0] = dfclouds.kid[mcene > 0]
# mccells = [-1,]
# paths = []
# path  = []
# prevkid = -1
# lastkid = -1
# for i, dd in enumerate(dds):
#     ckid = int(xpos[i][0]) if len(xpos[i]) == 1 else -1
#     if (ckid != prevkid): # to a different cells
#         if (ckid not in mccells):
#             mccells.append(ckid)
#             if (dd > 2.):
#                 # new init
#                 if (len(path) > 0): paths.append(path)
#                 path = [ckid]
#                 mcinit[ckid] = ckid
#                 lastkid      = ckid
#             else:
#                 # continue
#                 if (len(path) == 0):
#                     mcinit[ckid] = ckid
#                 path.append(ckid)
#                 if (lastkid != -1): mcpath[lastkid] = ckid
#                 lastkid = ckid
#         else:
#             # new empty
#             if (len(path) > 0): paths.append(path)
#             path = []
#             lastkid = -1
#     prevkid = ckid
# paths.append(path)
# print(mccells)
# for path in paths:
#     print('path', path[0], path[-1], ', nodes: ', path)
# npaths = len(paths)
# #for i in range(npaths):
# #    for j in range(i +1, npaths):
# #        print(np.sum(np.isin(paths[j], paths[i])))
#
# #print(mcpath)
# for path in paths:
#     print('init ', path[0], np.isin(path[0], mcinit[mcinit > -1]))
# ipaths = [path[0] for path in paths]
# print('inits ', sorted(mcinit[mcinit > -1]), 'len ', len(mcinit[mcinit > -1]))
# print('inits ', sorted(ipaths), 'len ', len(ipaths))
#
#
# #for path in paths:
# #    print('init ', path[-1],  mcpath[path[-1]] == path[-1]))
# lpaths = [path[-1] for path in paths]
# for path in paths:
#     print('end ', path[-1], np.isin(path[-1], mcpath[mcpath == dfclouds.kid]))
# print('ends ', sorted(mcpath[mcpath == dfclouds.kid]), 'len ', len(mcpath[mcpath == dfclouds.kid]))
# print('ends ', sorted(lpaths), 'len ', len(lpaths))
#

#HERE



#
#   Post-cloud utils
#-------------------------------


def sorted_by_energy(kids, enes):
    """ return the ids ordered by energy
    inputs:
        kids : array(int), array wih the IDs
        enes : array(float), array with the energies
    returs:
        xkids: array(int), array with the IDsordeed by energy (decreasing)
        enes : array(float), array with the energies in decreasing order
    """
    sel = enes > 0.
    nodes = sorted(zip(enes[sel], kids[sel]))
    nodes.reverse()
    xenes, xkids = [node[0] for node in nodes], [node[1] for node in nodes]
    return xkids, xenes



def get_path(kid, next_kid):
    """ given a kid, a local ID and the link array next_kid return the path of
    consecutive kids
    inputs:
        kid: int, local ID of the array to get the path
        next_kid: array, n-size dimension with the associated local ID cells to a give cell ID
    """
    path = []
    while next_kid[kid] != kid:
        path.append(kid)
        kid = next_kid[kid]
    path.append(kid)
    return path


def get_pass_path(kid, epath, lpath):
    """ return the path (list of cells IDs) given an ID of a pass-cell
    inputs:
        kid   : int, ID of the pass-cell
        epath : array(int), array with the IDs of the gradients
        lpath : array(int), array with the IDs of the links
    returns:
        path  : array(int), array with the IDs of the cells in the path of this pass
    """
    path1 = get_path(kid       , epath)
    path2 = get_path(lpath[kid], epath)
    path1.reverse()
    path = path1 + path2
    return path


def get_segment(cells, kids):
    """ Fron a list of local IDs returns a segment to plot
    inputs:
        cells: tuple(array), m-dim tuple with n-size array with the cells' cordinates positions
        kids: tuple(int), list of the ID to generate the segment
    """
    ndim = len(cells)
    segment = [np.array([float(cells[i][kid]) for kid in kids]) for i in range(ndim)]
    return segment


#--- Function with passes


def get_passes(epass, node, lnode):
    """
    
    return the list of pair (node0, node1) with the passes between nodes

    Parameters
    ----------
    epass : np.array, potential of the pass for this cell
    node  : np.array(int), k-index of the node for this cell
    lnode : np.array(int), k-index of the lined node for this cell

    Returns
    -------
    passes : list( (int, int) ), list of passes, pairs of (node0, node1)
    """
    
    ksel   = epass > 0.
    passes = list(zip(node[ksel], lnode[ksel]))
    
    #passes = list(zip(node[ksel], lnode[ksel]))
    #unodes = list(node[ksel]) + list(lnode[ksel])
    #knodes = np.unique(unodes)
    
    #dpasses = {}
    #for knode in knodes:
    #    dpasses[knode] = [ipass for ipass in passes if np.isin(knode, ipass)]
    
    return passes

def get_passes_dict(passes):
    """
    
    from the list of passes (node0, node1) returns a dictionary that for each node key
    returns the passes of that node

    Parameters
    ----------
    passes : list( (int, int)), list of passes (node0, node1)
        DESCRIPTION.

    Returns
    -------
    dpasses : dict(int) = list( (node0, node1)), dictionary, for each node-id returns the list of passes

    """
    
    knodes = np.unique(np.concatenate(list(passes)))
    dpasses = {}
    for knode in knodes:
        dpasses[knode] = [ipass for ipass in passes if np.isin(knode, ipass)]
        
    return dpasses
    

def nodes_idistance(passes):
    """
    
    From the list of passes compute the step distance of each node to an extreme of the track
    
    Parameters
    ----------
    passes : list( (int, int)), list of passes, each pass is a tuple (node0, node1)

    Returns
    -------
    udist : dic(int) = int, returns the number of node steps of the current node to an extreme

    """
    
    
    passes  = list(passes)
    dpasses = get_passes_dict(passes)
    knodes  = dpasses.keys()

    end_kids = np.array([k for k in knodes if (len(dpasses[k]) == 1)])
    #print(end_kids)

    udist = {}
    i = 1
    for kid in end_kids:
        udist[kid] = 1
    ok = True
    while ok > 0:
        i = i +1
        upasses = [pair for pair in passes if np.sum(np.isin(pair, end_kids)) > 0]
        uus = np.unique(np.concatenate(upasses))
        end_kids = uus[~np.isin(uus, end_kids)]
        for kid in end_kids:
            udist[kid] = i
        #print(i, end_kids)
        for upass in upasses:
            passes.pop(passes.index(upass))
        ok = (len(passes) > 0) and (len(end_kids) >0)
        
    return udist


def get_function_branches(passes):
    """
    
    returns a function, branches, that branches(int) where int is the id of the node
    returns the list of branches starting from this node, each branch is a list of consecutive nodes

    Parameters
    ----------
    passes : tuple( (int, int)), list of passes, each pass is (node0, node1)

    Returns
    -------
    branches; callable, branches(int) return the list of branches starting in that node

    """
    
    dpasses = get_passes_dict(passes)
    
    def grow_branch(path):
        ik = path[-1]
        if (ik not in dpasses.keys()):
            return []
        kpasses = dpasses[ik]
        paths   = []
        for i, ipass in enumerate(kpasses):
            ipass = np.array(ipass, int)
            ik0, ik1 = ipass
            ksel = np.isin(ipass, path)
            #print(ipass, path, ksel)
            ipath = list(path)
            if (np.sum(ksel) == 2): continue
            ikn  = ipass[~ksel][0] 
            ipath = list(path)
            ipath.append(ikn)
            paths.append(ipath)        
        return paths

    def get_branches(kid):    
        branches = []
        paths = [[kid],]
        while len(paths) > 0:
            npaths = [] 
            for path in paths:
                nipaths = grow_branch(path)
                if (len(nipaths)  == 0): 
                    branches.append(list(path))
                    #print('branch ', path)
                else:
                    npaths += nipaths
                    #print('paths ', nipaths)
                paths = npaths
        return branches
    
    return get_branches