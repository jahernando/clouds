import numpy             as np
import pandas            as pd
#import tables            as tb


#import hipy.utils  as ut

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
        hvals, _     = np.histogramdd(xcoors, bins, weights = values)
        vals         = hvals[icells]
        return vals

    return in_cells


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

    bins, cells, cells_ene, \
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

    cells_ranger, cells_eranger = clouds_rangers(cells_ene,
                                               cells_tnode, cells_tpass,
                                               cells_epath, cells_lpath,
                                               cells_kid)

    dat = {}
    for i in range(ndim):
        dat['x'+str(i)] = cells[i]            # positions of the cells
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

    dat['ranger']       = cells_ranger        # cell-ID of the most energy cell in the range
    dat['eranger']      = cells_eranger       # sum-energy of the cells that are associate to this cell-range

    return pd.DataFrame(dat)


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

    return bins, cells, enes, kids.astype(int)


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
        for jnode in nodes_kid[ i +1 :]:
            sel  = np.logical_and(((cells_node == inode) & (cells_lnode == jnode)), sel_passes)
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


def clouds_rangers(enes, tnode, tpass, epath, lpath, ckids):

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

#
#
# def clouds_tracks(cnode, enodes, epasses, lpaths, kids):
#
#     sel         = enodes > 0
#     knodes, _   = sorted_by_energy(kids[sel], enodes[sel])
#     sel         = epasses  > 0
#     kpasses,  _ = sorted_by_energy(kids[sel], epasses[sel])
#
#     kstaples = [tuple((cnode[kid], cnode[lpaths[kid]])) for kid in kpasses]
#
#     def valid_pass_(staple, nodes_in):
#         nodes_ = np.array((staple[0], staple[1])).astype(int)
#         sel    = np.isin(nodes_, nodes_in)
#         return (np.sum(sel) == 1)
#
#
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
#
#     def make_tracks(xnodes, xstaples, cells_node):
#         nsize       = len(cells_node)
#         main_nodes  = []
#         cells_track = np.full(nsize, -1).istype(int)
#         cells_tnode = np.full(nsize, -1).istype(int)
#         cells_tpass = np.full(nsize, -1).istype(int)
#         while (len(xnodes) > 0):
#             track_nodes, track_staples, xnodes, xstaples = new_track(xnodes, xstaples)
#             main_node = track_nodes[0]
#             main_nodes.append(main_node)
#             for inode in track_nodes:
#                 cells_track[cells_node == inode] = main_node
#         return cells_track, main_nodes
#
#     tracks, mnodes = make_tracks(knodes, kstaples, cnode)
#
#     return tracks, mnodes
#
# #
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
    path1 = get_path(kid             , epath)
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


#---- OLD code
#
# def potential(coors, steps, weights = None):
#     """ compute the clouds potential
#     inputs:
#         coors: tuple(arrays), a m-dim size list with the n-size coordinates
#         steps: tuple(float), a m-dim size with the size in each of the m-dimensions
#         weights: array, a n-size value of the weights
#     returns:
#         energy: array with the energy of the voxels in the m-dim space
#         bins  : m-dim size list with the bins (edges) of the space for each coordinate
#     """
#
#     ndim  = len(coors)
#     nsize = len(coors[0])
#     weights = weights if weights is not None else np.ones(nsize)
#
#     assert len(steps)   == ndim
#     for i in range(ndim): assert len(coors[i]) == nsize
#     assert len(weights) == nsize
#
#     bins         = [arstep(x, step, True) for x, step in zip(coors, steps)]
#     bins_centers = [ut.centers(ibin) for ibin in bins]
#
#     #icoors   = [np.digitize(x, xbins) -1 for x, xbins in zip(coors, bins)]
#     icoors       = [np.digitize(coors[i], bins[i]) - 1 for i in range(ndim)]
#     #idcoors  = get_ids(icoors)
#     #idpoints = get_ids(ipoints)
#
#     #icoors       = [np.digitize(coors[i], bins[i]) - 1 for i in range(ndim)]
#
#     pot, edges = np.histogramdd(coors, bins, weights = weights)
#
#     return pot, edges
#
#
# def voxels(potential, bins):
#     """ from the potential space and the bins returns the voxels with its potential
#     returns:
#         xcells: tuple(array), a m-dim list of n-size arrays with the coordinats of the voxels
#         potentials: array a n-size array with the potential of the voxels
#     """
#
#     centers      = [ut.centers(ibin) for ibin in bins]
#     sel          = potential > 0
#     cells        = to_coors(np.argwhere(sel))
#     ndim, nsize  = len(cells), len(cells[0])
#     xcells       = [centers[i][cells[i]] for i in range(ndim)]
#     weights      = potential[cells]
#
#     return xcells, weights
#
#
# def neighbours(potential, bins):
#     """ returns the number of neighbours with potential
#     returns:
#         xcells: tuple(array), a m-dim list of n-size arrays with the coordinats of the voxels
#         counts: array a n-size array with the number of neighbourgs
#     """
#
#     shape        = potential.shape
#     steps        = [ibin[1] - ibin[0] for ibin in bins]
#     centers      = [ut.centers(ibin) for ibin in bins]
#
#     sel          = potential > 0
#     cells        = to_coors(np.argwhere(sel))
#     ndim, nsize  = len(cells), len(cells[0])
#     xcells       = [centers[i][cells[i]] for i in range(ndim)]
#     weights      = potential[cells]
#
#     counts        = np.full(shape, 0)
#     counts[sel]   = 1
#
#     #moves = get_moves_updown(ndim)
#     moves = get_moves(ndim)
#
#     for move in moves:
#         coors_next         = [xcells[i] + steps[i] * move[i] for i in range(ndim)]
#         potential_next, _  = np.histogramdd(coors_next, bins, weights = weights)
#
#         isel                = potential_next > 0
#         counts[sel & isel] += 1
#
#     return xcells, counts[cells]
#
#
# def gradient(potential, bins):
#     """ returns the grandient potential within neighbourgs
#     returns:
#         xcells: tuple(array), a m-dim list of n-size arrays with the coordinats of the voxels
#         deltas: array a n-size array with the increase of the potential
#         dirs  : a m-dim size list with the n-size coordiates of the gradent direction
#     """
#
#
#     shape        = potential.shape
#     steps        = [ibin[1] - ibin[0] for ibin in bins]
#     centers      = [ut.centers(ibin) for ibin in bins]
#
#     sel          = potential > 0
#     cells        = to_coors(np.argwhere(sel))
#     ndim, nsize  = len(cells), len(cells[0])
#     xcells       = [centers[i][cells[i]] for i in range(ndim)]
#     weights      = potential[cells]
#
#     nn_potential   = np.copy(potential)
#     nn_ids         = np.full((*shape, ndim), 0)
#
#     #moves = get_moves_updown(ndim)
#     moves = get_moves(ndim)
#
#     for move in moves:
#         coors_next         = [xcells[i] + steps[i] * move[i] for i in range(ndim)]
#         potential_next, _  = np.histogramdd(coors_next, bins, weights = weights)
#
#
#         isel                     = potential_next > nn_potential
#         nn_potential[sel & isel] = potential_next[sel & isel]
#
#         if (np.sum(sel & isel) > 0):
#             nn_ids[sel & isel]    = -1 * np.array(steps) * move
#
#
#     deltas  = nn_potential[cells] - potential[cells]
#     dirs    = to_coors(nn_ids[cells])
#
#     return xcells, deltas, dirs
#
# def paths(cells, bins, steps, dirs):
#     """ from the gradiend directions (dirs) compute the paths for each voxel
#     to its node:
#     returns:
#         node : array, n-size array with the index of the node in the list of cells
#         ipath: array, n-size array with the index of next voxel in the path to its node
#         paths: list(list), list of indices of the voxels in the path to its node.
#                # TODO, we want to return this?
#     """
#
#     ndim, nsize = len(cells), len(cells[0])
#     print('dimensions ', ndim, 'size ', nsize)
#     icells = to_indices(cells, bins)
#
#     idcells    = to_ids(icells)
#     nn_ipath   = np.arange(nsize) # to_ids(icells)
#     nn_inode   = np.arange(nsize)
#
#     ipos  = to_vectors(icells)
#     idirs = (to_vectors(dirs)/np.array(steps)).astype(int)
#
#     vnull = np.zeros(ndim)
#
#     def _path(i, ipath):
#
#         ipath.append(i)
#
#         if (np.all(idirs[i] == vnull)):
#             return ipath
#         #print(True, 'pos i ', i, 'icoors', ipos[i], 'id', idcells[i], 'dir ', idirs[i],
#         #  'idnext', nn_kcell[i])
#
#         iloc  =  ipos[i] + idirs[i]
#         #idloc = to_ids(iloc)
#         idloc =  to_ids([(iloc[i],) for i in range(ndim)])[0]
#
#         isel = np.isin(idcells, idloc)
#         ii = int(np.argwhere(isel))
#         nn_ipath[i] = ii
#
#         return _path(ii, ipath)
#
#     paths = []
#     for i in range(nsize):
#         ipath       = _path(i, [])
#         nn_inode[i] = ipath[-1]
#         paths.append(ipath)
#
#     return nn_inode, nn_ipath, paths
#
#
# def energy_nodes(ene, nnode):
#     """ returns the energy of the nodes, from the ene, energy of the voxels, and nnode,
#     index of the node of the voxel
#     returns:
#         nenode: array, n-size array with the sum of the energy of the voxels in the node
#                        for the nodes, for the rest of the voxels is zero.
#     """
#     nsize = len(nnode)
#     enodes = np.zeros(nsize)
#     ks = np.unique(nnode)
#     for ki in ks:
#         sel = nnode == ki
#         enodes[ki] = np.sum(ene[sel])
#     return enodes
#
#
# def nodes_order(cells_enode, cells_node, cells_kid):
#     """ order the nodes by energy
#     inputs:
#         cells_enode: array n-size with the energy of the node in the cell node
#                                   (the other cells have zero)
#         cells_node : array n-size with the ID of the node-cell
#         cells_kid  : array n-size with the ID of the cell
#     returns:
#         nodes_ene   : array m-size m is the number of nodes with the energy of the nodes
#         nodes_kid   : array m-size with the ID of the node-cell
#         nodes_ncells: array m-size with the number of cells associated to the node
#     """
#
#     sel = cells_enode > 0.
#     #voxel_id  = np.arange(nsize)
#     nodes_kid  = cells_kid[sel]
#     nodes_ene  = cells_enode[sel]
#
#     nodes = sorted(zip(nodes_ene, nodes_kid))
#     nodes.reverse()
#     nnodes        = len(nodes)
#     nodes_ene     = [node[0] for node in nodes]
#     nodes_kid     = [node[1] for node in nodes]
#     nodes_ncells  = [np.sum(cells_node == node_id) for node_id in nodes_kid]
#
#     return nodes_ene, nodes_kid, nodes_ncells
#
#
#
# def staples(xbins, xcells, xcells_enes, xcells_nodes, xcells_kids):
#     """ returns the cells that are links between a second set of cells, cell_test
#     inputs:
#         bins : tupe(array), m-dim tuple with the bins in each coordinate
#         cells: tuple(array), m-dim tuple with n-size arrays with the cells coordinates
#         enes : array,        n-size array with the energy of the cells
#         cells_node :
#         cells_kid  :
#     returns:
#         cells_link_grad : array, n-size with the gradient to the link with respect to this cell
#         cells_link_kid  : array, n-size with the id of the gradient cell to this cell
#         cells_link_node : array, n-size with the id of the linked node to this cell
#     """
#
#     ndim, nsize  = len(xcells), len(xcells[0])
#
#     steps        = [ibin[1] - ibin[0] for ibin in xbins]
#
#     nodes, _     = np.histogramdd(xcells, xbins, weights = xcells_nodes)
#     potential, _ = np.histogramdd(xcells, xbins, weights = xcells_enes)
#     kids, _      = np.histogramdd(xcells, xbins, weights = xcells_kids)
#
#     shape        = potential.shape
#     sel_cells    = potential > 0
#
#     nn_potential = np.copy(potential)
#     nn_nodes     = np.copy(nodes).astype(int)
#     nn_kids      = np.copy(kids) .astype(int)
#
#     #moves = get_moves_updown(ndim)
#     moves = get_moves(ndim)
#
#     for move in moves:
#
#         coors_next         = [xcells[i] + steps[i] * move[i] for i in range(ndim)]
#         potential_next, _  = np.histogramdd(coors_next, xbins, weights = xcells_enes)
#         nodes_next, _      = np.histogramdd(coors_next, xbins, weights = xcells_nodes)
#         kids_next, _       = np.histogramdd(coors_next, xbins, weights = xcells_kids)
#
#         sel_nodes          = nodes_next != nodes
#         sel_pot_next       = potential + potential_next > nn_potential
#
#         sel = (sel_cells) & (sel_nodes) & (sel_pot_next)
#         nn_potential[sel] = potential[sel] + potential_next[sel]
#         nn_nodes    [sel] = nodes_next    [sel]
#         nn_kids     [sel] = kids_next     [sel]
#
#     link_grad  = nn_potential[sel_cells] #- potential[sel_cells]
#     link_nodes = nn_nodes    [sel_cells]
#     link_kids  = nn_kids     [sel_cells]
#
#     return link_grad, link_nodes, link_kids
#
# def set_staples(cells_node, cells_lnode, cells_kid, cells_lkid, cells_lgrad):
#
#     nsize = len(cells_node)
#     cells_staples = np.full(nsize, False)
#     nodes_kid = list(set(cells_node))
#     for i, inode in enumerate(nodes_kid):
#         for jnode in nodes_kid[ i +1 :]:
#             sel  = (cells_node == inode) & (cells_lnode == jnode)
#             if (np.sum(sel) > 0) :
#                 isel = np.argmax(cells_lgrad[sel])
#                 id1  = cells_kid [sel][isel]
#                 #id2  = cells_lkids[sel][isel]
#                 cells_staples[id1] = True
#
#     staples = zip(cells_kid[cells_staples], cells_lkid[cells_staples])
#     staples = sorted(staples)
#
#     return staples, cells_staples
#
#

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
#
#
# def get_staple_path(staple, next_kid):
#     path1 = get_path(staple[0], next_kid)
#     path2 = get_path(staple[1], next_kid)
#     path1.reverse()
#     path = path1 + path2
#     return path
#
#
# def get_segment(cells, kids):
#     """ Fron a list of local IDs returns a segment to plot
#     inputs:
#         cells: tuple(array), m-dim tuple with n-size array with the cells' cordinates positions
#         kids: tuple(int), list of the ID to generate the segment
#     """
#     ndim = len(cells)
#     segment = [[cells[i][kid] for kid in kids] for i in range(ndim)]
#     return segment


# def nodes_staples(bins, cells, enes, cells_test, enes_test):
#     """ returns the cells that are links between a second set of cells, cell_test
#     inputs:
#         bins : tupe(array), m-dim tuple with the bins in each coordinate
#         cells: tuple(array), m-dim tuple with n-size arrays with the cells coordinates
#         enes : array,        n-size array with the energy of the cells
#         cells_test: tuple(array), m-dim tuple with n'-size arrays with the second set of cells coordinates
#         ene_test  : tuple(array), n'-size array with the energy of the second set of cells
#     returns:
#         deltas: array, n-size with the sum energy of the both linked neighbour cell
#         dirs  : tuple(array), m-dim tule with the direction of the linked cell
#     """
#
#     ndim, nsize  = len(cells), len(cells[0])
#
#     steps        = [ibin[1] - ibin[0] for ibin in bins]
#     centers      = [ut.centers(ibin) for ibin in bins]
#
#     potential, _ = np.histogramdd(cells, bins, weights = enes)
#     shape        = potential.shape
#
#     sel          = potential > 0
#     shape        = potential.shape
#
#     nn_potential   = np.copy(potential)
#     nn_dirs        = np.full((*shape, ndim), 0)
#
#     #moves = get_moves_updown(ndim)
#     moves = get_moves(ndim)
#
#     for move in moves:
#
#         coors_next         = [cells_test[i] + steps[i] * move[i] for i in range(ndim)]
#         potential_next, _  = np.histogramdd(coors_next, bins, weights = enes_test)
#
#         isel                     = potential + potential_next > nn_potential
#         nn_potential[sel & isel] = potential[sel & isel] + potential_next[sel & isel]
#
#         if (np.sum(sel & isel) > 0):
#             nn_dirs[sel & isel] = -1 * np.array(steps) * move
#
#
#     deltas  = nn_potential[sel] - potential[sel]
#     dirs    = to_coors(nn_dirs[sel])
#
#     return deltas, dirs
#
#
#
# def nodes_links_(bins, cells, enes, cells_test, enes_test):
#     """ returns the cells that are links between a second set of cells, cell_test
#     inputs:
#         bins : tupe(array), m-dim tuple with the bins in each coordinate
#         cells: tuple(array), m-dim tuple with n-size arrays with the cells coordinates
#         enes : array,        n-size array with the energy of the cells
#         cells_test: tuple(array), m-dim tuple with n'-size arrays with the second set of cells coordinates
#         ene_test  : tuple(array), n'-size array with the energy of the second set of cells
#     returns:
#         deltas: array, n-size with the sum energy of the both linked neighbour cell
#         dirs  : tuple(array), m-dim tule with the direction of the linked cell
#     """
#
#     ndim, nsize  = len(cells), len(cells[0])
#
#     steps        = [ibin[1] - ibin[0] for ibin in bins]
#     centers      = [ut.centers(ibin) for ibin in bins]
#
#     potential, _ = np.histogramdd(cells, bins, weights = enes)
#     shape        = potential.shape
#
#     sel          = potential > 0
#     shape        = potential.shape
#
#     nn_potential   = np.copy(potential)
#     nn_dirs        = np.full((*shape, ndim), 0)
#
#     #moves = get_moves_updown(ndim)
#     moves = get_moves(ndim)
#
#     for move in moves:
#
#         coors_next         = [cells_test[i] + steps[i] * move[i] for i in range(ndim)]
#         potential_next, _  = np.histogramdd(coors_next, bins, weights = enes_test)
#
#         isel                     = potential + potential_next > nn_potential
#         nn_potential[sel & isel] = potential[sel & isel] + potential_next[sel & isel]
#
#         if (np.sum(sel & isel) > 0):
#             nn_dirs[sel & isel] = -1 * np.array(steps) * move
#
#
#     deltas  = nn_potential[sel] - potential[sel]
#     dirs    = to_coors(nn_dirs[sel])
#
#     return deltas, dirs
#
#
# def nodes_links(nodes_kid, bins, cells, cells_ene, cells_node, cells_kid, cells_hid):
#     """ return the staples/links between nodes
#     inputs:
#         nodes_kid: array, k-size with the nodes cells ID
#         cells    : tuple(array), m-dim tuple with n-size arrays with the cells coordinates
#         cells_ene: array, n-size array with the energy of the cells
#         bins     : tuple(array), m-dim tuple with arrays with the bing edges for each coordinate
#         cells_node : array, n-size array with the ID of the node which this cell is associated to
#         cells_kid  : array, n-size array with the ID of the cell
#         cells_hid  : array, n-size array with the global ID of the cell
#     returns:
#         staples_nodes: tuple( (int, int)), pairs of linked ID nodes
#         staples_kids : tuple( (int, int)), paris of linked ID cells
#     """
#
#     def _csel(vals, sel):
#         return [val[sel] for val in vals]
#
#     def _node(sel):
#         cells1 = _csel(cells, sel)
#         enes1  = cells_ene[sel]
#         return cells1, enes1
#
#
#     nnodes         = len(nodes_kid)
#     staples_nodes  = []
#     staples_kids   = []
#     #staples_lenghs = []
#     for inode, node1_kid in enumerate(nodes_kid):
#         sel1    = cells_node == node1_kid
#         cells1  = [cell[sel1] for cell in cells]
#         for node2_kid in (nodes_kid[ inode + 1 : ]):
#             sel2 = cells_node == node2_kid
#             xdeltas, xdirs = nodes_links_(bins, *_node(sel1), *_node(sel2))
#
#             if (np.sum(xdeltas > 0.) <= 0): continue
#
#             i1    = np.argmax(xdeltas)
#             kid1  = cells_kid[sel1][i1]
#
#             loc = [(cell[i1] + xdir[i1],) for cell, xdir in zip(cells1, xdirs)]
#             hid2 = to_ids(to_indices(loc, bins))
#
#             isel = np.isin(cells_hid, hid2)
#             kid2 = cells_kid[isel][0]
#
#             staple_nodes = (node1_kid, node2_kid)
#             staple_kids  = (kid1, kid2)
#
#             staples_nodes.append(staple_nodes)
#             staples_kids .append(staple_kids)
#
#     return staples_nodes, staples_kids
