import numpy             as np
import pandas            as pd

import matplotlib.pyplot as plt

import clouds        as clouds


#---- utils

def karg(name, value, kargs):
    """ if a parameter is not in the key-words dictiory then its include with value
    inputs:
        name: str, the name of the parameter
        value: -, the value of the parameter
        kargs: dict{str:-}, key-words dictionary
    returns:
        kargs: returns the updated (if so) key-words dictionary
    """
    kargs[name] = value if name not in kargs.keys() else kargs[name]
    return kargs


def canvas(ns : int, ny : int = 2, height : float = 5., width : float = 6.) -> callable:
    """ create a canvas with ns subplots and ny-columns,
    return a function to move to next subplot in the canvas
    """
    nx  = int(ns / ny + ns % ny)
    plt.figure(figsize = (width * ny, height * nx))
    def subplot(iplot, dim = '2d'):
        """ controls the subplots in a canvas
            inputs:
                iplot: int, index of the plot in the canvas
                dim  : str, '3d'  in the case the plot is 3d
            returns:
                nx, ny: int, int (the nx, ny rows and columns of the canvas)
        """
        assert iplot <= nx * ny
        plt.subplot(nx, ny, iplot)
        if (dim == '3d'):
            nn = nx * 100 +ny *10 + iplot
            plt.gcf().add_subplot(nn, projection = dim)
        return plt.gca()
    return subplot


def get_cells(df, ndim):
    return [df['x'+str(i)].values for i in range(ndim)]


def _ocells(cells, i = 0):
    return cells[i:] + cells[:i]


def _csel(vals, sel):
    return [val[sel] for val in vals]


#
# Low level clouds plotting elements
#

def dcloud_cells(cells, enes = None, xaxis = 0, **kargs):
    """ Draw cells, if enes, with its energy
    """

    ndim, nsize = len(cells), len(cells[0])

    enes = np.ones(nsize) if enes is None else enes

    kargs = karg('marker',   's', kargs)
    kargs = karg('c' ,  enes, kargs)
    #kargs = karg('s' ,  10 * enes, kargs)

    ax = plt.gca()
    xcells = _ocells(cells, xaxis)
    ax.scatter(*xcells, **kargs)
    return
    #if (chamber): draw_chamber(coors, ax)


def dcloud_nodes(cells, enodes, **kargs):
    """ Draw cells that enodes > 0
    """

    kargs = karg('marker',   '*', kargs)

    sel = enodes > 0

    kargs = karg('s', enodes[sel], kargs)

    dcloud_cells(_csel(cells, sel), enodes[sel], **kargs)
    return


def dcloud_grad(cells, epath, xaxis = 0, **kargs):
    """ Draw the gradient of the cells
    """

    ndim = len(cells)
    ncells = _csel(cells, epath)
    coors  = _ocells(cells , xaxis) if xaxis != 0 else cells
    vcoors = _ocells(ncells, xaxis) if xaxis != 0 else ncells

    xdirs =[vcoor - coor for vcoor, coor in zip(vcoors, coors)]
    opts = {'scale_units': 'xy', 'scale' : 2.} if ndim == 2 else {'length' : 0.4}

    plt.quiver(*coors, *xdirs, **opts, **kargs)


def dcloud_segments(cells, kids, epath, lpath, xaxis = 0, **kargs):
    """ Draw the segments associated to IDs kids (they should be a passes)
    """

    xcells   = _ocells(cells, xaxis) if xaxis != 0 else cells
    segments = [clouds.get_segment(xcells, clouds.get_pass_path(kid, epath, lpath)) for kid in kids]
    for segment in segments:
        #print(segment)
        kargs = karg('c', 'black', kargs)
        plt.plot(*segment, **kargs)


#
#  High Level
#

def dcloud_steps(dfclouds, ndim, scale = 1000., rscale = 5., xaxis = 0, ncolumns = 1):
    """ Draw every step of clouds: i) gradients, ii) nodes, ii) links kk) paths
    """

    cells  = get_cells(dfclouds, ndim)
    enes   = dfclouds.ene.values
    enodes = dfclouds.enode.values
    nodes  = dfclouds.node.values
    epath  = dfclouds.epath.values
    lpath  = dfclouds.lpath.values
    epass  = dfclouds.epass.values


    sdim = '3d' if ndim == 3 else '2d'
    subplot = canvas(6, ncolumns, 10, 12)

    subplot(1, sdim) # cloud
    dcloud_cells(cells, scale * enes, alpha = 0.3, xaxis = xaxis);
    #dcloud_grad(cells, epath, xaxis = xaxis)

    subplot(2, sdim) # gradient (cloud)
    dcloud_cells(cells, scale * enes, alpha = 0.05, xaxis = xaxis);
    dcloud_grad(cells, epath, xaxis = xaxis)

    subplot(3, sdim) # nodes (grandient, cloud)
    dcloud_cells(cells, nodes, alpha = 0.05, xaxis = xaxis);
    dcloud_grad (cells, epath, alpha = 0.1, xaxis = xaxis)
    dcloud_nodes(cells, scale * enodes, marker = 'o', alpha = 0.8, xaxis = xaxis)

    subplot(4, sdim) # links (nodes, cloud)
    dcloud_cells(cells, scale * nodes, alpha = 0.01, xaxis = xaxis);
    dcloud_grad (cells, lpath, xaxis = xaxis)
    dcloud_nodes(cells, scale * enodes, marker = 'o', alpha = 0.2, xaxis = xaxis)

    subplot(5, sdim) # pass (links, nodes, cloud)
    dcloud_cells(cells,     scale * nodes , alpha = 0.01, xaxis = xaxis);
    dcloud_nodes(cells,     scale * enodes, marker = 'o', alpha = 0.2, xaxis = xaxis)
    dcloud_grad (cells, lpath, alpha = 0.1, xaxis = xaxis)
    dcloud_nodes(cells, 5 * scale * epass , marker = '^', alpha = 0.9, xaxis = xaxis)

    subplot(6, sdim)
    dcloud_cells   (cells, alpha = 0.05, xaxis = xaxis);
    dcloud_nodes   (cells, scale * enodes, alpha = 0.8, marker = 'o', xaxis = xaxis)
    dcloud_nodes   (cells, 5 * scale * epass , marker = '^', alpha = 0.9, xaxis = xaxis)
    kids  = np.argwhere(epass > 0)
    dcloud_segments(cells, kids, epath, lpath, xaxis = xaxis)

    return



def dcloud_steps_tracks(dfclouds, ndim, ncolumns = 1, scale = 1000., xaxis = 0, **kargs):

    cells  = get_cells(dfclouds, ndim)
    enes   = dfclouds.ene.values
    enodes = dfclouds.enode.values
    nodes  = dfclouds.node.values
    epath  = dfclouds.epath.values
    lpath  = dfclouds.lpath.values
    epass  = dfclouds.epass.values

    track  = dfclouds.track.values
    tnode  = dfclouds.tnode.values
    tpass  = dfclouds.tpass.values

    sdim   = '3d' if ndim == 3 else '2d'
    rscale = 5.

    subplot = canvas(2, ncolumns, 10, 12)

    subplot(1, sdim)
    dcloud_cells   (cells, alpha = 0.05, xaxis = xaxis);
    dcloud_nodes   (cells, scale * enodes, alpha = 0.8, marker = 'o', xaxis = xaxis)
    dcloud_nodes   (cells, rscale * scale * epass , marker = '^', alpha = 0.9, xaxis = xaxis)
    kids  = np.argwhere(epass > 0)
    dcloud_segments(cells, kids, epath, lpath, xaxis = xaxis)
    plt.title('paths between passes')

    subplot(2, sdim)
    plt.title('tracks')
    kidtrack = np.unique(track)
    for ii, kid in enumerate(kidtrack):
        sel  = track == kid
        #print('kid ', kid, 'nodes', ckid[tnode == kid], ' kpass ', ckids[tpass == kid])
        dcloud_cells(_csel(cells, sel), alpha = 0.05, xaxis = xaxis)
        sel  = tnode == kid
        dcloud_nodes(_csel(cells, sel), scale * enodes[sel],  alpha = 0.8,
                           marker = 'o', xaxis = xaxis)
        sel  = tpass == kid
        dcloud_nodes(_csel(cells, sel), rscale * scale * epass[sel], alpha = 0.9,
                               marker = '^',  xaxis = xaxis)

        dcloud_segments(cells, np.argwhere(sel), epath, lpath, xaxis = xaxis)


    return


def dcloud_tracks(cells, track, tnode, tpass, enodes, epass, epath, lpath,
                  scale = 1000., xaxis = 0, mc = None, **kargs):

    rscale = 5.

    kidtrack = np.unique(track[track >= 0])
    for ii, kid in enumerate(kidtrack):
        sel  = track == kid
        dcloud_cells(_csel(cells, sel), alpha = 0.05, xaxis = xaxis)
        sel  = tnode == kid
        dcloud_nodes(_csel(cells, sel), scale * enodes[sel],  alpha = 0.8,
                           marker = 'o', xaxis = xaxis)
        sel  = tpass == kid
        dcloud_nodes(_csel(cells, sel), rscale * scale * epass[sel], alpha = 0.9,
                               marker = '^',  xaxis = xaxis)

        if (mc is not None):
            dcloud_nodes(_csel(cells, sel), 5 * rscale * scale * mc[sel], alpha = 0.9,
                                   marker = '*',  xaxis = xaxis)



        dcloud_segments(cells, np.argwhere(sel), epath, lpath, xaxis = xaxis)
    return



def dcloud_steps_tracks(dfclouds, ndim, ncolumns = 1, scale = 1000., xaxis = 0, **kargs):

    cells   = get_cells(dfclouds, ndim)
    enes    = dfclouds.ene.values
    enodes  = dfclouds.enode.values
    nodes   = dfclouds.node.values
    epath   = dfclouds.epath.values
    lpath   = dfclouds.lpath.values
    epass   = dfclouds.epass.values

    track   = dfclouds.track.values
    tnode   = dfclouds.tnode.values
    tpass   = dfclouds.tpass.values

    ranger  = dfclouds.ranger.values
    eranger = dfclouds.eranger.values

    rscale = 5.
    sdim   = '3d' if ndim == 3 else '2d'

    subplot = canvas(3, ncolumns, 10, 12)

    subplot(1, sdim)
    plt.title('paths')
    dcloud_cells   (cells, alpha = 0.05, xaxis = xaxis);
    dcloud_nodes   (cells, scale * enodes, alpha = 0.8, marker = 'o', xaxis = xaxis)
    dcloud_nodes   (cells, rscale * scale * epass , marker = '^', alpha = 0.9, xaxis = xaxis)
    kids  = np.argwhere(epass > 0)
    dcloud_segments(cells, kids, epath, lpath, xaxis = xaxis)
    plt.title('paths between passes')

    subplot(2, sdim)
    plt.title('tracks')
    dcloud_tracks(cells, track, tnode, tpass, enodes, epass, epath, lpath,
                  scale = scale, xaxis = xaxis, **kargs)

    subplot(3, sdim)
    plt.title('rangers')
    dcloud_tracks(cells, ranger, tnode, tpass, eranger, epass, epath, lpath,
                   scale = scale, xaxis = xaxis, **kargs)

    return


def dcloud_tracks_3dviews(dfclouds, mc = False, ncolumns = 2, type = 'ranger', views = [0, 1, 2],
                          scale = 1000., xaxis = 0, **kargs):

    """ TODO plot the MC-tracks
    """

    ndim   = 3

    cells  = get_cells(dfclouds, ndim)
    enes   = dfclouds.ene.values
    enodes = dfclouds.enode.values
    epass  = dfclouds.epass.values

    epath  = dfclouds.epath.values
    lpath  = dfclouds.lpath.values

    track  = dfclouds.track.values
    tnode  = dfclouds.tnode.values
    tpass  = dfclouds.tpass.values

    ranger  = dfclouds.ranger.values
    eranger = dfclouds.eranger.values

    sdim   = '3d'

    subplot = canvas(len(views), ncolumns, 10, 12)

    xlabels = 2 * ['x (mm)', 'y (mm)', 'z (mm)']

    xtrack = track  if type != 'ranger' else ranger
    xene   = enodes if type != 'ranger' else eranger

    def _view(i):
        subplot(i + 1, sdim)
        plt.title(type + ' view ' + str(i))
        xmc = None if mc is False else dfclouds.mc.values
        dcloud_tracks(cells, xtrack, tnode, tpass, xene, epass, epath, lpath,
                      scale = scale, xaxis = i, mc = xmc, **kargs)

    for i in views: _view(i)

    return


def get_draw_clouds(coors, steps, ene, mccoors = None, mcene = None):

    ndim = len(steps)

    dfclouds, mcpaths = None, None

    if (mccoors is None):
        dfclouds          = clouds.clouds(coors, steps, ene)
    else:
        dfclouds, mcpaths = clouds.clouds_mc(coors, steps, ene, mccoors, mcene)


    cells  = get_cells(dfclouds, ndim)        # all cells

    kid    = dfclouds.kid  .values            # cell id
    enes   = dfclouds.ene  .values            # cell energies

    enode  = dfclouds.enode.values            # energy of nodes
    egrad  = dfclouds.egrad.values            # gradient of the cell
    nodes  = dfclouds.node .values            # nodes id of each cell
    epath  = dfclouds.epath.values            # id of cell direction of gradient

    lpath  = dfclouds.lpath.values            # id of cell with largest gradient but different node
    epass  = dfclouds.epass.values            # energy of pass (sum of the 2 cell in the pass)
    lgrad  = dfclouds.lgrad.values            # gradient with respect cells in a different node

    track  = dfclouds.track.values            # ID of track of this cell
    tnode  = dfclouds.tnode.values            # ID of the track of this node (if cell is a node)
    tpass  = dfclouds.tpass.values            # ID of the track of this pass (is pass is a node)

    sdim    = '3d' if ndim == 3 else '2d'

    xlabels = 2 * ['x (mm)', 'y (mm)', 'z (mm)']
    def _setlabels(xaxis = 0):
        ax = plt.gca()
        plt.xlabel(xlabels[xaxis]); plt.ylabel(xlabels[xaxis + 1]);
        if (ndim == 3): ax.set_zlabel(xlabels[xaxis + 2])


    def draw(plots, xaxis = 0, scale = 1000., rscale = 5., **kargs):

        subplot = canvas(1, 1, 8, 10)
        ax      = subplot(1, sdim)
        plt.title(' view ' + str(xaxis))

        kargs = karg('alpha', 0.5, kargs)

        if (plots['MC-true']):
            assert mccoors is not None
            xxcoors   = _ocells(mccoors, xaxis) if xaxis != 0 else mccoors
            ax.scatter(*xxcoors, c = scale * mcene, s = scale * mcene,
                       marker = '.', label = 'MC-true', **kargs);
            _setlabels()

        if (plots['MC-cells']):
            assert 'mcene' in list(dfclouds.columns)
            xmcene = dfclouds.mcene.values
            pltclouds.dcloud_nodes(cells, rscale * scale * xmcene,
                                    label = 'MC-cells', marker = 'P', xaxis = xaxis, **kargs);

        if (plots['MC-tracks']):
            assert mcpaths is not None
            xcells   = _ocells(cells, xaxis) if xaxis != 0 else cells
            segments = [clouds.get_segment(xcells, path) for path in mcpaths]
            for ii, segment in enumerate(segments):
                plt.plot(*segment, c = 'blue', **kargs)

        if (plots['cells']):
            alpha = kargs['alpha']
            kargs['alpha'] = 0.05 #karg('alpha', 0.01, kargs)
            pltclouds.dcloud_cells(cells, xaxis = xaxis, label = 'cells', **kargs)
            kargs['alpha'] = alpha

        if (plots['gradients']):
            dcloud_grad(cells, epath, xaxis = xaxis, **kargs)

        if (plots['nodes']):
            pltclouds.dcloud_nodes(cells, scale * enode,
                                  marker = 'o', xaxis = xaxis, label = 'nodes', **kargs)

        if (plots['links']):
            dcloud_grad (cells, lpath, xaxis = xaxis, **kargs)


        if (plots['passes']):
            pltclouds.dcloud_nodes(cells, rscale * scale * epass,
                                    marker = '^',  xaxis = xaxis, label = 'passes', **kargs)

        if (plots['tracks']):
            kidtrack = np.unique(track[track >= 0])
            for ii, kid in enumerate(kidtrack):
                sel  = tpass == kid
                pltclouds.dcloud_segments(cells, np.argwhere(sel),
                                          epath, lpath, xaxis = xaxis, **kargs)


    return draw
