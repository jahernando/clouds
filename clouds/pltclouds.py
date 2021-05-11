import numpy             as np
#import pandas            as pd

import matplotlib.pyplot   as plt
import matplotlib.colors   as colors
import matplotlib.cm       as colormap
from mpl_toolkits               import mplot3d
from mpl_toolkits.mplot3d       import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


import clouds.clouds        as clouds
import clouds.graphs        as graphs


def cells_select(cells, sel):
    return [cell[sel] for cell in cells]


# def cells_sorted(cells, value):
#     size   = len(cells[0])
#     kids   = np.arange(size)
#     vals   = sorted(zip(value, kids))
#     uvalue = np.array([v[0] for v in vals])
#     kvalue = np.array([v[1] for v in vals]).astype(int)
#     ucells = cells_select(cells, kvalue)
#     return uvalue, cells


def draw_grad(cells, epath, **kargs):
    """ Draw the gradient of the cells
    """

    ndim   = len(cells)
    coors  = cells
    vcoors = cells_select(cells, epath)
    
    xdirs =[vcoor - coor for vcoor, coor in zip(vcoors, coors)]
    opts = {'scale_units': 'xy', 'scale' : 2.} if ndim == 2 else {'length' : 0.4}

    plt.gca().quiver(*coors, *xdirs, **opts, **kargs)
    
    return


def draw_path(cells, path, **kargs):
    segment = clouds.get_segment(cells, path)
    plt.plot(*segment, **kargs);
    return    


def _scale(values, a = 0, b = 1):
   
    xmin, xmax = np.min(values), np.max(values)
    scale  = (values - xmin)/(xmax - xmin)
    return scale


def draw_histd(cells, bins, values, **kargs):
    
    scale  = _scale(values.astype(float))
    #scale  = values

    ndim = len(bins)
    
    if (ndim == 2):
        plt.gca().hist2d(*cells, bins = bins, weights = values, **kargs)
        #plt.colorbar();
        return

    counts, _  = np.histogramdd(cells, bins = bins, weights = values)
    mask       = counts > 0

    xx, yy, zz = np.meshgrid(*bins)
    umask      = np.copy(mask)
    filled     = np.swapaxes(umask, 0, 1).astype(bool)

    norm      = colors  .Normalize(vmin=min(scale), vmax=max(scale), clip=True)
    mapper    = colormap.ScalarMappable(norm=norm, cmap=colormap.coolwarm)
    facecolor = mapper.to_rgba(scale)
    #ax.voxels(x, y, z, filled, alpha=0.5)

    #facecolor = 'blue'

    plt.gca(projection = '3d')
    plt.gca().voxels(xx, yy, zz, filled, facecolor = facecolor, **kargs);
    return
    
        
options = {'cloud'   : {'alpha'  : 0.2},
           'node'    : {'alpha'  : 0.2},
           'isnode'  : {'marker' : 'x', 'c' : 'black' , 'alpha' : 0.8},
           'isborder': {'marker' : '.', 'c' : 'gray'  , 'alpha' : 0.5},           
           'ispass'  : {'marker' : '|', 'c' : 'black' , 'alpha' : 0.8},
           'iscore'  : {'alpha' : 0.2},
           'isridge' : {'marker' : 'o', 'c' : 'black' , 'alpha' : 0.5},
           'ridge'   : {                'c' : 'black' , 'alpha' : 0.8}
        }


def draw_cloud(cells, bins, df, name = 'e', plot = True):
    
    
    evalue    = df[name+'value']   .values
    if (name == 'p'): evalue = -evalue
   # egrad     = df[name+'grad']    .values
    epath     = df[name+'path']    .values
    elink     = df[name+'link']    .values
    enode     = df[name+'node']    .values
    eisnode   = df[name+'isnode']  .values
    eisborder = df[name+'isborder'].values
    eispass   = df[name+'ispass']  .values
    eisridge  = df[name+'isridge'] .values

    eiscore   = df.iscore          .values
    
    ndim = len(cells)
    #sdim = '' if ndim == 2 else '3d'
    
    scale = _scale(evalue)
    
    ax = plt.gca(projection = '3d') if ndim == 3 else  plt.gca()
      
    def _umask(values):
        counts, _ = np.histogram(cells, bins = bins, weights = values)
        umask     = counts > 0
        return umask
    
    def _kargs(name, opts):
        kargs = dict(options[name])
        if name in opts.keys():
            kargs.update(opts[name])
        #print(name, kargs)
        return kargs
        
    
    def draw(cloud  = True, grad     = False, link = False, node = False,
             isnode = True, isborder = False, ispass = False,
             ridge  = True, isridge  = False, iscore = False,
             rotate = False, voxels  = True , opts = {}):
        
        if (cloud):
            kargs = _kargs('cloud', opts)
            if (voxels):
                draw_histd(cells, bins, evalue, **kargs)
            else:
                ax.scatter(*cells, c = scale, **kargs)
    
        if (node):
            kargs = _kargs('node', opts)
            if (voxels):
                draw_histd(bins, cells, enode, **kargs)
            else:
                ax.scatter(*cells, c = enode, **_kargs)

        if (isnode):
            kargs = _kargs('isnode', opts)
            ax.scatter(*cells_select(cells, eisnode), **kargs)
            
        if (isborder):
            kargs = _kargs('isborder', opts)
            ax.scatter(*cells_select(cells, eisborder), **kargs)
            
        if (ispass):
            kargs = _kargs('ispass', opts)
            ax.scatter(*cells_select(cells, eispass), **kargs)
            
        if (iscore):
            kargs = _kargs('iscore', opts)
            if (voxels):
                draw_histd(cells, bins, eiscore, **kargs)
            else:
                ax.scatter(*cells_select(cells, eiscore), **kargs)
                
            
        if (grad):
            draw_grad(cells, epath)
            
        if (link):
            draw_grad(cells, elink)
            
        if (isridge):
            kargs = _kargs('isridge', opts)
            ax.scatter(*cells_select(cells, eisridge), **kargs)
            
        if (ridge):
            kargs = _kargs('ridge', opts)
            paths    = clouds.get_new_ridges(eispass, epath, elink)
            for path in paths:
                draw_path(cells, path, **kargs)
    
        if (rotate):
            plot_rotate()
            
        return
            
    if plot: draw()
    return draw


def draw_graph(cells, enes, epath, nlinks,
               links  = None, node_size = 100, link_size = 2.):
    
    ndim = len(cells)
    size = len(cells[0])
    ax = plt.gca(projection = '3d') if ndim == 3 else  plt.gca()
    
    escale = _scale(enes)

    isnode = epath == np.arange(size)
    ax.scatter(*cells_select(cells, isnode), marker = 'o',
               c = escale[isnode], s = node_size * escale[isnode], alpha = 0.8)    
    
    #ispass = np.unique(nlinks[nlinks >= 0]) if links is None else np.unique(links)
    #ax.scatter(*cells_select(cells, ispass), marker = '|',
    #           c = escale[ispass], s = size * escale[ispass], alpha = 0.8)
    
    links    = graphs._graph_links(nlinks) if links is None else links
    print(links)
    #lenes    = [max(enes[]])
    paths    = [clouds.get_path_from_link(*ilink, epath) for ilink in links]
    segments = [clouds.get_segment(cells, path) for path in paths]
    for link, segment in zip(links, segments):
        #strength = 'black'
        i0, i1 = link
        iene   = (escale[i0] + escale[i1])/2.
        icolor = plt.cm.rainbow(iene)
        icolor = 'blue'
        plt.plot(*segment, lw = link_size * iene, c = icolor, alpha = 0.6)
        
    #plt.colorbar(p, ax = ax);
    return


def draw_voxels(bins, mask, cells, value = None, **kargs):
    
        
    xx, yy, zz = np.meshgrid(*bins)
    umask      = np.copy(mask)
    filled     = np.swapaxes(umask, 0, 1).astype(bool)

    def _facecolor(scale):
        norm   = colors  .Normalize(vmin=min(scale), vmax=max(scale), clip=True)
        mapper = colormap.ScalarMappable(norm=norm, cmap=colormap.coolwarm)
        fc     = mapper.to_rgba(scale)
        return fc
    #ax.voxels(x, y, z, filled, alpha=0.5)

    facecolor = 'blue' if value is None else _facecolor(value)

    plt.gca(projection = '3d')
    plt.gca().voxels(xx, yy, zz, filled, facecolor = facecolor, **kargs);
    return
    
    

#
#  OTher plotting
#

# rotate the axes and update
def plot_rotate(phi = 30):
    for angle in range(0, 360, 10):
        plt.gca().view_init(phi, angle)
        plt.gcf().canvas.draw()
        plt.pause(0.3)
    return


def plot_tview(cells, ene, t, tname = 't'):
    
    ndim = len(cells)
    
    plt.figure(figsize = (8, 8))
    plt.subplot(2, 2, 1)
    plt.gca().scatter(t, ene);
    plt.grid();
    plt.xlabel(tname); plt.ylabel('ene')
    
    cscale = _scale(ene)    
    xname = ['x', 'y', 'z']
    for i in range(ndim):
        plt.subplot(2, 2, 2 + i)
        x  = cells[i]
        plt.gca().scatter(t, x, c = cscale, s = 10);
        plt.grid();
        plt.xlabel(tname); plt.ylabel(xname[i])

    plt.tight_layout()
    plt.show()
    return


def plot_xyview(cells, ene, mccells = None, mcene = None):
    
    ndim = len(cells)
    plt.figure(figsize = (8, 8))
    ax_ = plt.gcf().add_subplot(2, 2 ,1, projection = '3d') if ndim == 3 else  plt.subplot(2, 2, 1)
   
    def _plot(cells, ene, cmap = 'rainbow'):
        cscale = _scale(ene)
        alpha = 0.2 if ndim == 3 else 0.5
        ax_.scatter(*cells, c = 0.1 * cscale, alpha = alpha)
        plt.grid();
    
        xname = ['x', 'y', 'z']
        for i in range(ndim):
            j = 0 if i == ndim - 1 else i + 1
            plt.subplot(2, 2, 2 + i)
            plt.gca().scatter(cells[i], cells[j], c = cscale, s = 100 * cscale,
                              cmap = cmap, alpha = 0.4)
            plt.grid();
            plt.xlabel(xname[i]); plt.ylabel(xname[j])
            
    _plot(cells, ene)
    if (mcene is not None):
        _plot(mccells, mcene, cmap = 'gray')
    
    plt.tight_layout();
    return


# import matplotlib
# %matplotlib widget
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import matplotlib.cm     as colormap
# from mpl_toolkits               import mplot3d
# from mpl_toolkits.mplot3d       import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# norm   = colors  .Normalize(vmin=min(voxel_ene), vmax=max(voxel_ene), clip=True)
# mapper = colormap.ScalarMappable(norm=norm, cmap=colormap.coolwarm)
# ax.voxels(x, y, z, filled, alpha=0.5, facecolor=mapper.to_rgba(voxel_ene))

#---- utils

# def karg(name, value, kargs):
#     """ if a parameter is not in the key-words dictiory then its include with value
#     inputs:
#         name: str, the name of the parameter
#         value: -, the value of the parameter
#         kargs: dict{str:-}, key-words dictionary
#     returns:
#         kargs: returns the updated (if so) key-words dictionary
#     """
#     kargs[name] = value if name not in kargs.keys() else kargs[name]
#     return kargs


# def canvas(ns : int, ny : int = 2, height : float = 5., width : float = 6.) -> callable:
#     """ create a canvas with ns subplots and ny-columns,
#     return a function to move to next subplot in the canvas
#     """
#     nx  = int(ns / ny + ns % ny)
#     plt.figure(figsize = (width * ny, height * nx))
#     def subplot(iplot, dim = '2d'):
#         """ controls the subplots in a canvas
#             inputs:
#                 iplot: int, index of the plot in the canvas
#                 dim  : str, '3d'  in the case the plot is 3d
#             returns:
#                 nx, ny: int, int (the nx, ny rows and columns of the canvas)
#         """
#         assert iplot <= nx * ny
#         plt.subplot(nx, ny, iplot)
#         if (dim == '3d'):
#             nn = nx * 100 +ny *10 + iplot
#             plt.gcf().add_subplot(nn, projection = dim)
#         return plt.gca()
#     return subplot


# def get_cells(df, ndim):
#     return [df['x'+str(i)].values for i in range(ndim)]


# def _ocells(cells, i = 0):
#     return cells[i:] + cells[:i]


# def _csel(vals, sel):
#     return [val[sel] for val in vals]


# cells_selection = _csel


# #
# # Low level clouds plotting elements
# #

# def dcloud_cells(cells, enes = None, xaxis = 0, **kargs):
#     """ Draw cells, if enes, with its energy
#     """

#     ndim, nsize = len(cells), len(cells[0])

#     enes = np.ones(nsize) if enes is None else enes

#     kargs = karg('marker',   's', kargs)
#     kargs = karg('c' ,  enes, kargs)
#     #kargs = karg('s' ,  10 * enes, kargs)

#     ax = plt.gca()
#     xcells = _ocells(cells, xaxis)
#     ax.scatter(*xcells, **kargs)
#     return
#     #if (chamber): draw_chamber(coors, ax)


# def dcloud_nodes(cells, enodes, **kargs):
#     """ Draw cells that enodes > 0
#     """

#     kargs = karg('marker',   '*', kargs)

#     sel = enodes > 0

#     kargs = karg('s', enodes[sel], kargs)

#     dcloud_cells(_csel(cells, sel), enodes[sel], **kargs)
#     return


# def dcloud_grad(cells, epath, xaxis = 0, **kargs):
#     """ Draw the gradient of the cells
#     """

#     ndim = len(cells)
#     ncells = _csel(cells, epath)
#     coors  = _ocells(cells , xaxis) if xaxis != 0 else cells
#     vcoors = _ocells(ncells, xaxis) if xaxis != 0 else ncells

#     xdirs =[vcoor - coor for vcoor, coor in zip(vcoors, coors)]
#     opts = {'scale_units': 'xy', 'scale' : 2.} if ndim == 2 else {'length' : 0.4}

#     plt.quiver(*coors, *xdirs, **opts, **kargs)


# def dcloud_paths(cells, kids, epath, xaxis = 0, **kargs):
#     """ Draw the segments associated to IDs kids (they should be a passes)
#     """

#     xcells   = _ocells(cells, xaxis) if xaxis != 0 else cells
#     segments = [clouds.get_segment(xcells, clouds.get_path(kid, epath)) for kid in kids]
#     for segment in segments:
#         #print(segment)
#         kargs = karg('c', 'black', kargs)
#         plt.plot(*segment, **kargs)

# def dcloud_segments(cells, kids, epath, lpath, xaxis = 0, **kargs):
#     """ Draw the segments associated to IDs kids (they should be a passes)
#     """

#     xcells   = _ocells(cells, xaxis) if xaxis != 0 else cells
#     segments = [clouds.get_segment(xcells, clouds.get_pass_path(kid, epath, lpath)) for kid in kids]
#     for segment in segments:
#         #print(segment)
#         kargs = karg('c', 'black', kargs)
#         plt.plot(*segment, **kargs)


# #
# #  High Level
# #

# def dcloud_steps(dfclouds, ndim, scale = 1000., rscale = 5., xaxis = 0, ncolumns = 1):
#     """ Draw every step of clouds: i) gradients, ii) nodes, ii) links kk) paths
#     """

#     cells  = get_cells(dfclouds, ndim)
#     enes   = dfclouds.ene.values
#     enodes = dfclouds.enode.values
#     nodes  = dfclouds.node.values
#     epath  = dfclouds.epath.values
#     lpath  = dfclouds.lpath.values
#     epass  = dfclouds.epass.values

#     track  = dfclouds.track.values
#     tnode  = dfclouds.tnode.values
#     tpass  = dfclouds.tpass.values


#     sdim = '3d' if ndim == 3 else '2d'
#     subplot = canvas(7, ncolumns, 6, 8)

#     subplot(1, sdim) # cloud
#     dcloud_cells(cells, scale * enes, alpha = 0.3, xaxis = xaxis);
#     #dcloud_grad(cells, epath, xaxis = xaxis)

#     subplot(2, sdim) # gradient (cloud)
#     dcloud_cells(cells, scale * enes, alpha = 0.05, xaxis = xaxis);
#     dcloud_grad(cells, epath, xaxis = xaxis)

#     subplot(3, sdim) # nodes (grandient, cloud)
#     dcloud_cells(cells, nodes, alpha = 0.05, xaxis = xaxis);
#     dcloud_grad (cells, epath, alpha = 0.1, xaxis = xaxis)
#     dcloud_nodes(cells, scale * enodes, marker = 'o', alpha = 0.8, xaxis = xaxis)

#     subplot(4, sdim) # links (nodes, cloud)
#     dcloud_cells(cells, scale * nodes, alpha = 0.01, xaxis = xaxis);
#     dcloud_grad (cells, lpath, xaxis = xaxis)
#     dcloud_nodes(cells, scale * enodes, marker = 'o', alpha = 0.2, xaxis = xaxis)

#     subplot(5, sdim) # pass (links, nodes, cloud)
#     dcloud_cells(cells,     scale * nodes , alpha = 0.01, xaxis = xaxis);
#     dcloud_nodes(cells,     scale * enodes, marker = 'o', alpha = 0.2, xaxis = xaxis)
#     dcloud_grad (cells, lpath, alpha = 0.1, xaxis = xaxis)
#     dcloud_nodes(cells, 5 * scale * epass , marker = '^', alpha = 0.9, xaxis = xaxis)

#     subplot(6, sdim)
#     dcloud_cells   (cells, alpha = 0.05, xaxis = xaxis);
#     dcloud_nodes   (cells, scale * enodes, alpha = 0.8, marker = 'o', xaxis = xaxis)
#     dcloud_nodes   (cells, 5 * scale * epass , marker = '^', alpha = 0.9, xaxis = xaxis)
#     kids  = np.argwhere(epass > 0)
#     dcloud_segments(cells, kids, epath, lpath, xaxis = xaxis)

#     subplot(7, sdim)
#     kidtrack = np.unique(track)
#     for ii, kid in enumerate(kidtrack):
#         sel  = track == kid
#         dcloud_cells(_csel(cells, sel), alpha = 0.05, xaxis = xaxis)
#         sel  = tnode == kid
#         dcloud_nodes(_csel(cells, sel), scale * enodes[sel],  alpha = 0.8,
#                      marker = 'o', xaxis = xaxis)
#         sel  = tpass == kid
#         dcloud_nodes(_csel(cells, sel), rscale * scale * epass[sel], alpha = 0.9,
#                      marker = '^',  xaxis = xaxis)
#         dcloud_segments(cells, np.argwhere(sel), epath, lpath, xaxis = xaxis)

#     return


# #
# # def dcloud_steps_tracks(dfclouds, ndim, ncolumns = 1, scale = 1000., xaxis = 0, **kargs):
# #
# #     cells  = get_cells(dfclouds, ndim)
# #     enes   = dfclouds.ene.values
# #     enodes = dfclouds.enode.values
# #     nodes  = dfclouds.node.values
# #     epath  = dfclouds.epath.values
# #     lpath  = dfclouds.lpath.values
# #     epass  = dfclouds.epass.values
# #
# #     track  = dfclouds.track.values
# #     tnode  = dfclouds.tnode.values
# #     tpass  = dfclouds.tpass.values
# #
# #     sdim   = '3d' if ndim == 3 else '2d'
# #     rscale = 5.
# #
# #     subplot = canvas(2, ncolumns, 10, 12)
# #
# #     subplot(1, sdim)
# #     dcloud_cells   (cells, alpha = 0.05, xaxis = xaxis);
# #     dcloud_nodes   (cells, scale * enodes, alpha = 0.8, marker = 'o', xaxis = xaxis)
# #     dcloud_nodes   (cells, rscale * scale * epass , marker = '^', alpha = 0.9, xaxis = xaxis)
# #     kids  = np.argwhere(epass > 0)
# #     dcloud_segments(cells, kids, epath, lpath, xaxis = xaxis)
# #     plt.title('paths between passes')
# #
# #     subplot(2, sdim)
# #     plt.title('tracks')
# #     kidtrack = np.unique(track)
# #     for ii, kid in enumerate(kidtrack):
# #         sel  = track == kid
# #         #print('kid ', kid, 'nodes', ckid[tnode == kid], ' kpass ', ckids[tpass == kid])
# #         dcloud_cells(_csel(cells, sel), alpha = 0.05, xaxis = xaxis)
# #         sel  = tnode == kid
# #         dcloud_nodes(_csel(cells, sel), scale * enodes[sel],  alpha = 0.8,
# #                            marker = 'o', xaxis = xaxis)
# #         sel  = tpass == kid
# #         dcloud_nodes(_csel(cells, sel), rscale * scale * epass[sel], alpha = 0.9,
# #                                marker = '^',  xaxis = xaxis)
# #
# #         dcloud_segments(cells, np.argwhere(sel), epath, lpath, xaxis = xaxis)
# #
# #
# #     return
# #
# #
# # def dcloud_tracks(cells, track, tnode, tpass, enodes, epass, epath, lpath,
# #                   scale = 1000., xaxis = 0, mc = None, **kargs):
# #
# #     rscale = 5.
# #
# #     kidtrack = np.unique(track[track >= 0])
# #     for ii, kid in enumerate(kidtrack):
# #         sel  = track == kid
# #         dcloud_cells(_csel(cells, sel), alpha = 0.05, xaxis = xaxis)
# #         sel  = tnode == kid
# #         dcloud_nodes(_csel(cells, sel), scale * enodes[sel],  alpha = 0.8,
# #                            marker = 'o', xaxis = xaxis)
# #         sel  = tpass == kid
# #         dcloud_nodes(_csel(cells, sel), rscale * scale * epass[sel], alpha = 0.9,
# #                                marker = '^',  xaxis = xaxis)
# #
# #         if (mc is not None):
# #             dcloud_nodes(_csel(cells, sel), 5 * rscale * scale * mc[sel], alpha = 0.9,
# #                                    marker = '*',  xaxis = xaxis)
# #
# #
# #
# #         dcloud_segments(cells, np.argwhere(sel), epath, lpath, xaxis = xaxis)
# #     return
# #
# #
# #
# # def dcloud_steps_tracks(dfclouds, ndim, ncolumns = 1, scale = 1000., xaxis = 0, **kargs):
# #
# #     cells   = get_cells(dfclouds, ndim)
# #     enes    = dfclouds.ene.values
# #     enodes  = dfclouds.enode.values
# #     nodes   = dfclouds.node.values
# #     epath   = dfclouds.epath.values
# #     lpath   = dfclouds.lpath.values
# #     epass   = dfclouds.epass.values
# #
# #     track   = dfclouds.track.values
# #     tnode   = dfclouds.tnode.values
# #     tpass   = dfclouds.tpass.values
# #
# #     ranger  = dfclouds.ranger.values
# #     eranger = dfclouds.eranger.values
# #
# #     rscale = 5.
# #     sdim   = '3d' if ndim == 3 else '2d'
# #
# #     subplot = canvas(3, ncolumns, 10, 12)
# #
# #     subplot(1, sdim)
# #     plt.title('paths')
# #     dcloud_cells   (cells, alpha = 0.05, xaxis = xaxis);
# #     dcloud_nodes   (cells, scale * enodes, alpha = 0.8, marker = 'o', xaxis = xaxis)
# #     dcloud_nodes   (cells, rscale * scale * epass , marker = '^', alpha = 0.9, xaxis = xaxis)
# #     kids  = np.argwhere(epass > 0)
# #     dcloud_segments(cells, kids, epath, lpath, xaxis = xaxis)
# #     plt.title('paths between passes')
# #
# #     subplot(2, sdim)
# #     plt.title('tracks')
# #     dcloud_tracks(cells, track, tnode, tpass, enodes, epass, epath, lpath,
# #                   scale = scale, xaxis = xaxis, **kargs)
# #
# #     subplot(3, sdim)
# #     plt.title('rangers')
# #     dcloud_tracks(cells, ranger, tnode, tpass, eranger, epass, epath, lpath,
# #                    scale = scale, xaxis = xaxis, **kargs)
# #
# #     return
# #
# #
# # def dcloud_tracks_3dviews(dfclouds, mc = False, ncolumns = 2, type = 'ranger', views = [0, 1, 2],
# #                           scale = 1000., xaxis = 0, **kargs):
# #
# #     """ TODO plot the MC-tracks
# #     """
# #
# #     ndim   = 3
# #
# #     cells  = get_cells(dfclouds, ndim)
# #     enes   = dfclouds.ene.values
# #     enodes = dfclouds.enode.values
# #     epass  = dfclouds.epass.values
# #
# #     epath  = dfclouds.epath.values
# #     lpath  = dfclouds.lpath.values
# #
# #     track  = dfclouds.track.values
# #     tnode  = dfclouds.tnode.values
# #     tpass  = dfclouds.tpass.values
# #
# #     ranger  = dfclouds.ranger.values
# #     eranger = dfclouds.eranger.values
# #
# #     sdim   = '3d'
# #
# #     subplot = canvas(len(views), ncolumns, 10, 12)
# #
# #     xlabels = 2 * ['x (mm)', 'y (mm)', 'z (mm)']
# #
# #     xtrack = track  if type != 'ranger' else ranger
# #     xene   = enodes if type != 'ranger' else eranger
# #
# #     def _view(i):
# #         subplot(i + 1, sdim)
# #         plt.title(type + ' view ' + str(i))
# #         xmc = None if mc is False else dfclouds.mc.values
# #         dcloud_tracks(cells, xtrack, tnode, tpass, xene, epass, epath, lpath,
# #                       scale = scale, xaxis = i, mc = xmc, **kargs)
# #
# #     for i in views: _view(i)
# #
# #     return
# #


# def get_draw_clouds(dfclouds, mccoors = None, mcene = None):

#     ndim = 3 if 'x2' in list(dfclouds.columns) else 2

#     #def get_draw_clouds(coors, steps, ene, mccoors = None, mcene = None):

#     #ndim = len(steps)

#     #dfclouds, mcpaths = None, None

#     #if (mccoors is None):
#     #    dfclouds          = clouds.clouds(coors, steps, ene)
#     #else:
#     #    dfclouds, mcpaths = clouds.clouds_mc(coors, steps, ene, mccoors, mcene)


#     cells  = get_cells(dfclouds, ndim)        # all cells

#     kid    = dfclouds.kid  .values            # cell id
#     enes   = dfclouds.ene  .values            # cell energies

#     enode  = dfclouds.enode.values            # energy of nodes
#     egrad  = dfclouds.egrad.values            # gradient of the cell
#     nodes  = dfclouds.node .values            # nodes id of each cell
#     epath  = dfclouds.epath.values            # id of cell direction of gradient

#     lpath  = dfclouds.lpath.values            # id of cell with largest gradient but different node
#     epass  = dfclouds.epass.values            # energy of pass (sum of the 2 cell in the pass)
#     lgrad  = dfclouds.lgrad.values            # gradient with respect cells in a different node

#     track  = dfclouds.track.values            # ID of track of this cell
#     tnode  = dfclouds.tnode.values            # ID of the track of this node (if cell is a node)
#     tpass  = dfclouds.tpass.values            # ID of the track of this pass (is pass is a node)

#     crest  = dfclouds.crest.values             # ID of the crest

#     sdim    = '3d' if ndim == 3 else '2d'

#     xlabels = 2 * ['x (mm)', 'y (mm)', 'z (mm)']
#     def _setlabels(xaxis = 0):
#         ax = plt.gca()
#         plt.xlabel(xlabels[xaxis]); plt.ylabel(xlabels[xaxis + 1]);
#         if (ndim == 3): ax.set_zlabel(xlabels[xaxis + 2])


#     plots = {}
#     mc_true  = mccoors is not None
#     if (mc_true):
#         plots['MC-true']     = False
#     mc_cells = 'mcene' in list(dfclouds.columns)
#     if (mc_cells):
#         plots['MC-cells']    = False
#     #plots['MC-tracks']   = False
#     plots['cells']       = True
#     plots['gradients']   = False
#     plots['nodes']       = True
#     plots['links']       = False
#     plots['passes']      = False
#     plots['segments']    = False
#     plots['tracks']      = False
#     plots['crests']      = True


#     def draw(plots, xaxis = 0, scale = 1000., rscale = 3., **kargs):

#         subplot = canvas(1, 1, 6, 8)
#         ax      = subplot(1, sdim)
#         plt.title(' view ' + str(xaxis))

#         kargs = karg('alpha', 0.5, kargs)

#         if (mc_true):
#             if plots['MC-true']:
#                 xxcoors   = _ocells(mccoors, xaxis) if xaxis != 0 else mccoors
#                 ax.scatter(*xxcoors, c = scale * mcene, s = scale * mcene,
#                            marker = '.', label = 'MC-true', **kargs);

#         if (mc_cells):
#             if plots['MC-cells']:
#                 xmcene = dfclouds.mcene.values
#                 dcloud_nodes(cells, rscale * scale * xmcene, label = 'MC-cells',
#                              marker = 'P', xaxis = xaxis, **kargs);


#         # if (plots['MC-tracks']):
#         #     assert mcpaths is not None
#         #     xcells   = _ocells(cells, xaxis) if xaxis != 0 else cells
#         #     segments = [clouds.get_segment(xcells, path) for path in mcpaths]
#         #     for ii, segment in enumerate(segments):
#         #         plt.plot(*segment, c = 'blue', **kargs)

#         if (plots['cells']):
#             alpha = kargs['alpha']
#             kargs['alpha'] = 0.05 #karg('alpha', 0.01, kargs)
#             dcloud_cells(cells, xaxis = xaxis, label = 'cells', **kargs)
#             kargs['alpha'] = alpha

#         if (plots['gradients']):
#             dcloud_grad(cells, epath, xaxis = xaxis, **kargs)

#         if (plots['nodes']):
#             dcloud_nodes(cells, scale * enode, marker = 'o',
#              xaxis = xaxis, label = 'nodes', **kargs)

#         if (plots['links']):
#             dcloud_grad (cells, lpath, xaxis = xaxis, **kargs)

#         if (plots['passes']):
#             dcloud_nodes(cells, rscale * scale * epass, marker = '^',
#                          xaxis = xaxis, label = 'passes', **kargs)

#         if (plots['segments']):
#             sel  = epass > 0
#             dcloud_segments(cells, np.argwhere(sel), epath, lpath,
#                             xaxis = xaxis, **kargs)

#         if (plots['tracks']):
#             kidtrack = np.unique(track[track >= 0])
#             for ii, kid in enumerate(kidtrack):
#                 sel  = tpass == kid
#                 dcloud_segments(cells, np.argwhere(sel), epath, lpath,
#                                 xaxis = xaxis, **kargs)

#         if (plots['crests']):
#             kidtrack = np.unique(crest[crest >= 0])
#             for ii, kid in enumerate(kidtrack):
#                 sel  = tpass == kid
#                 dcloud_segments(cells, np.argwhere(sel), epath, lpath,
#                                 xaxis = xaxis, **kargs)

#         _setlabels(xaxis)
#         plt.legend()
#         return

#     return draw, plots
