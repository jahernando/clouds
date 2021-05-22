#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:21:10 2021

@author: hernando
"""
import re
import glob

import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt

import clouds.clouds    as clouds
import clouds.mc_clouds as mcclouds


#--- io

DATADIR = "/Users/hernando/investigacion/NEXT/data/MC/NEW/bb0nu_esmeralda/"
MAPFILE = '/Users/hernando/investigacion/NEXT/data/MC/NEW/bb0nu_esmeralda/map_8264.h5'


ut_scale     = clouds.ut_scale
cells_select = clouds.cells_select

get_file_number = lambda filename: int(re.findall("_[0-9]+_", filename)[0][1:-1])

def get_filenames(datadir = None):
    
    #datadir   = "/home/hernando/data/NEW/MC/bb0nu_esmeralda/"
    datadir   = DATADIR if datadir is None else datadir
    files     = glob.glob(datadir + '*.h5')
    def file_number(file):
        fname = file .split('/')[-1]
        ifile = fname.split('_')[1]
        return str(ifile)
    #print(files)
    filenames = sorted(files, key = file_number)
    print('total files', len(filenames), get_file_number(filenames[1]))
    return filenames


def get_event_numbers(filename):
    data_events = pd.read_hdf(filename, "Run/events")
    events       = data_events["evt_number"].values
    return events


def _get_dv(mapfile = None):
    fname = MAPFILE if mapfile is None else mapfile
    maps_te = pd.read_hdf(fname, "time_evolution")
    dv = maps_te.dv.mean()
    return dv


def get_event(filename = None, event = None):
    filename  = np.random.choice(get_filenames()) \
        if filename is None else filename

    CHITS_lowTh  = pd.read_hdf(filename, "/CHITS/lowTh") .groupby("event")
    #CHITS_highTh = pd.read_hdf(filename, "/CHITS/highTh").groupby("event")

    MChits      = pd.read_hdf(filename, "MC/hits").groupby("event_id")
    data_events = pd.read_hdf(filename, "Run/events")
    event       = np.random.choice(data_events["evt_number"]) if event is None else event
    #print('filename, event  = ', get_file_number(filename), ', ', event)
    
    low  = CHITS_lowTh .get_group(event)
    #high = CHITS_highTh.get_group(event)
    true = MChits      .get_group(event)

    def split_hits(hitsdf, weight="E"):
    
        xyz = hitsdf[["X", "Y", "Z"]].values
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        w = hitsdf[weight].values 
        return x, y, z, w

    x, y, z, w = split_hits(low, weight="E")
    coors = (x, y, z)
    ene   = 1e-5 * w

    dv = _get_dv()

    mcx, mcy, mcz, mcid = true["x"].values, true["y"].values, \
        dv*true["z"].values, true['particle_id'].values
    mccoors = (mcx, mcy, mcz)
    mcene, mctime = true["energy"].values, true['time'].values
    
    #print('Energy MC ', np.sum(mcene), ', RC ', np.sum(ene))
    
    #if (np.sum(ene) <= 2.1): 
    #    return get_event()
    
    return coors, ene, mccoors, mcene, mctime, mcid


def run_event(steps, coors, ene, mccoors, mcene, mctime, mcid):
    bins, mask, cells, df = clouds.clouds(coors, steps, ene)
    dft = mcclouds.mctrue(bins, mask, cells, df, mccoors, mcene, mctime, mcid)
    dft = df_add_uevar(df)
    dft = df_add_pair(dft)
    dft = df_add_pair_extreme(dft)
    return dft


def run(nfiles = 100, steps = (10, 10, 6)):
    
    filenames = get_filenames()
    dfs = []
    for i, filename in enumerate(filenames[:nfiles]):
        events = get_event_numbers(filename)
        print('file ', i, filename, ', events ', events)
        for event in events:
            try:
                evtdata = get_event(filename, event)
            except:
                continue
            ene = evtdata[1]
            if (np.sum(ene) < 2.1): continue
            #print('Event ', event, np.sum(ene))
            idf = run_event(steps, *evtdata)
            idf['file']  = i
            idf['event'] = event
            dfs.append(idf)

    df = pd.concat(dfs, ignore_index = True)
    return df
    

def df_add_uevar(dft):
    for i in range(3):
        ge = dft['ge'+str(i)].values
        vg = dft.vgrad.values
        ue = ge/vg
        ue[np.isnan(ue)] = -1.2
        dft['ue'+str(i)] = ue
    return dft

def df_add_pair(dft):
    egrel = (dft.egradrel.values).astype(int)
    cenes = dft.energy.values
    epair = cenes + cenes[egrel]
    kid   = dft.kid
    epair[kid == egrel] = cenes[kid == egrel]
    dft['epair'] = epair
    return dft

def df_add_pair_extreme(dft):
    isextreme = dft.mceextr.values == 1
    epathrel  = (dft.epathrel.values).astype(int)
    ispair    = (isextreme) | (isextreme[epathrel])
    dft['mcpairextr'] = ispair
    return dft

#--- plotting


def get_plotter_var(mc, mcextr):

    sel0 = np.full(len(mc), True, dtype = bool)

    def _vplot(var, nbins = 100, vname = '', sel = None):
        x0, x1 = np.min(var), np.max(var)
        bins = np.linspace(x0, x1, nbins)
        sel = sel0 if sel is None else sel
        plt.hist(var[sel & ~mc], bins, histtype = 'step', density = True, label = '!mc');
        plt.hist(var[sel &  mc], bins, histtype = 'step', density = True, label = 'mc');
        if (len(mcextr == 1) > 0):
            plt.hist(var[sel & (mcextr == 1)], bins,  histtype = 'step',
                     density = True, label = 'ext');
        plt.xlabel(vname); plt.legend();

    return _vplot


def get_drawer(cells):

    size = len(cells[0])
    def _plot(var = None, sel = None,
              size_marker = 50, color = None, size_scale = True, 
              **kargs):
        
        var  = np.ones(size)                     if var is None else var
        sel  = np.full(size, True, dtype = bool) if sel is None else sel
        scale = ut_scale(var) if np.min(var) < np.max(var) else np.ones(size)
        col   =               scale[sel] if color is None else color
        siz   = size_marker * scale[sel] if size_scale    else size_marker
        if (np.all(sel)):
            plt.gca().scatter(*cells, c = col, s = siz, **kargs)
        else:
            plt.gca().scatter(*cells_select(cells, sel), 
                              c = col, s = siz, **kargs)
            plt.xlabel('x'); plt.ylabel('y'); 
    return _plot   


def froc(mc, mcext):
    
    ntrues = np.sum(mc == True)
    nextr  = mcext == True
    
    def _roc(val):
        
        ntot = len(val)
        nmc  = len(val[mc == True])
        nok  = len(val[mcext]) if nextr >0 else 0
        
        eff = float(nmc)/float(ntrues)
        pur = float(nmc)/float(ntot)
        return eff, pur, nok
        
    
    