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

import clouds as clouds


#--- io

DATADIR = "/Users/hernando/investigacion/NEXT/data/MC/NEW/bb0nu_esmeralda/"
MAPFILE = '/Users/hernando/investigacion/NEXT/data/MC/NEW/bb0nu_esmeralda/map_8264.h5'



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


def _get_dv(mapfile = None):
    fname = MAPFILE if mapfile is None else mapfile
    maps_te = pd.read_hdf(fname, "time_evolution")
    dv = maps_te.dv.mean()
    return dv


def get_event(filename = None, event = None):
    filenames = get_filenames() if filename is None else None 
    filename  = np.random.choice(filenames) if filename is None else filename

    CHITS_lowTh  = pd.read_hdf(filename, "/CHITS/lowTh") .groupby("event")
    #CHITS_highTh = pd.read_hdf(filename, "/CHITS/highTh").groupby("event")

    MChits = pd.read_hdf(filename, "MC/hits").groupby("event_id")
    data_events = pd.read_hdf(filename, "Run/events")
    event       = np.random.choice(data_events["evt_number"]) if event is None else event
    print('filename, event  = ', get_file_number(filename), ', ', event)
    
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
    
    print('Energy MC ', np.sum(mcene), ', RC ', np.sum(ene))
    
    if (np.sum(ene) <= 2.1): 
        return get_event()
    
    return coors, ene, mccoors, mcene, mctime, mcid


#--- analysis

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
        
    
    