#!/usr/bin/env python
import os
import pandas as pd
from matlabtools import convert_sonics, convert_met_20Hz, convert_turbine

verbose = True

default_compression = {
    'zlib': True,
    'complevel': 9,
}

# TODO: specify data input/output options
#matpath = '.' # location of mat files
matpath = '/Users/equon/WakeDynamics/DAP/from_Tommy'
outpath = 'ncdata' # location of new netcdf files
identifier = 'z01.b0'

# TODO: specify datetime range of files to convert
datetimes = pd.date_range('2017-03-08 02:00','2017-03-08 02:00',
                          freq='10min')

# TODO: specify metadata
sonics_description = 'Description of raw sonic data here!'
met_description = 'Description of 20-Hz met data here!'
turbine_description = 'TTU Turbine WTGa1 (?)'

# TODO: pick one of the sonic processing options
# - separate_datasets (exact same data as matlab)
# - round_times (nearest value at regular output intervals, may lose some data)
# - interp_times (cubic interpolation to regular output intervals, data will be
#   slightly modified)
separate_datasets = True
round_times = False
interp_times = False

#==============================================================================
#
# EXECUTION STARTS HERE
#
if not os.path.isdir(outpath):
    os.makedirs(outpath)

for datetime in datetimes:
    tag = identifier + datetime.strftime('.%Y%m%d.%H%M%S')
    print('Processing',tag)

    # sonics output (100 Hz)

    metpath = os.path.join(matpath, 'met.'+tag+'.mat')
    sonics = convert_sonics(metpath,
                            separate_datasets=separate_datasets,
                            round_times=round_times,
                            interp_times=interp_times,
                            description=sonics_description,
                            verbose=verbose)
    if separate_datasets:
        for key,ds in sonics.items():
            fpath = os.path.join(outpath, key+'.'+tag+'.nc')
            ds.to_netcdf(fpath, encoding={varname: default_compression for varname in ds.var()})
    else:
        fpath = os.path.join(outpath, 'sonics.'+tag+'.nc')
        sonics.to_netcdf(fpath, encoding={varname: default_compression for varname in sonics.var()})

    # other met output (~20 Hz)

    met_20Hz = convert_met_20Hz(metpath,
                                description=met_description,
                                verbose=verbose)
    met_20Hz.to_netcdf(os.path.join(outpath, 'met_20Hz.'+tag+'.nc'),
                       encoding={varname: default_compression for varname in met_20Hz.var()})

    # turbine output (~50 Hz)

    turb = convert_turbine(os.path.join(matpath, 'turbine.'+tag+'.mat'),
                           description=turbine_description,
                           verbose=verbose)
    turb.to_netcdf(os.path.join(outpath, 'turbine.'+tag+'.nc'),
                   encoding={varname: default_compression for varname in
                       turb.var()})

