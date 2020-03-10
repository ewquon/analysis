import numpy as np
from scipy.io import loadmat
import pandas as pd
import xarray as xr

def convert_met(fpath,
                sonic_outputs=['T','U','V','W','WS','WD'],
                height_unit='m',
                verbose=True):
    """
    Process sonic data from matlab file
    """
    if verbose:
        print('Loading',fpath)
    data = loadmat(fpath, struct_as_record=False, squeeze_me=True)

    # check contents, save attributes
    attrs = {}
    sonic_heights = {}
    for key,val in data.items():
        if key.startswith('__') and key.endswith('__'):
            if not val == []:
                attrs[key.strip('_')] = val
        elif key.startswith('sonic_') and key.endswith(height_unit):
            hgt = float(key[len('sonic_'):-1])
            sonic_heights[key] = hgt
        elif key == 'log20Hz':
            continue
        else:
            raise KeyError('Unexpected "'+key+'" in mat struct')
    if verbose:
        print('Attributes',attrs)
        
    # process sonics
    # note: now the matlab structures no longer behave like dictionaries...
    sonic_outputs = ['Sonic'+output for output in sonic_outputs]
    sonic_starttime = None
    sonic_units = {}
    dflist = []
    for key,hgt in sonic_heights.items():
        sonic = data[key]
     
        # check data
        contents = dir(sonic)
        for output in sonic_outputs+['units']:
            assert output in contents, output+' missing in sonic data'
        
        # get units
        allunits = getattr(sonic,'units')
        for output in sonic_outputs:
            units = getattr(allunits,output)
            if units in sonic_units.keys():
                assert units == sonic_units[output], 'mismatched sonic units'
            else:
                sonic_units[output] = units
        
        # get start time for timestamp conversion
        t0 = pd.to_datetime(sonic.units.Time,
                            format='seconds from %Y-%m-%d %H:%M:%S UTC')
        if sonic_starttime is None:
            sonic_starttime = t0
        else:
            assert t0 == sonic_starttime, 'mismatched sonic start time'
        
        # now, get the data
        df = None
        sonic_times = None
        for output in sonic_outputs:
            if verbose:
                print('Processing',output,sonic_units[output],'at',hgt,height_unit)
                
            # get timestamp from seconds since ...
            tseconds = getattr(sonic,'Time')
            if any(pd.isna(tseconds)):
                # interpolate, assuming equally spaced sample times
                # - need to do this before converting to timedelta
                isnat = np.where(pd.isna(tseconds))[0]
                tseconds = pd.Series(tseconds).interpolate()
                print('WARNING: NaT(s) found')
                if verbose:
                    for i in isnat:
                        print('  interpolated timestamp at',
                              t0 + pd.to_timedelta(tseconds.iloc[i], unit='s'))
            t = t0 + pd.to_timedelta(tseconds, unit='s')
            if sonic_times is None:
                # save times for datetime index
                sonic_times = t
            else:
                # sanity check
                assert all(t == sonic_times), 'mismatched sonic times'
            
            # create dataframe if necessary
            if df is None:
                df = pd.DataFrame(index=sonic_times)
                df['height'] = hgt
            
            df[output] = getattr(sonic,output)

        dflist.append(df)
    
    # convert to xarray with metadata
    df = pd.concat(dflist)
    df.index.name = 'datetime'
    df = df.set_index('height',append=True)
    ds = df.to_xarray()
    ds.attrs = attrs
    
    return ds

