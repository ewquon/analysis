import numpy as np
from scipy.io import loadmat
import pandas as pd
import xarray as xr

sensor_descriptions = {
    'TRH': 'MAYBE_RM_YOUNG?', # for temperature/relative humidity
    'BP': 'SENSOR_NAME_HERE', # for barometric pressure
    'Calc_BP': 'SENSOR_NAME_HERE', # for barometric pressure
    'Calc_Rho': 'SENSOR_NAME_HERE', # for barometric pressure
    'cup': 'cup anemometer',
    'vane': 'wind vane',
}

sonic_descriptions = {
    'SonicT': 'sonic temperature',
    'SonicU': 'sonic west-east wind component',
    'SonicV': 'sonic south-north wind component',
    'SonicW': 'sonic vertical wind componentt',
    'SonicWS': 'sonic wind speed',
    'SonicWD': 'sonic wind direction',
}

other_descriptions = {
    'BP': 'barometric pressure',
    'Calc_BP': 'calculated barometric pressure (?)',
    'Calc_Rho': 'calculated density (?)',
    'CupWS': 'wind speed',
    'RH': 'relative humidity',
    'Temp': 'air temperature',
    'VaneWD': 'wind direction',
}

time_name = 'Time'
height_unit = 'm'


def convert_met(fpath,
                sonic_outputs=['T','U','V','W','WS','WD'],
                description=None,
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
                if isinstance(val,bytes):
                    val = val.decode()
                attrs[key.strip('_')] = val
        elif key.startswith('sonic_') and key.endswith(height_unit):
            height = float(key[len('sonic_'):-1])
            sonic_heights[key] = height
        elif key == 'log20Hz':
            continue
        else:
            raise KeyError('Unexpected "'+key+'" in mat struct')
    if description:
        attrs['description'] = description
    if verbose:
        print('Attributes',attrs)
        
    # process sonics
    # note: after the first level, the matlab structures no longer behave like
    #       dictionaries...
    sonic_outputs = ['Sonic'+output for output in sonic_outputs]
    sonic_starttime = None
    sonic_units = {}
    dflist = []
    for key,height in sonic_heights.items():
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
        t0 = pd.to_datetime(getattr(sonic.units,time_name),
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
                print('Processing',output,sonic_units[output],
                      'at',height,height_unit)
                
            # get timestamp from seconds since ...
            tseconds = getattr(sonic,time_name)
            if any(pd.isna(tseconds)):
                # interpolate, assuming equally spaced sample times
                # - need to do this before converting to timedelta
                isnat = np.where(pd.isna(tseconds))[0]
                tseconds = pd.Series(tseconds).interpolate()
                if verbose:
                    print('WARNING: NaT(s) found')
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
            outputchannel = getattr(sonic,output)
            if df is None:
                df = pd.DataFrame(index=sonic_times,dtype=outputchannel.dtype)
                df['height'] = height
            df[output] = outputchannel

        dflist.append(df)
    
    # convert to xarray with metadata
    df = pd.concat(dflist)
    df.index.name = 'datetime'
    df = df.set_index('height',append=True)
    ds = df.to_xarray()
    
    # assign attributes
    ds.attrs = attrs
    for output,desc in sonic_descriptions.items():
        ds[output] = ds[output].assign_attrs(description=desc,
                                             units=sonic_units[output])
    return ds


def convert_met_20Hz(fpath,
                     sensor_groups={
                         'BP': ['BP'],
                         'Calc_BP': ['Calc_BP'],
                         'Calc_Rho': ['Calc_Rho'],
                         'TRH': ['Temp','RH'],
                         'cup': ['CupWS'],
                         'vane': ['VaneWD'],
                     },
                     description=None,
                     verbose=True):
    """
    Process 20-hz data from matlab file for instruments other than sonics
    """
    if verbose:
        print('Loading',fpath)
    data = loadmat(fpath, struct_as_record=False, squeeze_me=True)

    # save attributes
    attrs = {}
    for key,val in data.items():
        if key.startswith('__') and key.endswith('__'):
            if not val == []:
                if isinstance(val,bytes):
                    val = val.decode()
                attrs[key.strip('_')] = val
    if description:
        attrs['description'] = description
    if verbose:
        print('Attributes',attrs)
    data = data['log20Hz']

    # read time
    t0 = pd.to_datetime(getattr(data.units,time_name),
                        format='seconds from %Y-%m-%d %H:%M:%S UTC')
    tseconds = getattr(data,time_name)
    if any(pd.isna(tseconds)):
        # interpolate, assuming equally spaced sample times
        # - need to do this before converting to timedelta
        isnat = np.where(pd.isna(tseconds))[0]
        tseconds = pd.Series(tseconds).interpolate()
        if verbose:
            print('WARNING: NaT(s) found')
            for i in isnat:
                print('  interpolated timestamp at',
                      t0 + pd.to_timedelta(tseconds.iloc[i], unit='s'))
    t = t0 + pd.to_timedelta(tseconds, unit='s')

    # check specified sensor_groups, get units
    output_units = {}
    sensor_outputs = {}
    tmp = data._fieldnames
    tmp.remove(time_name)
    tmp.remove('units')
    for sensor,outputs in sensor_groups.items():
        sensor_outputs[sensor] = {}
        for prefix in outputs:
            sensor_outputs[sensor][prefix] = []
            for output in tmp.copy():
                if output.startswith(prefix) \
                        and output.split('_')[-1].endswith(height_unit):
                    units = getattr(data.units,output)
                    try:
                        assert output_units[prefix] == units
                    except KeyError:
                        output_units[prefix] = units
                    assert len(getattr(data,output)) == len(t)
                    sensor_outputs[sensor][prefix].append(output)
                    tmp.remove(output)
    if verbose and (len(tmp) > 0):
        print('Ungrouped outputs (these will be ignored):',tmp)

    # now read all the data
    ds = xr.Dataset()
    for sensor,outputs in sensor_groups.items():
        for prefix in outputs:
            df = pd.DataFrame(index=t)
            for output in sensor_outputs[sensor][prefix]:
                # loop over outputs for the given sensor and field (indicated
                # by prefix) at one or more heights
                height = float(output.split('_')[-1][:-len(height_unit)])
                if verbose:
                    print('Processing',output,'at',height,height_unit,'from',sensor)
                series = getattr(data,output)
                if np.all(pd.isna(series)):
                    if verbose:
                        print('WARNING:',output,'is all NaN')
                else:
                    df[height] = series
#            print('Setting',prefix)
#            print(df)
            df = df.stack()
            df.index.names = ['datetime','height_'+sensor]
            ds[prefix] = df.to_xarray()

    # assign attributes
    ds.attrs = attrs
    for sensor,outputs in sensor_groups.items():
        for prefix in outputs:
            desc = '{:s} {:s}'.format(sensor_descriptions[sensor],
                                      other_descriptions[prefix])
            ds[prefix] = ds[prefix].assign_attrs(description=desc,
                                                 units=output_units[prefix])
    return ds

