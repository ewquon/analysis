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
    'SonicT': 'sonic anemometer temperature',
    'SonicU': 'sonic anemometer west-east wind component',
    'SonicV': 'sonic anemometer south-north wind component',
    'SonicW': 'sonic anemometer vertical wind componentt',
    'SonicWS': 'sonic anemometer wind speed',
    'SonicWD': 'sonic anemometer wind direction',
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

turbine_descriptions = {
    'Azimuth': 'azimuthal position (of blade 1?)',
    'BldPitch1': 'blade 1 pitch angle',
    'Ctr_PitchRef1': 'DESCRIPTION HERE',
    'Ctr_YawRef': 'DESCRIPTION HERE',
    'DriveState': 'DESCRIPTION HERE',
    'GenPwr': 'generator power',
    'GenSpd': 'generator speed',
    'GenTq': 'generator torque',
    'HSShftSpd': 'high-speed shaft speed', # same as generator speed?
    'NacWindDir': 'nacelle (wind vane?) wind direction',
    'NacWindSpd': 'nacelle (??? anemometer?) wind speed',
    'ProdCtrlState': 'DESCRIPTION HERE',
    'RotorSpd': 'rotor speed', # low-speed shaft speed?
    'TurbineState': 'DESCRIPTION HERE',
    'YawHeading': 'yaw heading',
    'YawOffset': 'DESCRIPTION HERE', # YawHeading - NacWindDir ?
}


time_name = 'Time'
height_unit = 'm'


def convert_sonics(fpath,
                   sonic_outputs=['T','U','V','W','WS','WD'],
                   description=None,
                   separate_datasets=False,
                   round_times=False,
                   interp_times=False, interp_method='cubic',
                   verbose=True):
    """
    Process sonic data from matlab file

    If separate_datasets is True, then a list of xarray datasets will be
    returned for each sonic.

    If round_times is True, then determine a common date-time index
    based on an estimated sampling frequency. However, unless the
    original timestamps have the same sampling frequency, this will
    likely result in a loss of some data points. Unlike interp_times,
    this will not modify the original data.

    If interp_times is True, then determine a common date-time index
    based on an estimated sampling frequency. Unless the original time-
    stamps have the same sampling frequency, this will modify the
    original data through interpolation (cubic by default).
    """
    assert np.count_nonzero([separate_datasets, round_times, interp_times]) <= 1, \
            'Specify only one option: separate_datasets, round_times or interp_times'

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
    dfdict = {}
    for key,height in sonic_heights.items():
        sonic = data[key]
     
        # check data
        contents = dir(sonic)
        for output in sonic_outputs+['units']:
            assert output in contents, output+' missing in sonic data'
        
        # get units
        allunits = getattr(sonic,'units')
        sonic_units[time_name] = getattr(allunits,time_name)
        for output in sonic_outputs:
            units = getattr(allunits,output)
            if units in sonic_units.keys():
                assert units == sonic_units[output], 'mismatched sonic units'
            else:
                sonic_units[output] = units
        
        # get start time for timestamp conversion
        t0 = pd.to_datetime(getattr(sonic.units,time_name),
                            format='seconds from %Y-%m-%d %H:%M:%S UTC')
        #attrs['startdate'] = str(t0)  # 1970-01-01 00:00:00
        if sonic_starttime is None:
            sonic_starttime = t0
        else:
            assert t0 == sonic_starttime, 'mismatched sonic start time'
        
        # now, get the data
        df = None
        sonic_times = None
        offset = None # for rounding datetime index
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
            t = pd.DatetimeIndex(t0 + pd.to_timedelta(tseconds, unit='s'),
                                 name='datetime')

            if round_times or interp_times:
                # estimate sampling frequency
                all_dt = np.diff(tseconds)
                approx_freq = 1.0 / np.nanmean(all_dt)
                est_dt = 1.0 / np.round(approx_freq)
                if offset is None:
                    offset = '{:d}ms'.format(int(1e3*est_dt))
                else:
                    assert (offset == '{:d}ms'.format(int(1e3*est_dt)))
                if verbose:
                    print('Approx sampling frequency:',approx_freq)
                    print('  approx dt={:g} s, round to "{:s}" offset'.format(
                          est_dt, offset))
                maxdev = np.nanmax(np.abs((t - t.round(offset)).total_seconds().values))
                if verbose:
                    print('  max deviation from exact sampling time:',maxdev)
                assert (maxdev < est_dt/2)

            if sonic_times is None:
                # save times for datetime index
                sonic_times = t
                if verbose:
                    print('Datetime range:',t[0],t[-1])
            else:
                # sanity check: all channels for the output have the same times
                assert all(t == sonic_times), 'mismatched sonic times'

            # create dataframe if necessary
            outputchannel = getattr(sonic,output)
            if df is None:
                df = pd.DataFrame(index=sonic_times,dtype=outputchannel.dtype)
            df[output] = outputchannel

        if round_times or interp_times:
            if verbose:
                print('Resampling datetime index to',1.0/est_dt,'Hz')
                print('  original timeseries length:',len(df))
            if round_times:
                # save original times
                df['Time'] = tseconds.values # .values to ignore different indices
                # one-to-one mapping not guaranteed:
                df = df.resample(offset).nearest()
                df = df[~df.index.duplicated()]
                if verbose:
                    before,after = len(tseconds), len(np.unique(df['Time']))
                    if after < before:
                        print('  NOTE: number of unique times decreased:',
                              before,after)
            elif interp_times:
                newidx = pd.date_range(df.index[0].floor(offset),
                                       df.index[-1].ceil(offset),
                                       freq=offset, closed='left',
                                       name='datetime')
                if verbose:
                    print('Interpolating to',newidx)
                oldidx = df.index
                df = df.reindex(oldidx.union(newidx)).interpolate(method=interp_method)
                if newidx[0] < oldidx[0]:
                    # extrapolate
                    df.loc[newidx[0],:] = df.loc[oldidx[0],:]
                df = df.reindex(newidx)
            else:
                raise ValueError('Should not be here')
            if verbose:
                print('  new timeseries length:',len(df))

        df['height'] = height
        dfdict[key] = df
    
    if separate_datasets:
        dslist = {}
        for key,df in dfdict.items():
            # convert to xarray with metadata
            df = df.set_index('height',append=True)
            ds = df.to_xarray()

            # assign attributes
            ds.attrs = attrs
            if round_times:
                ds['Time'] = ds['Time'].assign_attrs(units=sonic_units['Time'])
            for output,desc in sonic_descriptions.items():
                ds[output] = ds[output].assign_attrs(description=desc,
                                                     units=sonic_units[output])
            dslist[key] = ds
        return dslist

    else:
        # convert to xarray with metadata
        df = pd.concat(dfdict.values())
        df = df.set_index('height',append=True).sort_index()
        ds = df.to_xarray()
        
        # assign attributes
        ds.attrs = attrs
        if round_times:
            ds['Time'] = ds['Time'].assign_attrs(units=sonic_units['Time'])
        for output,desc in sonic_descriptions.items():
            ds[output] = ds[output].assign_attrs(description=desc,
                                                 units=sonic_units[output])
        return ds


def convert_met_20Hz(fpath,
                     sensor_groups={
                         'BP': ['BP','Calc_BP','Calc_Rho'],
                         'TRH': ['Temp','RH'],
                         'cup': ['CupWS'],
                         'vane': ['VaneWD'],
                     },
                     description=None,
                     verbose=True):
    """
    Process 20-Hz met data from matlab file for instruments other than
    sonics
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
        dflist = []
        for prefix in outputs:
            # loop over all output fields for a given sensor
            # note: all fields in this group share the same height(s)
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
            df = df.stack(dropna=False)
            df.index.names = ['datetime','height_'+sensor]
            df.name = prefix
            dflist.append(df)
        df = pd.concat(dflist, axis=1)
        tmp = df.to_xarray()
        for prefix in outputs:
            ds[prefix] = tmp[prefix]

    # assign attributes
    ds.attrs = attrs
    for sensor,outputs in sensor_groups.items():
        for prefix in outputs:
            desc = '{:s} {:s}'.format(sensor_descriptions[sensor],
                                      other_descriptions[prefix])
            ds[prefix] = ds[prefix].assign_attrs(description=desc,
                                                 units=output_units[prefix])
    return ds


def convert_turbine(fpath,
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
    data = data['turbine']

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

    # check output channels, get units
    output_units = {}
    tmp = data._fieldnames
    tmp.remove(time_name)
    tmp.remove('units')
    for channel in dir(data):
        for output in tmp.copy():
            units = getattr(data.units,output)
            try:
                assert output_units[output] == units
            except KeyError:
                output_units[output] = units
            assert len(getattr(data,output)) == len(t)
            tmp.remove(output)
    outputs = list(output_units.keys())
    assert (len(tmp) == 0)

    # now read all the data
    df = pd.DataFrame(index=t)
    for output in outputs:
        if verbose:
            print('Processing',output)
        series = getattr(data,output)
        if np.all(pd.isna(series)):
            if verbose:
                print('WARNING:',output,'is all NaN')
        else:
            df[output] = series
    df.index.name = 'datetime'
    ds = df.to_xarray()

    # assign attributes
    ds.attrs = attrs
    for output in outputs:
        ds[output] = ds[output].assign_attrs(
                description=turbine_descriptions[output],
                units=output_units[output])
    return ds
