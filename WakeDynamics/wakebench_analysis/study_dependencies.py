import os
import numpy as np
import pandas as pd


class Turbine(object):
    """Base class for turbine data"""
    def __init__(self):
        self.rotor_area = np.pi/4 * self.D**2

    def thrust(self,Uref):
        """Calculate reference thrust/density"""
        if not hasattr(self,'CTref'):
            raise ValueError('Thrust coefficient (CT) not defined')
        return self.CTref * 0.5 * Uref**2 * self.rotor_area

class CaseStudy(object):
    """Base class with common functions for studies"""
    def __init__(self,casedir,prefix,suffix='',turbine=None,dt=1.0):
        self.casedir = casedir
        self.prefix = prefix
        self.suffix = suffix
        self.turbine = turbine()
        self.dt = dt
        self.Navg = int(self.Tavg / dt)
        
    def _calc_offset(self,downD):
        dist = self.turbine.D * (downD - self.upstreamD)
        Tref = dist / self.Uref
        return int(Tref / self.dt)
        
    def trim_time(self,downD):
        """Input to dataloader, to synchronize the wake data and the upstream inflow data"""
        Noffset = self._calc_offset(downD)
        print('Calculated offset:',Noffset)
        return slice(self.Navg+Noffset-1,None)
        
    def get_turbine_datafile(self):
        fpath = os.path.join(self.casedir,'{:s}_wtg_response.txt'.format(self.prefix))
        if not os.path.isfile(fpath):
            raise OSError(fpath+' not found')
        print('Selected datafile:',fpath)
        return fpath
    
    def get_wake_datafile(self,loc):
        fpath = os.path.join(self.casedir,'{:s}_uvw_{:g}D{:s}.nc'.format(self.prefix,
                                                                         loc,
                                                                         self.suffix))
        if not os.path.isfile(fpath):
            raise OSError(fpath+' not found')
        print('Selected datafile:',fpath)
        return fpath
    
    def get_inflow(self,downD):
        """Pre-calculated with `estimate_inflow.ipynb`"""
        fpath = os.path.join(self.casedir,'inflow.npz')
        if not os.path.isfile(fpath):
            raise OSError(fpath+' not found')
        Uprofile = np.load(fpath)['U']
        Noffset = self._calc_offset(downD)
        return Uprofile[self.Navg-1:-Noffset,:]

    def get_outputs(self,name,downD,suffix=''):
        """Return dictionary with trajectory_file and outlines_file"""
        name = name.replace(' ','_').replace('(','').replace(')','')
        dpath = os.path.join(self.casedir,name+suffix)
        if not os.path.isdir(dpath):
            os.makedirs(dpath)
        tfile = 'trajectory_{:g}D.csv'.format(downD)
        ofile = 'outlines_{:g}D.pkl'.format(downD)
        outputs = {
            'trajectory_file': os.path.join(self.casedir,name+suffix,tfile),
            'outlines_file': os.path.join(self.casedir,name+suffix,ofile),
        }
        return outputs

