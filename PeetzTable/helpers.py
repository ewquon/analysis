import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

from weio import FASTInputFile


# from Chris 2020-06-09
def excel2datetime(excel_num,round_to=None):
    """
    conversion of matlab datenum format to python datetime
 
    Parameters:
    ----------------
    matlab_datenum : np.array
        array of matlab datenum to be converted
 
    Returns:
    -----------------
    time : DateTimeIndex
        array of corresponding python datetime values
    """
    # check data types
    try:
        excel_num = np.array(excel_num)
    except:
        pass
    assert isinstance(excel_num, np.ndarray), 'data must be of type np.ndarray'
 
    # convert to datetime
    time = pd.to_datetime('1899-12-30')+pd.to_timedelta(excel_num,'D')
 
    return time

def extract_from_mat(fpath,prefix=None,stats='mean10min',name='data_out',local_to_UTC=None):
    """
    Extract data from a matlab datafile and return a dataframe
    
    Parameters
    ----------
    fpath : str
        Filepath to .mat file
    prefix : str or None
        Prefix corresponding to data channels to be extracted, e.g.,
        "T2" for turbine 2; set to None to simply return list of all
        available channels
    stats : str
        One of ['scaled_data', 'mean10min', 'max10min', 'min10min', 'stdev10min']
        where 'data' means that the raw data should be returned
    name : str or None
        Name of structure within matlab file to extract data from
    local_to_UTC : None or timedelta
        Timedelta to convert timestamps
    """
    mat = loadmat(fpath, struct_as_record=False, squeeze_me=True)
    if name is not None:
        mat = mat[name]
    if prefix is not None:
        channels = [channel for channel in mat._fieldnames if channel.startswith(prefix)]
        if len(channels) == 0:
            print('No channels starting with ',prefix)
    if (prefix is None) or (len(channels) == 0):
        return mat._fieldnames
    if stats in ['data','scaled_data']:
        datetime = excel2datetime(getattr(mat.MS_Excel_Timestamp,'data'))
        if local_to_UTC is not None:
            datetime += local_to_UTC
        df = pd.DataFrame(index=datetime,columns=channels)
    else:
        datetime = excel2datetime(getattr(mat.MS_Excel_Timestamp,'min10min'))
        if local_to_UTC is not None:
            datetime += local_to_UTC
        df = pd.DataFrame(index=[datetime],columns=channels)
    for channel in channels:
        try:
            df[channel] = getattr(getattr(mat,channel),stats)
        except AttributeError:
            print('Error retrieving',stats,'from channel',channel)
    return df


def calc_rotor_mass(inpfile,r0,r1):
    inp = FASTInputFile(inpfile)
    bladeprops = inp['BldProp'] # BlFract, PitchAxis, StrcTwst, BMassDen, FlpStff, EdgStff
    r = r0 + bladeprops[:,0]*(r1-r0)
    blademass = np.trapz(bladeprops[:,3], x=r)
    return 3*blademass


class ThrustEstimator(object):
    """Estimate thrust from measured moments and properties defined by
    an OpenFAST model
    """
    def __init__(self,fstfile,rotormass=None,g=9.81,upwind=True):
        self.modelpath = os.path.split(fstfile)[0]
        self.g = g
        self.upwind = upwind
        self.TowerBaseMoment = None
        self.RotorAeroMoment = None
        print('Using OpenFAST model in',self.modelpath)
        self.fst = FASTInputFile(fstfile)
        elastodyn_input = os.path.join(self.modelpath, self.fst['EDFile'].strip('"'))
        aerodyn_input = os.path.join(self.modelpath, self.fst['AeroFile'].strip('"'))
        print('- reading',elastodyn_input)
        self.elastodyn = FASTInputFile(elastodyn_input)
        elastodyn_tower_input = os.path.join(self.modelpath, self.elastodyn['TwrFile'].strip('"'))
        self._get_elastodyn_properties(rotormass)
        print('- reading',elastodyn_tower_input)
        self.elastodyn_tower = FASTInputFile(elastodyn_tower_input)
        self._get_elastodyn_tower_properties()
        print('- reading',aerodyn_input)
        self.aerodyn = FASTInputFile(aerodyn_input)
        self._get_aerodyn_properties()

    def _get_elastodyn_properties(self,rotormass=None):
        self.HubRad = self.elastodyn['HubRad']
        self.TipRad = self.elastodyn['TipRad']
        self.TowerHt = self.elastodyn['TowerHt']
        self.Twr2Shft = self.elastodyn['Twr2Shft']
        self.TwrBsHt = self.elastodyn['TwrBsHt']
        self.Overhang = self.elastodyn['Overhang']
        if self.upwind:
            self.Overhang = -self.Overhang
        self.ShftGagL = self.elastodyn['ShftGagL']
        self.HubCM = self.elastodyn['HubCM']
        self.NacCMxn = self.elastodyn['NacCMxn']
        self.ShftTilt = self.elastodyn['ShftTilt']
        if self.upwind:
            self.ShftTilt = -self.ShftTilt
        self.ShftTilt = np.radians(self.ShftTilt)
        self.HubMass = self.elastodyn['HubMass']
        self.NacMass = self.elastodyn['NacMass']
        # Read/calculate rotor mass
        if rotormass is None:
            elastodyn_blade_input = os.path.join(self.modelpath, self.elastodyn['BldFile(1)'].strip('"'))
            print('- reading',elastodyn_blade_input)
            self.RotorMass = calc_rotor_mass(elastodyn_blade_input,
                                             r0=self.HubRad, r1=self.TipRad)
            print('  calculated rotor mass = ',self.RotorMass)
        else:
            self.RotorMass = rotormass
        # calculated quantities
        self.zhub = self.TowerHt - self.TwrBsHt + self.Twr2Shft
        self.Wrotor = self.RotorMass * self.g / 1000. # [kN]
        self.Whub = self.HubMass * self.g / 1000. # [kN]
        self.Wnac = self.NacMass * self.g / 1000. # [kN]
        self.HubCMxn = self.Overhang - self.HubCM

    def _get_elastodyn_tower_properties(self):
        self.towerprops = pd.DataFrame(self.elastodyn_tower['TowProp'],
                                       columns=['HtFract','TMassDen','TwFAStif','TwSSStif'])
        assert self.towerprops.iloc[0]['HtFract'] == 0
        assert self.towerprops.iloc[-1]['HtFract'] == 1
        self.towerprops['height'] = self.towerprops['HtFract'] * self.zhub
        self.towerprops = self.towerprops.set_index('height')
        self.EI = self.towerprops['TwFAStif']

    def _get_aerodyn_properties(self):
        self.AirDens = self.aerodyn['AirDens']
        self.TowerAeroMoment = 0.

    def set_towerbase_foreaft_moment(self,foreaft_moment,height_from_base=0):
        """Corresponding to measured fore-aft bending moment [kN-m] at
        or near the tower base, at a height relative to TwrBsHt.
        """
        self.TowerBaseMoment = foreaft_moment
        self.height_from_base = height_from_base

    def set_mainshaft_pitching_moment(self,pitching_moment):
        """Corresponding to measured pitching moment [kN-m] at location
        ShftGagL; this is used to calculate the pitching moment due to
        rotor aerodynamics
        """
        self.RotorAeroMoment = (
            pitching_moment
            + self.Wrotor * self.ShftGagL * np.cos(self.ShftTilt) 
            + self.Whub * (self.ShftGagL-self.HubCM) * np.cos(self.ShftTilt)
        )

    def set_tower_drag_from_inflowwind(self):
        """Read steady inflow profile from InflowWind input and
        calculate resulting constant aerodynamic bending moment due to
        drag
        """
        inflowwind_input = os.path.join(self.modelpath, self.fst['InflowFile'].strip('"'))
        inf = FASTInputFile(inflowwind_input)
        assert (inf['WindType'] == 1), 'InflowWind WindType '+str(inf['WindType'])+' not supported'
        Uref = inf['HWindSpeed']
        zref = inf['RefHt']
        shear = inf['PLexp']
        print(f'Reference wind profile: U(z={zref})={Uref} with alpha={shear}')
        self.set_tower_drag(Uref,shear,zref)
                                  
    def set_tower_drag(self,Uref,shear=0.2,zref=80.):
        """Calculate aerodynamic bending moment due to drag, either due
        to:
        1) a steady power-law wind profile dictated by `Uref`, `shear`,
           and `zref`;
        2) a steady wind profile as dictated by `Uref` and `zref`,
           where both are array-like; or
        3) a time-varying wind profile as dictated by Uref(t,z) and
           `zref` where the first dimension of `Uref` should match the
           length of thet specified tower-base fore-aft moment time-
           series and the second dimension of `Uref` should match the
           length of `zref`.
        If a steady or time-varying profile is provided, it should
        span the bottom of the rotor up to hub height.
        """
        self.TowerAeroMoment = np.zeros_like(self.TowerBaseMoment)
        self.toweraero = pd.DataFrame(self.aerodyn['TowProp'],
                                      columns=['TwrElev','TwrDiam','TwrCd'])
        self.toweraero = self.toweraero.set_index('TwrElev')
        z = self.toweraero.index.values
        issteady = True
        if isinstance(Uref,(float,int)):
            # power-law profile
            print('Calculating drag from power-law profile')
            self.toweraero['Uinf'] = Uref * (z/zref)**shear
        elif len(Uref.shape) == 1:
            # steady profile
            print('Calculating drag from steady profile')
            assert (len(Uref) == len(zref)), 'specify zref to be heights corresponding to Uref'
            self.toweraero['Uinf'] = np.interp(z, zref, Uref)
        else:
            # time-varying profile
            print('Calculating drag from time-varying profile')
            assert (len(Uref.shape) == 2), 'specify Uref to be U(t,z)'
            assert (Uref.shape[0] == len(self.TowerBaseMoment)), \
                    'the first dimension of Uref should correspond to the tower-base moment time series'
            assert (Uref.shape[1] == len(zref)), \
                    'specify zref to be heights corresponding to Uref'
            if isinstance(Uref,pd.DataFrame):
                Uref = Uref.values
            issteady = False
        if issteady:
            self.toweraero['TwrSectionalDrag'] = self.toweraero['TwrCd'] \
                    * 0.5 * self.AirDens * self.toweraero['Uinf']**2 * self.toweraero['TwrDiam']
            M = np.trapz(self.toweraero['TwrSectionalDrag']*z, x=z)
            self.TowerAeroMoment[:] = M / 1000 # [kN]
        else:
            for i,ti in enumerate(self.TowerBaseMoment.index):
                Uinf = np.interp(z, zref, Uref[i,:])
                sectdrag = self.toweraero['TwrCd'] \
                    * 0.5 * self.AirDens * Uinf**2 * self.toweraero['TwrDiam']
                self.TowerAeroMoment[i] = np.trapz(sectdrag*z, x=z)
            self.TowerAeroMoment[:] /= 1000 # [kN]
        self.TowerAeroMoment = pd.Series(self.TowerAeroMoment,
                                         index=self.TowerBaseMoment.index,
                                         name='calculated tower aerodynamic moment [kN-m]')

    def _estimate_thrust(self,towertop_deflection=0):
        assert (self.TowerBaseMoment is not None), 'Need to set_towerbase_foreaft_moment()'
        assert (self.RotorAeroMoment is not None), 'Need to set_mainshaft_pitching_moment()'
        thrust = (self.TowerBaseMoment - self.RotorAeroMoment
                + self.Wrotor * (self.Overhang - towertop_deflection) * np.cos(self.ShftTilt)
                + self.Whub * (self.HubCMxn - towertop_deflection) * np.cos(self.ShftTilt)
                - self.Wnac * (self.NacCMxn + towertop_deflection)
                - self.TowerAeroMoment
               ) / ((self.zhub-self.height_from_base) * np.cos(self.ShftTilt))
        thrust = pd.Series(thrust, index=self.TowerBaseMoment.index,
                           name='calculated rotor aerodynamic thrust [kN]')
        return thrust

    def _estimate_towertop_deflection(self,thrust,avg_seconds=60):
        if isinstance(thrust.index, (pd.TimedeltaIndex, pd.DatetimeIndex)):
            last1min = (thrust.index >
                    (thrust.index[-1] - pd.to_timedelta(int(avg_seconds),unit='s')))
            meanthrust = thrust.loc[last1min].mean()
            meanmoment = self.RotorAeroMoment.loc[last1min].mean()
        else:
            Navg = int(float(avg_seconds) / self.fst['DT_Out'])
            meanthrust = np.mean(thrust.iloc[-Navg:])
            meanmoment = np.mean(self.RotorAeroMoment.iloc[-Navg:])
        z = self.EI.index
        M_EI = np.zeros_like(self.EI.index)
        # calculate M(x)/EI(x)
        for i,zi in enumerate(z):
            M = meanmoment + (self.zhub - zi)*meanthrust # [kN-m]
            M_EI[i] = 1000 * M / self.EI.loc[zi] # convert to [N-m]
        # integrate M/EI, 0 .. z for all z
        intM_EI = np.array([np.trapz(M_EI[:i+1], x=z[:i+1]) for i in range(len(z))])
        # integrate again, 0 .. zhub to get the tower-top deflection
        dx = np.trapz(intM_EI, x=z)
        return dx
        
    def calc(self,elastic=False,tol=1e-3,maxiter=10):
        """Estimate the rotor thrust (along the main shaft)

        If `elastic` is True, then estimate the tower-top deflection
        using classical beam theory. This uses an iterative approach
        to estimate the rotor thrust and deflection simultaneously.
        `tol` and `maxiter` are used to control the iterations.
        """
        thrust = self._estimate_thrust()
        if elastic:
            lastval = thrust.iloc[-1]
            print(f'iteration 0 : T={lastval:f} kN')
            i = 0
            while i < 10:
                i += 1
                TTdefl = self._estimate_towertop_deflection(thrust)
                thrust = self._estimate_thrust(towertop_deflection=TTdefl)
                reltol = np.abs(thrust.iloc[-1]-lastval) / thrust.iloc[-1]
                lastval = thrust.iloc[-1]
                print(f'iteration {i:d} : T={lastval:f} kN, deflection={TTdefl:f} m, reltol={reltol:g}')
                if reltol < tol:
                    break
            return thrust, TTdefl
        else:
            return thrust

    def plot_inputs(self):
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(nrows=3,sharex=True,figsize=(8,8))
        self.TowerBaseMoment.plot(ax=ax[0])
        self.RotorAeroMoment.plot(ax=ax[1])
        self.TowerAeroMoment.plot(ax=ax[2])
        ax[0].set_ylabel('INPUT\ntotal tower-base\nfore-aft moment\n[kN-m]')
        ax[1].set_ylabel('INPUT\nmain-shaft\npitching moment\n[kN-m]')
        ax[2].set_ylabel('CALCULATED\ntower-base\naerodynamic moment\n[kN-m]')

