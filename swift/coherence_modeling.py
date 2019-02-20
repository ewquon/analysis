"""
Helper functions for coherence modeling, tested with TTU tower data at SWiFT

TODO: add this code to a2e-mmc/mmctools
"""
import numpy as np
from scipy.optimize import minimize

def Coh_model(f,r,zref=80.,Uref=8.0,a=12.,b=None,exp=0.):
    """General coherence model in TurbSim
    
    For IEC model: exp=0, Um=Uhub, a=(8.8, 12,...), and b=b(Lc) where
    Lc=Lc(zhub) is a coherence scale parameter.
    """
    if b is None:
        Lc = 5.67 * min(60, zref) # IEC 61400-1 3rd ed
        b = 0.12/Lc
    return np.exp(-a * (r/zref)**exp * np.sqrt((f*r/Uref)**2 + (b*r)**2))


def fit_IEC(coh,f,r,zref=74.7,Uref=8.0,weighting=None):
    """Obtains parameters a and b by solving an optimization problem."""
    guess = [12, 0.12/(5.67*60)]
    bounds = [(0,1000),(0,0.1)]
    if weighting is None:
        w = 1.0
    else:
        w = np.exp(-(f/weighting)**2)
    def errfun(params):
        a = params[0]
        b = params[1]
        x = Coh_model(f,r=r,zref=zref,Uref=Uref,a=a,b=b)
        err = w*(x - coh)
        return np.sqrt(err.dot(err))
    res = minimize(errfun,guess,method='SLSQP',bounds=bounds)
    if res.success:
        return res.x
    else:
        return None

def fit_general(coh,f,r,zref=74.7,Uref=8.0,adjustU=False,weighting=None):
    """Obtains parameters a, b, CohExp, (and optionally, Um) by solving
    an optimization problem.
    """
    guess = [12, 0, 0]
    bounds = [(0,1000), (0,0.1), (0,2)]
    if weighting is None:
        w = 1.0
    else:
        w = np.exp(-(f/weighting)**2)
    if adjustU:
        guess.append(Uref)
        bounds.append((0.5*Uref,1.5*Uref))
        def errfun(params):
            a = params[0]
            b = params[1]
            exp = params[2]
            U = params[3]
            x = Coh_model(f,r=r,zref=zref,Uref=U,a=a,b=b,exp=exp)
            err = w*(x - coh)
            return np.sqrt(err.dot(err))
    else:
        def errfun(params):
            a = params[0]
            b = params[1]
            exp = params[2]
            x = Coh_model(f,r=r,zref=zref,Uref=Uref,a=a,b=b,exp=exp)
            err = w*(x - coh)
            return np.sqrt(err.dot(err))        
    res = minimize(errfun,guess,method='SLSQP',bounds=bounds)
    if res.success:
        return res.x
    else:
        return None
