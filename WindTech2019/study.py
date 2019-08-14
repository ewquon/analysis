import numpy as np
from study_dependencies import Turbine, CaseStudy

class V27(Turbine):
    """Turbine data container"""
    D = 27.
    zhub = 32.1
    CTref = 0.62 # old openfast model


class neutral(CaseStudy):
    Uref = 8.7 # [m/s]
    zi = 750. # inversion height [m]
    TI = 10.7 # [%]
    alpha = 0.14 # [-]
    upstreamD = -2.5 # upstream distance, in rotor diameters
    downstreamD = range(2,9)
    Tavg = 60. # rolling window to estimate inflow profile [s]

