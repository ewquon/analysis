import numpy as np
import matplotlib.colors as mcolors
import itertools


def make_colormap(seq,values=None):
    """Return a LinearSegmentedColormap
    seq: RGB-tuples. 
    values: corresponding values (location betwen 0 and 1)
    """
    n=len(seq)
    if values is None:
        values=np.linspace(0,1,n)

    doubled     = list(itertools.chain.from_iterable(itertools.repeat(s, 2) for s in seq))
    doubled[0]  = (None,)* 3
    doubled[-1] = (None,)* 3
    cdict = {'red': [], 'green': [], 'blue': []}
    for i,v in enumerate(values):
        r1, g1, b1 = doubled[2*i]
        r2, g2, b2 = doubled[2*i + 1]
        cdict['red'].append([v, r1, r2])
        cdict['green'].append([v, g1, g2])
        cdict['blue'].append([v, b1, b2])
    #print(cdict)
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def get_cmap(minSpeed,maxSpeed):
    DS=0.001
    # MathematicaDarkRainbow=[(60 /255,86 /255,146/255), (64 /255,87 /255,142/255), (67 /255,107/255,98 /255), (74 /255,121/255,69 /255), (106/255,141/255,61 /255), (159/255,171/255,67 /255), (207/255,195/255,77 /255), (223/255,186/255,83 /255), (206/255,128/255,76 /255), (186/255,61 /255,58 /255)]

    #     ManuDarkOrange  = np.array([198 ,106,1   ])/255.;
    #     ManuLightOrange = np.array([255.,212,96  ])/255.;
    # (1,212/255,96/255),  # Light Orange
    # (159/255,159/255,204/255), # Light Blue
    #     MathematicaLightGreen = np.array([158,204,170 ])/255.;
    # (159/255,159/255,204/255), # Light Blue
    seq=[
    (63/255 ,63/255 ,153/255), # Dark Blue
    (159/255,159/255,204/255), # Light Blue
    (158/255,204/255,170/255), # Light Green
    (1,212/255,96/255),  # Light Orange
    (1,1,1),  # White
    (1,1,1),  # White
    (1,1,1),  # White
    (138/255 ,42/255 ,93/255), # DarkRed
    ]
    valuesOri=np.array([
    minSpeed,  # Dark Blue
    0.90,
    0.95,
    0.98,
    1.00-DS , # White
    1.00    , # White
    1.00+DS , # White
    maxSpeed         # DarkRed
    ])
    values=(valuesOri-min(valuesOri))/(max(valuesOri)-min(valuesOri))

    valuesOri=np.around(valuesOri[np.where(np.diff(valuesOri)>DS)[0]],2)

    cmap= make_colormap(seq,values=values)
    return cmap,np.concatenate((valuesOri,[maxSpeed]))

cmap,valuesOri=get_cmap(0.5,1.03)
print(cmap)

