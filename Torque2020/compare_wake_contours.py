#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import scipy.io
from cmap import get_cmap

# --- Main params
CTlist = [0.40, 0.95]
suffixlist = ['noGE', 'GE']
desc = 'rev1_ds01R'
zhub = 1.5 #R

# --- function to load AD data
def loadAD(CT=0.4, HH=9.9, Lambda=99, Yaw=0):
    base='AD_CT0{:d}_HH{:02d}_Lambda{:02d}_Yaw{:02d}'.format(int(CT*100),int(HH*10),Lambda,-Yaw)
    CFDDataDir ='./'
    filename=base+'.mat'
    M=scipy.io.loadmat(filename)
    x=np.sort(np.unique(np.round(M['x'].ravel(),3)))
    z=np.sort(np.unique(np.round(M['z'].ravel(),3)))
    U=M['u']
    V=M['v']
    W=M['w']

    return z,x,W.T,U.T

def loadGP(CT=0.4,suffix=''):
    subdir = 'CT0{:02d}_{:s}'.format(int(CT*100), suffix)
    case = os.path.join(subdir,
            'Layout1x1_CT0{:02d}_{:s}_{:s}.npz'.format(int(CT*100),desc,suffix))
    d = np.load(case)
    x1 = d['x1']
    y1 = d['y1']
    z1 = d['z1']
    u0 = d['u0']
    u  = d['u']
    v  = d['v']
    w  = d['w']

    yslice = 0
    xx,zz = np.meshgrid(x1,(z1-zhub),indexing='ij')
    j = np.argmin(np.abs(yslice-y1))
    u0=u0[:,j,:]
    u = u[:,j,:]
    w = w[:,j,:]

    return xx,zz,u0,u,w

# --- 
fig,ax = plt.subplots(nrows=2, ncols=2,
                      sharex=True, sharey=True,
                      figsize=(16,8))
#cmap = 'magma'
cmap,valuesOri=get_cmap(0,1.05)
levelsLines = np.sort([1.05,1.0,0.99,0.98,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0])

for irow,suffix in enumerate(suffixlist):
    for icol,CT in enumerate(CTlist):
        xAD,zAD,uAD,wAD = loadAD(CT=CT)
        xGP,zGP,u0,uGP,wGP = loadGP(CT=CT,suffix=suffix)
        cm = ax[irow,icol].contourf(xAD,zAD,uAD,levels=levelsLines,cmap=cmap)
        cnt = ax[irow,icol].contour(xAD,zAD,uAD,levels=levelsLines,
                                    colors='k',linewidths=0.5)
        cnt2 = ax[irow,icol].contour(xGP,zGP,uGP,levels=levelsLines,
                                     colors='k',linewidths=0.5,linestyles='--')
        txt = ax[irow,icol].text(0.025,0.95,suffix,
                                 horizontalalignment='left',
                                 verticalalignment='top',
                                 transform=ax[irow,icol].transAxes)
for icol,CT in enumerate(CTlist):
    ax[0,icol].set_title('CT = {:g}'.format(CT),fontsize='xx-large')
    ax[-1,icol].set_xlabel('$x/R$ [-]',fontsize='x-large')
for axi in ax[:,0]:
    axi.set_ylabel('$z/R$ [-]',fontsize='x-large')
ax[0,0].set_xlim((-5,0))
ax[0,0].set_ylim((0,2))

figname = 'u_compare_contours'
fig.savefig(figname+'.png',bbox_inches='tight',dpi=150)
fig.savefig(figname+'.pdf',bbox_inches='tight',dpi=150)
plt.show()


