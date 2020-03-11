#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import scipy.io
from cmap import get_cmap

# --- Main params
CTlist = [0.40, 0.95]
suffix = '_rev1_ds01R_noGE'
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

def loadGP(CT=0.4):
    subdir = 'CT0{:02d}_{:s}'.format(int(CT*100), suffix.split('_')[-1])
    case = os.path.join(subdir,
            'Layout1x1_CT0{:02d}{:s}.npz'.format(int(CT*100),suffix))
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
NCT = len(CTlist)
fig,ax = plt.subplots(nrows=NCT, ncols=3,
                      sharex=True,sharey=True,
                      figsize=(11,NCT*1.25))
#cmap = 'magma'
cmap,valuesOri=get_cmap(0,1.05)
levelsLines = np.sort([1.05,1.0,0.99,0.98,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0])

for irow,CT in enumerate(CTlist):

    xAD,zAD,uAD,wAD = loadAD(CT=CT)
    xGP,zGP,u0,uGP,wGP = loadGP(CT=CT)

    cm = ax[irow,0].contourf(xAD,zAD,uAD,levels=levelsLines,cmap=cmap)
    cm = ax[irow,1].contourf(xGP,zGP,u0,levels=levelsLines,cmap=cmap)
    cm = ax[irow,2].contourf(xGP,zGP,uGP,levels=levelsLines,cmap=cmap)
    cnt = ax[irow,0].contour(xAD,zAD,uAD,levels=levelsLines,colors='k',linewidths=0.5)
    cnt = ax[irow,1].contour(xGP,zGP,u0,levels=levelsLines,colors='k',linewidths=0.5)
    cnt = ax[irow,2].contour(xGP,zGP,uGP,levels=levelsLines,colors='k',linewidths=0.5)

ax[0,0].set_title('actuator disk')
ax[0,1].set_title('Gauss')
ax[0,2].set_title('Gauss projection')
for axi in ax[-1,:]: axi.set_xlabel('$x/R$ [-]',fontsize='x-large')
for axi in ax[:,0]: axi.set_ylabel('$z/R$ [-]',fontsize='x-large')
for irow, axi in enumerate(ax[:,0]):
    axi.text(0.05,0.95, r'$C_T$ = {:g}'.format(CTlist[irow]),
             horizontalalignment='left', verticalalignment='top',
             transform=axi.transAxes)
ax[0,0].set_xlim((-3,3))
ax[0,0].set_ylim((0,1.5))

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.78])
cbar = fig.colorbar(cm, cax=cbar_ax)
#for lvl in ulevels:
#    cbar.ax.axhline(lvl,color=(0,1,0))
cbar.set_label('$u/U_0$ [-]',fontsize='x-large')

figname = 'u_compare_all_wakes_' + suffix.split('_')[-1]
fig.savefig(figname+'.png',bbox_inches='tight',dpi=150)
fig.savefig(figname+'.pdf',bbox_inches='tight',dpi=150)
plt.show()


