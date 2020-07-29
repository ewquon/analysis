#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys

import warnings
# suppress FutureWarning:
#/Users/equon/anaconda3/envs/forge/lib/python3.7/site-packages/pyamg/gallery/stencil.py:110: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
#  diag[s] = 0
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('/Users/equon/floris')
import floris.tools as wfct
from floris.utilities import Vec3  # to specify grid resolution

import pyamg

import os
if not os.path.isdir('figures'):
    os.makedirs('figures')


#case = sys.argv[1]
case = 'Layout_5x5.json'

# HARD-CODED PARAMS
debug = False
tol = 1e-6
savefloris = True

xrange = (-10, 10) # rotor radii
buf = 10  # lateral/vertical extent, in rotor radii
#spacing = 0.4 # grid spacing, in rotor radii
#spacing = 0.2 # grid spacing, in rotor radii
spacing = 0.1 # grid spacing, in rotor radii
groundeffect = False

# u contour levels to plot
ulevels = [0.7, 0.8, 0.9, 0.95, 0.98, 0.99]


#
# Begin calculations
#
fi = wfct.floris_interface.FlorisInterface(case)

# get turbine properties from floris inputs
zhub = fi.floris.farm.turbine_map.turbines[0].hub_height
R = fi.floris.farm.turbine_map.turbines[0].rotor_diameter / 2

# setup domain
yrange = (-1-buf, 1+buf)
if groundeffect:
    #zrange = (0, zhub+1+buf)
    zrange = (spacing, zhub+1+buf) # our domain does _not_ include boundary points
else:
    zrange = (zhub-1-buf, zhub+1+buf)
Nx = int((xrange[1]-xrange[0]) / spacing) + 1
Ny = int((yrange[1]-yrange[0]) / spacing) + 1
Nz = int((zrange[1]-zrange[0]) / spacing) + 1
N = Nx*Ny*Nz
print('Nx,Ny,Nz =',Nx,Ny,Nz)
x1 = (np.arange(Nx)*spacing + xrange[0]) * R
y1 = (np.arange(Ny)*spacing + yrange[0]) * R
z1 = (np.arange(Nz)*spacing + zrange[0]) * R


print('Calculating floris wake')

fi.floris.farm.flow_field.reinitialize_flow_field(
    with_resolution=Vec3(Nx,Ny,Nz),
    bounds_to_set=list(xrange)+list(yrange)+list(zrange),
)
fi.calculate_wake()


print('Setting up poisson system (N={:d})'.format(N))

A = pyamg.gallery.poisson((Nx,Ny,Nz), format='csr')

# RHS
u0 = fi.floris.farm.flow_field.u
du0_dx = np.empty(u0.shape)
du0_dx[1:-1,:,:] = (u0[2:,:,:] - u0[:-2,:,:]) / (2*spacing)
du0_dx[0,:,:] = (u0[1,:,:] - u0[0,:,:]) / spacing
du0_dx[-1,:,:] = (u0[-1,:,:] - u0[-2,:,:]) / spacing
b = -du0_dx.ravel()

# enforce compatibility conditions for neumann
b -= np.mean(b)
if debug:
    print('sum RHS =',np.sum(b))


print('Enforcing neumann conditions')
def modify_diag(i=slice(None),j=slice(None),k=slice(None)):
    if not i==slice(None):
        print('- setting i={:d} to Neumann'.format(i))
    if not j==slice(None):
        print('- setting j={:d} to Neumann'.format(j))
    if not k==slice(None):
        print('- setting k={:d} to Neumann'.format(k))
    inplane = np.empty((Nx,Ny,Nz),dtype=bool)
    inplane[:] = False
    inplane[i,j,k] = True
    indices = np.arange(N)[inplane.ravel()]
    A[indices,indices] -= 1
modify_diag(i=0) # upstream
modify_diag(i=-1) # downstream
modify_diag(j=0) # -side
modify_diag(j=-1) # +side
modify_diag(k=0) # lower
modify_diag(k=-1) # upper

A = A / spacing**2

def set_ref(i,j,k,refval=0):
    """Not sure if you'd evere want to pick refval not 0..."""
    print('Setting reference point at ({:g}, {:g}, {:g}) to {:g}'.format(
            x1[i],y1[j],z1[k],refval))
    idx = np.ravel_multi_index((i,j,k),(Nx,Ny,Nz))
    row = A.getrow(idx)
    for j in row.indices:
        # zero all off-diagonal entries
        if j != idx:
            A[idx,j] = 0
    A[idx,idx] = 1
    b[idx] = refval
set_ref(0,0,0)


print('Solving poisson system')

#x,flag = pyamg.krylov.cg(A, b, tol=tol)
#assert flag == 0

ml = pyamg.smoothed_aggregation_solver(A)
resid = []
x = ml.solve(b, tol=tol, accel='cg', residuals=resid)
print('L2 norm of residual:',np.linalg.norm(b - A.dot(x)))

if debug:
    print(ml)
    # A_i: operator on level i (0 is finest)
    # P_i: prolongation operator, mapping from level i+1 to i
    for lvl in range(len(ml.levels)-1):
        print("A_{}: {}x{}   P_{}: {}x{}".format(lvl, ml.levels[lvl].A.shape[0], ml.levels[lvl].A.shape[1],
                                                 lvl, ml.levels[lvl].P.shape[0], ml.levels[lvl].P.shape[1]))
    print("A_{}: {}x{}".format(lvl, ml.levels[-1].A.shape[0], ml.levels[-1].A.shape[1]))

resid = np.array(resid)
fig,ax = plt.subplots()
ax.semilogy(resid/resid[0],'o-')
ax.axhline(tol,color='k',ls='--')
fig.savefig('convergence_history.png',bbox_inches='tight')


print('Postprocessing')

p = x.reshape((Nx,Ny,Nz))
if debug:
    pmean = np.mean(p)
    print('p offset',pmean)

dp_dx = np.empty(u0.shape)
dp_dx[1:-1,:,:] = (p[2:,:,:] - p[:-2,:,:]) / (2*spacing)
dp_dx[0,:,:] = (p[1,:,:] - p[0,:,:]) / spacing
dp_dx[-1,:,:] = (p[-1,:,:] - p[-2,:,:]) / spacing

dp_dy = np.empty(u0.shape)
dp_dy[:,1:-1,:] = (p[:,2:,:] - p[:,:-2,:]) / (2*spacing)
dp_dy[:,0,:] = (p[:,1,:] - p[:,0,:]) / spacing
dp_dy[:,-1,:] = (p[:,-1,:] - p[:,-2,:]) / spacing

dp_dz = np.empty(u0.shape)
dp_dz[:,:,1:-1] = (p[:,:,2:] - p[:,:,:-2]) / (2*spacing)
dp_dz[:,:,0] = (p[:,:,1] - p[:,:,0]) / spacing
dp_dz[:,:,-1] = (p[:,:,-1] - p[:,:,-2]) / spacing

if debug:
    print('max slope errors at boundaries:')
    print(np.max(np.abs(dp_dx[0,:,:])))
    print(np.max(np.abs(dp_dx[-1,:,:])))
    print(np.max(np.abs(dp_dy[:,0,:])))
    print(np.max(np.abs(dp_dy[:,-1,:])))
    print(np.max(np.abs(dp_dz[:,:,0])))
    print(np.max(np.abs(dp_dz[:,:,-1])))

u = u0 - dp_dx
v = -dp_dy
w = -dp_dz


if case.endswith('.json'):
    case = os.path.split(case)[1]
    case = case[:-len('.json')]
case += '_ds{:s}R'.format(str(spacing).replace('.',''))
if groundeffect:
    case += '_GE'
else:
    case += '_noGE'

datafile = case+'.npz'
outputdata = dict(x1=x1,y1=y1,z1=z1,
                  u=u,v=v,w=w,p=p)
if savefloris:
    outputdata['u0'] = u0
np.savez_compressed(datafile,**outputdata)
print('Wrote',datafile)


#
# Make plots
#
x1n = (np.arange(Nx+1)*spacing + xrange[0] - spacing/2) * R
y1n = (np.arange(Ny+1)*spacing + yrange[0] - spacing/2) * R
z1n = (np.arange(Nz+1)*spacing + zrange[0] - spacing/2) * R

# y slices

yslice = 0
j = int((yslice-yrange[0])/spacing)
xx,zz = np.meshgrid(x1,z1-zhub,indexing='ij')
xxn,zzn = np.meshgrid(x1,z1n-zhub,indexing='ij') # nodes for pcolormesh

fig,ax = plt.subplots(figsize=(11,3))
cm = ax.pcolormesh(xxn,zzn,u[:,j,:],cmap='viridis',vmin=0.7,vmax=1)
cont = ax.contour(xx,zz,u[:,j,:], levels=ulevels, colors='k', linewidths=0.5)
cbar = fig.colorbar(cm)
cbar.set_label('$u/U_0$ [-]',fontsize='x-large')
ax.set_xlabel('$x/R$ [-]',fontsize='x-large')
ax.set_ylabel('$z/R$ [-]',fontsize='x-large')
ax.set_xlim((-5,0))
#ax.set_ylim((0,2))
ax.set_ylim((-2,2))
fig.savefig('figures/u_{:s}.png'.format(case),bbox_inches='tight')

fig,ax = plt.subplots(figsize=(11,3))
cm = ax.pcolormesh(xxn,zzn,w[:,j,:],cmap='bwr',vmin=-0.1,vmax=0.1)
cbar = fig.colorbar(cm)
cbar.set_label('$w/U_0$ [-]',fontsize='x-large')
ax.set_xlabel('$x/R$ [-]',fontsize='x-large')
ax.set_ylabel('$z/R$ [-]',fontsize='x-large')
ax.set_xlim((-5,0))
#ax.set_ylim((0,2))
ax.set_ylim((-2,2))
fig.savefig('figures/w_{:s}.png'.format(case),bbox_inches='tight')

#fig,ax = plt.subplots(figsize=(11,3))
#cm = ax.pcolormesh(xxn,zzn,p[:,j,:],cmap='bwr')
#cbar = fig.colorbar(cm)
#cbar.set_label('$\lambda$ [-]',fontsize='x-large')
#ax.set_xlabel('$x/R$ [-]',fontsize='x-large')
#ax.set_ylabel('$z/R$ [-]',fontsize='x-large')
#fig.savefig('figures/pc_rotorxz_{:s}.png'.format(case),bbox_inches='tight')

# x slices

#xslice = 0
#i = int((xslice-xrange[0])/spacing)
#yy,zz = np.meshgrid(y1,z1-zhub,indexing='ij')
#yyn,zzn = np.meshgrid(y1n,z1n-zhub,indexing='ij') # nodes for pcolormesh

#fig,ax = plt.subplots(figsize=(11,3))
#cm = ax.pcolormesh(yyn,zzn,p[i,:,:],cmap='bwr')
#cbar = fig.colorbar(cm)
#cbar.set_label('$\lambda$ [-]',fontsize='x-large')
#ax.set_xlabel('$y/R$ [-]',fontsize='x-large')
#ax.set_ylabel('$z/R$ [-]',fontsize='x-large')
#fig.savefig('figures/pc_rotoryz_{:s}.png'.format(case),bbox_inches='tight')

# z slices

#zslice = zhub
#k = int((zslice-zrange[0])/spacing)
#xx,yy = np.meshgrid(x1,y1,indexing='ij')
#xxn,yyn = np.meshgrid(x1n,y1n,indexing='ij') # nodes for pcolormesh

#fig,ax = plt.subplots(figsize=(11,3))
#cm = ax.pcolormesh(xxn,yyn,p[:,:,k],cmap='bwr')
#cbar = fig.colorbar(cm)
#cbar.set_label('$\lambda$ [-]',fontsize='x-large')
#ax.set_xlabel('$x/R$ [-]',fontsize='x-large')
#ax.set_ylabel('$y/R$ [-]',fontsize='x-large')
#fig.savefig('figures/pc_rotorxy_{:s}.png'.format(case),bbox_inches='tight')

