#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read data
inputfile = 'data/combined_radar_data.csv'
newfile = 'data/combined_radar_data_qc.csv'
removedfile = 'data/qc_removed_points.csv'
removed = []

global df
df = pd.read_csv(inputfile,parse_dates=['datetime'])
df.set_index('datetime').to_csv(inputfile+'.backup')

# save everything when we exit properly
def wrapup():
    try:
        pd.concat(removed).sort_index().to_csv(removedfile)
    except ValueError:
        print('No points removed')
    else:
        df.set_index('datetime').to_csv(newfile)
    sys.exit()

# setup plot
fig,ax = plt.subplots(ncols=2,sharex=True,sharey=True,figsize=(11,4),dpi=150)
style = dict(s=4,cmap='RdBu_r',vmin=-20,vmax=20)
ptmask = np.empty(len(df),dtype=bool)
ptmask.fill(True)
global uplot,vplot
uplot = ax[0].scatter(df['t_index'],df['height'],c=df['u'],**style)
vplot = ax[1].scatter(df['t_index'],df['height'],c=df['v'],**style)
def updateplot():
    global uplot,vplot,df
    uplot.remove()
    vplot.remove()
    uplot = ax[0].scatter(df['t_index'],df['height'],c=df['u'],**style)
    vplot = ax[1].scatter(df['t_index'],df['height'],c=df['v'],**style)
usel, = ax[0].plot([],[],'o',color=[0,1,0],markerfacecolor='none',markeredgewidth=2)
vsel, = ax[1].plot([],[],'o',color=[0,1,0],markerfacecolor='none',markeredgewidth=2)
fig.colorbar(uplot,ax=ax[0],label='u')
fig.colorbar(vplot,ax=ax[1],label='v')

# setup mouse handler
def onclick(event):
    global xsel,ysel,isel,point,df
    if event.button==3:
        # right click
        xsel,ysel = event.xdata, event.ydata
        r2 = (xsel-df['t_index'])**2 + ((ysel-df['height'])/10)**2
        isel = r2.idxmin()
        point = df.loc[[isel]]
        outstr = point.to_string(header=None)
        sys.stderr.write('({:g},{:g}) : {:s}\n'.format(xsel,ysel,outstr))
        # show selection circle
        usel.set_data(point['t_index'],point['height'])
        vsel.set_data(point['t_index'],point['height'])
        fig.canvas.draw()
fig.canvas.mpl_connect('button_press_event', onclick)

# setup keyboard handler
def onkey(event):
    global xsel,ysel,isel,point,df
    if event.key == 'enter':
        removed.append(point)
        df.drop(index=isel,inplace=True)
        updateplot()
        # hide selection circle
        usel.set_data([],[])
        vsel.set_data([],[])
        fig.canvas.draw()
    elif event.key == 'escape':
        wrapup()
fig.canvas.mpl_connect('key_press_event', onkey)

plt.show()

