import sys,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def moving_median(series, window=3):
    return series.rolling(window, center=True).median()

class Trajectory(object):
    def __init__(self,case,method):
        self.case = case
        self.method = method
        self.D = self.case.turbine.D
        self._read_trajectory_files()

    def _read_trajectory_files(self):
        """Read a collection of trajectory files with a common prefix
        and store data in a dataframe for processing.
        """
        dflist = []
        self.Ntimes = {}
        for downD in self.case.downstreamD:
            outputs = self.case.get_outputs(self.method,downD)
            print(outputs['trajectory_file'])
            df = pd.read_csv(outputs['trajectory_file'],
                             header=None,
                             usecols=[0,1,2])
            df.columns = ['t','y','z']
            df['x'] = downD * self.case.turbine.D
            df['z'] -= self.case.turbine.zhub
            df = df.set_index(['t','x'])[['y','z']]
            self.Ntimes[downD] = len(df.index.levels[0])
            dflist.append(df)
        self.df = pd.concat(dflist).sort_index()

    def write_trajectory_files(self,suffix='--filtered'):
        """Write collection of trajectory files to a new directory.
        """
        for downD in self.case.downstreamD:
            xi = downD * self.case.turbine.D
            inputs = self.case.get_outputs(self.method,downD)
            outputs = self.case.get_outputs(self.method,downD,suffix=suffix)
            print(outputs['trajectory_file'])
            df = pd.read_csv(inputs['trajectory_file'],
                             header=None)
            # should have at least 3 columns
            # 0: time, 1: ywake, 2: zwake
            newdf = self.df.xs(xi, level='x').iloc[:self.Ntimes[downD]]
            assert (len(newdf) == len(df))
            notna = ~pd.isna(newdf['y'])
            print('updated',np.count_nonzero(notna),'/',len(newdf),'at x=',xi)
            df.loc[notna,1] = newdf.loc[notna,'y']
            df.loc[notna,2] = newdf.loc[notna,'z'] + self.case.turbine.zhub
            df.to_csv(outputs['trajectory_file'],
                      header=None,index=None)


    def identify_outliers(self,df,yrange,zrange,edgebuffer=1.0):
        zrange = np.array(zrange)
        zrange -= self.case.turbine.zhub
        leftedge   = (df['y'] < yrange[0]+edgebuffer)
        rightedge  = (df['y'] > yrange[1]-edgebuffer)
        bottomedge = (df['z'] < zrange[0]+edgebuffer)
        topedge    = (df['z'] > zrange[1]-edgebuffer)
        is_y_outlier = (leftedge | rightedge)
        is_z_outlier = (bottomedge | topedge)
        yout = df.loc[is_y_outlier]
        zout = df.loc[is_z_outlier]
        for xi in df.index.levels[1]:
            try:
                Nouty = len(yout.xs(xi, level='x'))
            except KeyError:
                Nouty = 0
            try:
                Noutz = len(zout.xs(xi, level='x'))
            except KeyError:
                Noutz = 0
            print('x=',xi,'outliers in y/z :',Nouty,Noutz)
        return is_y_outlier, is_z_outlier

    def remove_outliers(self,yrange,zrange,edgebuffer=1.0):
        yout,zout = self.identify_outliers(self.df,yrange,zrange,edgebuffer)
        self.df.loc[yout,'y'] = np.nan
        self.df.loc[zout,'z'] = np.nan

    def _interpolate(self,method='linear'):
        unstacked = self.df.unstack()
        unstacked.interpolate(method=method,inplace=True)
        self.df = unstacked.stack(dropna=False)

    def _filter(self,method=moving_median,**kwargs):
        unstacked = self.df.unstack()
        unstacked = method(unstacked,**kwargs)
        self.df = unstacked.stack(dropna=False)

    def filter(self,yrange,zrange,edgebuffer=1.0,
               interp='linear',
               method=moving_median,**filter_kwargs):
        """Convenience function for performing all QC operations"""
        self.df0 = self.df.copy()
        self.remove_outliers(yrange,zrange,edgebuffer)
        unstacked = self.df.unstack()
        unstacked.interpolate(method=interp,inplace=True)
        unstacked = method(unstacked,**filter_kwargs)
        self.df = unstacked.stack(dropna=False)

    def rms_error(self,df,ref):
        assert (len(df)==len(ref))
        valid = np.where(np.isfinite(ref) & np.isfinite(df))
        err = df.values[valid] - ref[valid]
        return np.sqrt(np.mean(err**2))

    def plot_wake_hist(self,wakedir,downstreamD=None,
                       norm=False,annotate=True,label='',
                       applyfilter=None,
                       ref=None, # for calculating rms error
                       fig=None,ax=None,
                       **kwargs):
        if downstreamD is None:
            # select all distances
            downstreamD = self.case.downstreamD
        elif not hasattr(downstreamD, '__iter__'):
            downstreamD = [downstreamD]
        if (fig is None) or (ax is None):
            fig,ax = plt.subplots(nrows=len(downstreamD), sharex=True,
                                  figsize=(8,len(downstreamD)*3))
        if len(downstreamD) == 1:
            ax = [ax]
        for iplot, downD in enumerate(downstreamD):
            x = downD * self.D
            df = self.df.xs(x,level='x')[wakedir]
            if applyfilter is not None:
                df = applyfilter(df)
            if ref is not None:
                rmse = self.rms_error(df, ref)
            if norm:
                df /= self.D
            ax[iplot].plot(df.index, df.values, label=label, **kwargs)
            if annotate:
                ax[iplot].text(0.01, 0.95, 'x = {:g}D'.format(downD),
                               horizontalalignment='left',
                               verticalalignment='center',
                               transform=ax[iplot].transAxes)
        if len(downstreamD) == 1:
            ax = ax[0]
        if ref is not None:
            return fig,ax,rmse
        else:
            return fig,ax

    def init_plot(self,heightcmap='coolwarm'):
        fig,ax = plt.subplots(nrows=2,sharex=True,figsize=(11,6))
        self.yctr, = ax[0].plot([],[],linestyle='',marker='o',color='k')
        self.zctr, = ax[1].plot([],[],linestyle='',marker='o',color='k')
        # plot rotor
        ax[0].plot([0,0], [-0.5,0.5], lw=5, color='k')
        ax[1].plot([0,0], [-0.5,0.5], lw=5, color='k')
        # plot ground
        ax[1].axhspan(-1.25,-self.case.turbine.zhub/self.D,
                      hatch='//', facecolor='0.5')
        # time label
        self.title = ax[0].set_title('')
        # formatting
        #ax[0].axis('scaled')
        ax[0].set_xlim((0, (self.case.downstreamD[-1]+0.5)))
        ax[0].set_ylim((-1.25, 1.25))
        ax[1].set_ylim((-1.25, 1.25))
        ax[0].set_ylabel(r'$y/D$',fontsize='x-large')
        ax[1].set_ylabel(r'$z/D$',fontsize='x-large')
        ax[-1].set_xlabel(r'$x/D$',fontsize='x-large')
        return fig,ax

    def plot_xy(self,itime=0,**kwargs):
        """Plot lateral meandering"""
        print('NOTE: THIS DOES _NOT_ ACCOUNT FOR INFLOW ADVECTION TIME OFFSETS')
        fig,ax = self.init_plot(**kwargs)
        self.update_plot(itime)
        return fig,ax
    
    def update_plot(self,itime):
        df = self.df.xs(itime,level=0)
        x = df.index / self.D
        y = df['y'] / self.D
        z = df['z'] / self.D
        self.yctr.set_data(x,y)
        self.zctr.set_data(x,z)
        self.title.set_text('itime = {:d}'.format(itime))
        sys.stderr.write('\rUpdated plot for itime={:d}'.format(itime))
        return self.scat

    def animate(self,fname=None,**kwargs):
        fig,ax = self.init_plot(**kwargs)
        Nframes = len(self.df.xs(self.case.downstreamD[-1]*self.D, level='x'))
        print('Animating',Nframes,'frames')
        anim = FuncAnimation(fig, self.update_plot, frames=Nframes)
        if fname is None:
            fname = os.path.join(self.case.casedir,
                                 '{:s}_anim.mp4'.format(self.method))
        anim.save(fname, fps=24)
        print('Wrote',fname)


