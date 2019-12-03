import os, os.path
import glob
import pickle
import numpy
from numpy.polynomial import Polynomial
from scipy import ndimage, signal, interpolate
import matplotlib
#matplotlib.use('PDF')
#from galpy.orbit import Orbit
#from galpy.potential import MWPotential2014
#from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.util import bovy_conversion, bovy_plot, save_pickles, bovy_coords
import pal5_util
from pal5_util import R0, V0
#import run_pal5_abc
#import custom_stripping_df
#import bispectrum
#import seaborn as sns
#from matplotlib import cm, pyplot
#from matplotlib.ticker import FuncFormatter, NullFormatter
scaling= 'spectrum'
#save_figures= False
#minxi=0


# Load the smooth and peppered stream, only need 1 time-sampling, 
# bc only use it to convert to obs. space at present
sdf_smooth= pal5_util.setup_pal5model()
pepperfilename= 'pal5pepper1sampling.pkl'
if os.path.exists(pepperfilename):
    with open(pepperfilename,'rb') as savefile:
        sdf_pepper= pickle.load(savefile)
else:
    import simulate_streampepper
    timpacts= simulate_streampepper.parse_times('1sampling',9.)
    sdf_pepper= pal5_util.setup_pal5model(timpact=timpacts,
                                        hernquist=True)
    save_pickles(pepperfilename,sdf_pepper)

# Convert track to xi, eta
trackRADec=\
  bovy_coords.lb_to_radec(sdf_smooth._interpolatedObsTrackLB[:,0],
                          sdf_smooth._interpolatedObsTrackLB[:,1],
                          degree=True)
trackXiEta=\
  pal5_util.radec_to_pal5xieta(trackRADec[:,0],
                    trackRADec[:,1],degree=True)
smooth_track= []
for coord in range(6):
    if coord < 2:
        smooth_track.append(\
            interpolate.InterpolatedUnivariateSpline(sdf_smooth._interpolatedThetasTrack,
                                                     trackXiEta[:,coord]))
    else:
        smooth_track.append(\
            interpolate.InterpolatedUnivariateSpline(sdf_smooth._interpolatedThetasTrack,
                                                     sdf_smooth._interpolatedObsTrackLB[:,coord]))
smooth_ll= interpolate.InterpolatedUnivariateSpline(trackXiEta[:,0],
                                                    sdf_smooth._interpolatedThetasTrack)



def convert_dens_to_obs(apars,dens,dens_smooth,mO,poly_deg=3):
    """
    NAME:
        convert_dens_to_obs
    PURPOSE:
        Convert track to observed coordinates
    INPUT:
        apars - parallel angles
        dens - density(apars)
        dens_smooth - smooth-stream density(apars)
        mO= (None) mean parallel frequency (1D) 
            [needs to be set to get density on same grid as track]
        poly_deg= (3) degree of the polynomial to fit for the 'smooth' stream
    OUTPUT:
        (xi,dens/smooth)
    """
    mT= sdf_pepper.meanTrack(apars,_mO=mO,coord='lb')
    mradec= bovy_coords.lb_to_radec(mT[0],mT[1],degree=True)
    mxieta= pal5_util.radec_to_pal5xieta(mradec[:,0],mradec[:,1],degree=True)
    outll= numpy.arange(minxi,14.35,0.1)
    # Interpolate density
    ipll= interpolate.InterpolatedUnivariateSpline(mxieta[:,0],apars)
    ipdens= interpolate.InterpolatedUnivariateSpline(apars,dens/dens_smooth)
    return (outll,ipdens(ipll(outll)))
def convert_track_to_obs(apars,mO,coord):
    """
    NAME:
        convert_track_to_obs
    PURPOSE:
        Convert track to observed coordinates
    INPUT:
        apars - parallel angles
        mO - mean parallel frequency (1D)
        coord - coordinate to convert to (1: eta, 2: distance, 3: vlos, 4: pmll, 5: pmbb)
    OUTPUT:
        (longitude,(track-smooth)[coord])
    """
    mT= sdf_pepper.meanTrack(apars,_mO=mO,coord='lb')
    mradec= bovy_coords.lb_to_radec(mT[0],mT[1],degree=True)
    mxieta= pal5_util.radec_to_pal5xieta(mradec[:,0],mradec[:,1],degree=True)
    mT[0]= mxieta[:,0]
    mT[1]= mxieta[:,1]
    # Interpolate
    ipll= interpolate.InterpolatedUnivariateSpline(mT[0],apars)
    ipcoord= interpolate.InterpolatedUnivariateSpline(apars,mT[coord])
    outll= numpy.arange(minxi,14.35,0.1)
    return (outll,ipcoord(ipll(outll))-smooth_track[coord](smooth_ll(outll)))    
def read_data(filename):
    data1= numpy.genfromtxt(filename,delimiter=',',max_rows=5002)
    # Search for batches
    batchfilenames= glob.glob(filename.replace('.dat','.*.dat'))
    for bfilename in batchfilenames:
        try:
            datab= numpy.genfromtxt(bfilename,delimiter=',')
        except: continue
        data1= numpy.vstack((data1,datab[2:]))
    return data1
def read_dens(filename,indx=None,rand=False,poly_deg=3):
    densdata= read_data(filename)
    mOfilename= filename.replace('dens','omega')
    mOdata= read_data(mOfilename)
    apars= densdata[0]
    nsim= len(densdata)-2 # first two are apar and smooth
    ll,tdata1= convert_dens_to_obs(apars,densdata[2],densdata[1],
                                   mOdata[2],poly_deg=poly_deg)
    if not indx is None:
        return convert_dens_to_obs(apars,densdata[2+indx],densdata[1],
                                   mOdata[2+indx])
    if rand:
        rindx= int(numpy.floor(numpy.random.uniform()*nsim))
        print rindx
        return convert_dens_to_obs(apars,densdata[2+rindx],densdata[1],
                                   mOdata[2+rindx])
    out= numpy.zeros((nsim,len(tdata1)))
    for ii in range(nsim):
        ll, tdata1= convert_dens_to_obs(apars,densdata[2+ii],densdata[1],
                                        mOdata[2+ii])
        out[ii]= tdata1
    return (ll,out)
def median_csd(filename,filename2=None,scatter=False,
              coord1=1,coord2=1,
              err1=None,err2=None,
              poly_deg=3):
    # Compute the median CSD, if filename2 is not None, compute cross
    data1= read_data(filename)
    if filename2 is None:
        data2= data1
    else:
        data2= read_data(filename2)
    if coord1 == 'dens':
        mOfilename= filename.replace('dens','omega')
        mOdata1= read_data(mOfilename)
    if coord2 == 'dens':
        if filename2 is None:
            mOdata2= mOdata1
        else:
            mOfilename2= filename2.replace('dens','omega')
            mOdata2= read_data(mOfilename2)
    apars= data1[0]
    nsim= len(data1)-2 # first two are apar and smooth
    if nsim < 1000:
        print "WARNING: Using fewer than 1,000 (%i) simulations to compute medians from %s" % (nsim,filename)
    if coord1 == 'dens':
        ll,tdata1= convert_dens_to_obs(apars,data1[2],data1[1],mOdata1[2],poly_deg=poly_deg)
    else:
        ll,tdata1= convert_track_to_obs(apars,data1[2],coord1)
    if coord2 == 'dens':
        ll,tdata2= convert_dens_to_obs(apars,data2[2],data2[1],mOdata2[2],poly_deg=poly_deg)
    else:
        ll,tdata2= convert_track_to_obs(apars,data2[2],coord2)
    px= signal.csd(tdata1,tdata2,fs=1./(ll[1]-ll[0]),scaling=scaling,nperseg=len(ll))[0]
    ppy= numpy.empty((nsim,len(px)))
    ppy_err= numpy.empty((nsim,len(px)))
    for ii in range(nsim):
        # convert
        if coord1 == 'dens':
            ll,tdata1= convert_dens_to_obs(apars,data1[2+ii],data1[1],mOdata1[2+ii],poly_deg=poly_deg)
        else:
            ll,tdata1= convert_track_to_obs(apars,data1[2+ii],coord1)
        if numpy.any(data1[2+ii] != data2[2+ii]):
            if coord2 == 'dens':
                ll,tdata2= convert_dens_to_obs(apars,data2[2+ii],data2[1],mOdata2[2+ii],poly_deg=poly_deg)
            else:
                ll,tdata2= convert_track_to_obs(apars,data2[2+ii],coord2)
            ppy[ii]= numpy.absolute(signal.csd(tdata1,tdata2,
                                               fs=1./(ll[1]-ll[0]),scaling=scaling,
                                               nperseg=len(ll))[1])
        else:
            ppy[ii]= signal.csd(tdata1,tdata1,
                                fs=1./(ll[1]-ll[0]),scaling=scaling,
                                nperseg=len(ll))[1].real
        # Same for errors
        if not err1 is None:
            tmock1= err1*numpy.random.normal(size=len(ll))
            if numpy.any(data1[2+ii] != data2[2+ii]):
                tmock2= err2*numpy.random.normal(size=len(ll))
                ppy_err[ii]= numpy.absolute(signal.csd(tmock1,tmock2,
                                                       fs=1./(ll[1]-ll[0]),scaling=scaling,
                                                       nperseg=len(ll))[1])
            else:
                ppy_err[ii]= signal.csd(tmock1,tmock1,
                                        fs=1./(ll[1]-ll[0]),scaling=scaling,
                                        nperseg=len(ll))[1].real
    # Following is the correct scaling in the sense that random noise gives same CSD no matter how long the stream
    if not scatter:
        return (1./px,numpy.sqrt(numpy.nanmedian(ppy,axis=0)*(ll[-1]-ll[0])),
                numpy.sqrt(numpy.nanmedian(ppy_err,axis=0)*(ll[-1]-ll[0])))
    else:
        out= numpy.sqrt(numpy.nanmedian(ppy,axis=0)*(ll[-1]-ll[0]))
        ppy.sort(axis=0)
        return (1./px,out,
                numpy.sqrt(ppy[int(numpy.round(0.25*nsim))]*(ll[-1]-ll[0])),
                numpy.sqrt(ppy[int(numpy.round(0.75*nsim))]*(ll[-1]-ll[0])),
                numpy.sqrt(numpy.nanmedian(ppy_err,axis=0)*(ll[-1]-ll[0])))
def plot_dens(filename,poly_deg=3,color='k',zorder=1,ls='-',
              fill=False,fill_color='0.65',fill_zorder=0,
              err_color='r',err_zorder=0,
              err=None,scale=1.,add_err=0., label=None,fill_alpha=0.5):
    px, py, py_err= median_csd(filename,coord1='dens',coord2='dens',
                               err1=err,poly_deg=poly_deg)
    py= numpy.sqrt(py**2.+add_err**2.)
    loglog(px,scale*py,lw=2.,color=color,zorder=zorder,ls=ls, label=label)
    if not err is None: loglog(px,scale*py_err,lw=2.,color=err_color,zorder=err_zorder,ls=ls)
    if fill:
        plotx, dum, low, high, dum= median_csd(filename,scatter=True,
                                               coord1='dens',coord2='dens',
                                               poly_deg=poly_deg)
        fill_between(plotx,scale*low,scale*high,color=fill_color,zorder=fill_zorder,alpha=fill_alpha)
    return None
def plot_all_track(filename,color='k',zorder=1,ls='-',
                   fill=False,fill_color='0.65',fill_zorder=0,
                   err_color='r',err_zorder=0,
                   errs=[None,None,None],
                   scale=1.):
    subplot(1,3,1)
    px, py, py_err= median_csd(filename,coord1=1,err1=errs[0])
    loglog(px,scale*py,lw=2.,color=color,zorder=zorder,ls=ls)
    if not errs[0] is None: loglog(px,scale*py_err,lw=2.,color=err_color,zorder=err_zorder,ls=ls)
    if fill:
        plotx, dum, low, high, dum= median_csd(filename,scatter=True,coord1=1)
        fill_between(plotx,scale*low,scale*high,color=fill_color,zorder=fill_zorder,alpha=0.5)
    subplot(1,3,2)
    px, py, py_err= median_csd(filename,coord1=2,err1=errs[1])
    loglog(px,scale*py,lw=2.,color=color,zorder=zorder,ls=ls)
    if not errs[1] is None: loglog(px,scale*py_err,lw=2.,color=err_color,zorder=err_zorder,ls=ls)
    if fill:
        plotx, dum, low, high, dum= median_csd(filename,scatter=True,coord1=2)
        fill_between(plotx,scale*low,scale*high,color=fill_color,zorder=fill_zorder,alpha=0.5)
    subplot(1,3,3)
    px, py, py_err= median_csd(filename,coord1=3,err1=errs[2])
    loglog(px,scale*py,lw=2.,color=color,zorder=zorder,ls=ls)
    if not errs[2] is None: loglog(px,scale*py_err,lw=2.,color=err_color,zorder=err_zorder,ls=ls)
    if fill:
        plotx, dum, low, high, dum= median_csd(filename,scatter=True,coord1=3)
        fill_between(plotx,scale*low,scale*high,color=fill_color,zorder=fill_zorder,alpha=0.5)
    return None
def plot_all_track_dens(filename,color='k',zorder=1,ls='-',
                        fill=False,fill_color='0.65',fill_zorder=0,
                        err_color='r',err_zorder=0,
                        err1=None,err2s=[None,None,None],
                        scale=1.):
    subplot(1,3,1)
    px, py, py_err= median_csd(filename,filename2=filename.replace('dens','omega'),
                               coord1='dens',coord2=1,err1=err1,err2=err2s[0])
    loglog(px,scale*py,lw=2.,color=color,zorder=zorder,ls=ls)
    if not err2s[0] is None: loglog(px,scale*py_err,lw=2.,color=err_color,zorder=err_zorder,ls=ls)
    if fill:
        plotx, dum, low, high, dum= median_csd(filename,filename2=filename.replace('dens','omega'),
                                               scatter=True,coord1='dens',coord2=1)
        fill_between(plotx,scale*low,scale*high,color=fill_color,zorder=fill_zorder,alpha=0.5)
    subplot(1,3,2)
    px, py, py_err= median_csd(filename,filename2=filename.replace('dens','omega'),
                               coord1='dens',coord2=2,err1=err1,err2=err2s[1])
    loglog(px,scale*py,lw=2.,color=color,zorder=zorder,ls=ls)
    if not err2s[1] is None: loglog(px,scale*py_err,lw=2.,color=err_color,zorder=err_zorder,ls=ls)
    if fill:
        plotx, dum, low, high, dum= median_csd(filename,filename2=filename.replace('dens','omega'),
                                               scatter=True,coord1='dens',coord2=2)
        fill_between(plotx,scale*low,scale*high,color=fill_color,zorder=fill_zorder,alpha=0.5)
    subplot(1,3,3)
    px, py, py_err= median_csd(filename,filename2=filename.replace('dens','omega'),
                               coord1='dens',coord2=3,err1=err1,err2=err2s[2])
    loglog(px,scale*py,lw=2.,color=color,zorder=zorder,ls=ls)
    if not err2s[2] is None: loglog(px,scale*py_err,lw=2.,color=err_color,zorder=err_zorder,ls=ls)
    if fill:
        plotx, dum, low, high, dum= median_csd(filename,filename2=filename.replace('dens','omega'),
                                               scatter=True,coord1='dens',coord2=3)
        fill_between(plotx,scale*low,scale*high,color=fill_color,zorder=fill_zorder,alpha=0.5)
    return None
def set_ranges_and_labels_dens():
    bovy_plot.bovy_text(r'$\sqrt{\delta\delta}$',top_left=True,size=18.)
    xlabel(r'$1/k_{\xi}\,(\mathrm{deg})$')
    ylim(0.01,10.)
    xlim(1.,100.)
    for axis in [gca().xaxis,gca().yaxis]:
        axis.set_major_formatter(FuncFormatter(
                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
    return None
def set_ranges_and_labels():
    subplot(1,3,1)
    bovy_plot.bovy_text(r'$\sqrt{\eta\eta}$',top_left=True,size=18.)
    xlabel(r'$1/k_{\xi}\,(\mathrm{deg})$')
    ylim(0.001,1.)
    xlim(1.,100.)
    for axis in [gca().xaxis,gca().yaxis]:
        axis.set_major_formatter(FuncFormatter(
                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
    subplot(1,3,2)
    bovy_plot.bovy_text(r'$\sqrt{DD}$',top_left=True,size=18.)
    xlabel(r'$1/k_{\xi}\,(\mathrm{deg})$')
    ylim(0.001,1.)
    xlim(1.,100.)
    gca().xaxis.set_major_formatter(FuncFormatter(
            lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
    gca().yaxis.set_major_formatter(NullFormatter())
    subplot(1,3,3)
    bovy_plot.bovy_text(r'$\sqrt{V_{\mathrm{los}} V_{\mathrm{los}}}$',top_left=True,size=18.)
    xlabel(r'$1/k_{\xi}\,(\mathrm{deg})$')
    ylim(0.001,1.)
    xlim(1.,100.)
    gca().xaxis.set_major_formatter(FuncFormatter(
            lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
    gca().yaxis.set_major_formatter(NullFormatter())
    return None
def set_ranges_and_labels_cross():
    subplot(1,3,1)
    bovy_plot.bovy_text(r'$\sqrt{|\delta b|}$',top_left=True,size=18.)
    xlabel(r'$1/k_{\xi}\,(\mathrm{deg})$')
    ylim(0.001,10.)
    xlim(1.,100.)
    for axis in [gca().xaxis,gca().yaxis]:
        axis.set_major_formatter(FuncFormatter(
                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
    subplot(1,3,2)
    bovy_plot.bovy_text(r'$\sqrt{|\delta D|}$',top_left=True,size=18.)
    xlabel(r'$1/k_{\xi}\,(\mathrm{deg})$')
    ylim(0.001,10.)
    xlim(1.,100.)
    gca().xaxis.set_major_formatter(FuncFormatter(
            lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
    gca().yaxis.set_major_formatter(NullFormatter())
    subplot(1,3,3)
    bovy_plot.bovy_text(r'$\sqrt{|\delta V_{\mathrm{los}}|}$',top_left=True,size=18.)
    xlabel(r'$1/k_{\xi}\,(\mathrm{deg})$')
    ylim(0.001,10.)
    xlim(1.,100.)
    gca().xaxis.set_major_formatter(FuncFormatter(
            lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
    gca().yaxis.set_major_formatter(NullFormatter())
    return None
def color_from_colormap(val,cmap,cmin,cmax):
    return cmap((val-cmin)/(cmax-cmin))
def add_colorbar_dens(vmin,vmax,clabel,save_figures=False):
    fig= pyplot.gcf()
    if save_figures:
        cbar_ax = fig.add_axes([0.775,0.135,0.05,0.815])
    else:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.925, 0.13, 0.075, 0.83])
    sm = pyplot.cm.ScalarMappable(cmap=cmap,norm=pyplot.Normalize(vmin=vmin,vmax=vmax))
    sm._A = []
    cbar= fig.colorbar(sm,cax=cbar_ax,use_gridspec=True,format=r'$%.1f$')
    cbar.set_label(clabel)
    return None
def add_discrete_colorbar_dens(vmin,vmax,clabel,ticks,save_figures=False):
    fig= pyplot.gcf()
    if save_figures:
        cbar_ax = fig.add_axes([0.775,0.135,0.05,0.815])
    else:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.925, 0.13, 0.075, 0.83])
    tcmap = matplotlib.colors.ListedColormap([cmap(f) for f in numpy.linspace(0.,1.,len(ticks))])
    dtick= (ticks[1]-ticks[0])
    sm = pyplot.cm.ScalarMappable(cmap=tcmap,norm=pyplot.Normalize(vmin=vmin-0.5*dtick,vmax=vmax+0.5*dtick))
    sm._A = []
    cbar= fig.colorbar(sm,cax=cbar_ax,use_gridspec=True,format=r'$%.1f$',ticks=ticks)
    cbar.set_label(clabel)
    return None
def add_colorbar(vmin,vmax,clabel,save_figures=False):
    fig= pyplot.gcf()
    if save_figures:
        cbar_ax = fig.add_axes([0.9,0.135,0.025,0.815])
    else:
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.975, 0.13, 0.025, 0.83])
    sm = pyplot.cm.ScalarMappable(cmap=cmap,norm=pyplot.Normalize(vmin=vmin,vmax=vmax))
    sm._A = []
    cbar= fig.colorbar(sm,cax=cbar_ax,use_gridspec=True,format=r'$%.1f$')
    cbar.set_label(clabel)
    return None


def plot_dens_data(filename=None, backg=400., color='k',zorder=10,marker='o',
                   poly_deg=1,minxi=0.25,
                   errsim_color='k',errsim_zorder=0,
                   err_color=0.7,err_zorder=0,err=False):
    """Plots the power spectrum of the data"""
    # Read the data
    if filename == None:
        data= numpy.loadtxt('data/ibata_fig7b_raw.dat',delimiter=',')
        data_lowerr= numpy.loadtxt('data/ibata_fig7b_rawlowerr.dat',delimiter=',')
        data_uperr= numpy.loadtxt('data/ibata_fig7b_rawuperr.dat',delimiter=',')
    else:
        data= numpy.loadtxt('data/fakeobs/'+filename,delimiter=',')
        data_lowerr= numpy.loadtxt('data/fakeobs/'+filename+'_lower',delimiter=',')
        data_uperr= numpy.loadtxt('data/fakeobs/'+filename+'_upper',delimiter=',')
    sindx= numpy.argsort(data[:,0])
    data= data[sindx]
    sindx= numpy.argsort(data_lowerr[:,0])
    data_lowerr= data_lowerr[sindx]
    sindx= numpy.argsort(data_uperr[:,0])
    data_uperr= data_uperr[sindx]
    data_err= 0.5*(data_uperr-data_lowerr)
    # CUTS
    if filename == None: indx= (data[:,0] > minxi-0.05)*(data[:,0] < 14.35)
    else: indx= (data[:,0] >0.)*(data[:,0] <14.4)
    data= data[indx]
    data_lowerr= data_lowerr[indx]
    data_uperr= data_uperr[indx]
    data_err= data_err[indx]
    # Compute power spectrum
    tdata= data[:,1]-backg
    pp= Polynomial.fit(data[:,0],tdata,deg=poly_deg,w=1./data_err[:,1])
    tdata/= pp(data[:,0])
    data_err= data_err[:,1]/pp(data[:,0])
    ll= data[:,0]
    px, py= signal.csd(tdata,tdata,
                        fs=1./(ll[1]-ll[0]),scaling=scaling,
                        nperseg=len(ll))
    py= py.real
    px= 1./px
    py= numpy.sqrt(py*(ll[-1]-ll[0]))
    # Perform simulations of the noise to determine the power in the noise
    nerrsim= 1000
    ppy_err= numpy.empty((nerrsim,len(px)))
    for ii in range(nerrsim):
        tmock= data_err*numpy.random.normal(size=len(ll))
        ppy_err[ii]= signal.csd(tmock,tmock,
                                fs=1./(ll[1]-ll[0]),scaling=scaling,
                                nperseg=len(ll))[1].real
    py_err= numpy.sqrt(numpy.median(ppy_err,axis=0)*(ll[-1]-ll[0]))
    pcut= np.nanmedian(ppy_err) # Only trust points above this, then remove noise
    #loglog(px[py>pcut],numpy.sqrt(py[py>pcut]**2.-py_err[py>pcut]**2.),
    #       marker=marker,color=color,zorder=zorder,ls='none')
    yvals = numpy.sqrt(py**2.-py_err**2.)
    loglog(px,yvals,marker=marker,color=color,zorder=zorder,ls='none',label=filename)
    errorbar(px[np.isnan(yvals)],py_err[np.isnan(yvals)],
             yerr=numpy.array([.03+0.*px[np.isnan(yvals)],.03+0.*px[np.isnan(yvals)]]),
             uplims=True,capthick=2.,ls='none',color=color,zorder=zorder)
    loglog(px,py_err,lw=2.,color=errsim_color,zorder=errsim_zorder)
    return None

def plot_dens_multi(filename,poly_deg=3,color='k',zorder=1,ls='-',
              fill=False,fill_color='0.65',fill_zorder=0,
              err_color='r',err_zorder=0,
              err=None,scale=1.,add_err=0.):
    px, py, py_err= median_csd(filename,coord1='dens',coord2='dens',
                               err1=err,poly_deg=poly_deg)
    py= numpy.sqrt(py**2.+add_err**2.)
    loglog(px,scale*py,lw=2.,color=color,zorder=zorder,ls=ls)
    if not err is None: loglog(px,scale*py_err,lw=2.,color=err_color,zorder=err_zorder,ls=ls)
    if fill:
        plotx, dum, low, high, dum= median_csd(filename,scatter=True,
                                               coord1='dens',coord2='dens',
                                               poly_deg=poly_deg)
        fill_between(plotx,scale*low,scale*high,color=fill_color,zorder=fill_zorder,alpha=0.5)
    return None

#############################################
#############################################
#for real pal5
minxi= 0.25
poly_deg= 3
save_figures=False
bovy_plot.bovy_print(axes_labelsize=18.,xtick_labelsize=14.,ytick_labelsize=14.)
if save_figures:
    figsize(11/3.,4.25)
    fig= figure()
    fig.subplots_adjust(left=0.15,right=0.8875,bottom=0.135,top=0.95,wspace=0.075)
else:
    figsize(16/3.,6)
plot_dens('./data/pal5_t64sampling_X3_6-9_dens.dat',
          poly_deg=poly_deg,
          color='k',fill_color='k',zorder=2,ls='-',fill=True, label='3xCDM', fill_alpha=0.25)
plot_dens('./data/pal5_t64sampling_X3_7-9_dens.dat',
          poly_deg=poly_deg,
          color='r',fill_color='r',zorder=2,ls='-',fill=True, label='CDM', fill_alpha=0.25)
#plot_dens_data(poly_deg=poly_deg,minxi=minxi)
set_ranges_and_labels_dens()


#add some fake pal5 data
#############################################
#############################################
#for real pal5
minxi= 0.25
poly_deg= 3
save_figures=False
bovy_plot.bovy_print(axes_labelsize=18.,xtick_labelsize=14.,ytick_labelsize=14.)
if save_figures:
    figsize(11/3.,4.25)
    fig= figure()
    fig.subplots_adjust(left=0.15,right=0.8875,bottom=0.135,top=0.95,wspace=0.075)
else:
    figsize(16/3.,6)
plot_dens('./data/pal5_t64sampling_X3_7-9_dens.dat',
          poly_deg=poly_deg,
          color='k',zorder=2,ls='-',fill=True)
plot_dens_data(poly_deg=poly_deg,minxi=minxi,color='k',errsim_color='k')
plot_dens_data(filename='CFHT',poly_deg=1,color='r',errsim_color='r',backg=0.)
plot_dens_data(filename='LSST',poly_deg=1,color='g',errsim_color='g',backg=0.)
plot_dens_data(filename='WFIRST',poly_deg=1,color='b',errsim_color='b',backg=0.)
#plot_dens_data(filename='ibata',poly_deg=3,color='orange',errsim_color='orange',backg=400)
set_ranges_and_labels_dens()

plt.legend(loc='lower right')

