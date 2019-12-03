import streampepperdf
import os, os.path
import numpy
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.orbit import Orbit
from galpy.df import streamdf, streamgapdf
from galpy.util import bovy_plot, bovy_conversion, bovy_coords
import seaborn as sns
import pal5_util
import astropy.units as u
import scipy
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
import photErrorModel as pem
from scipy.interpolate import interp1d
import pickle
from numpy.polynomial import Polynomial
import read_girardi
import astropy.io.fits as fits
import time

#lbr = pickle.load('./lbr.pkl')
bg_interps = pickle.load(open('./bg_interps.pkl', "rb" ))


maglim =28.
nbins = 200
magerror_mod = 1.
pstype='dens'
plottype='snr'


iso = isoinstance()
bg = bginstance()
errormodels = dict()
filt='g'
errormodels['LSST'] = interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), filt, survey='LSST'), bounds_error=False, fill_value=Inf)
errormodels['LSST10'] = interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), filt, survey='LSST10'), bounds_error=False, fill_value=Inf)
errormodels['CFHT'] = interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), filt, survey='CFHT'), bounds_error=False, fill_value=Inf)
errormodels['SDSS'] = interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), filt, survey='SDSS'), bounds_error=False, fill_value=Inf)
errormodels['CASTOR'] = interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), filt, survey='SDSS'), bounds_error=False, fill_value=Inf)



surveys = ['LSST10', 'CFHT', 'SDSS']
colors = ['k', 'b', 'r']
runs = 100
nbins = 200
tdata_out =   np.zeros((len(surveys),runs,nbins-1))
terr_out =    np.zeros((len(surveys),runs,nbins-1))
px_out =      np.zeros((len(surveys),runs,nbins/2))
py_out =      np.zeros((len(surveys),runs,nbins/2))
py_err_out =  np.zeros((len(surveys),runs,nbins/2))

for j in np.arange(len(surveys)):
	survey =surveys[j]
	c= colors[j]

	for i in np.arange(runs):
		print 100*float(i)/runs
		try: tdata_out[j][i], terr_out[j][i], px_out[j][i], py_out[j][i], py_err_out[j][i] = analysis('lbr'+str(i+1), survey=survey,nbins=nbins, plot=False, which = pstype)
		except: continue


if plottype=='ps':
	for j in np.arange(len(surveys)):
		survey =surveys[j]
		c= colors[j]

		plt.fill_between((px_out[j][0][1:]), np.nanpercentile((py_out[j]), (25), axis=0)[1:],  y2= np.nanpercentile((py_out[j]), (75), axis=0)[1:], color=c, alpha=0.2)
		#plt.loglog(px_out[0], np.median(py_out, axis=0),     c='k', lw=2, label='LSST',nonposx='clip', nonposy='clip')
		#plt.loglog(px_out[0], np.median(py_err_out, axis=0),     c='k', lw=2,nonposx='clip', nonposy='clip')#, label='SDSS')

		#plt.plot(np.log10(px_out[j][0][1:]), np.log10(np.nanmedian(py_out[j], axis=0)[1:]    ), c=c, lw=2, label=survey)
		#plt.plot(np.log10(px_out[j][0][1:]), np.log10(np.nanmedian(py_err_out[j], axis=0)[1:]),     c=c, lw=2, ls='--')#, label='SDSS')

		plt.loglog((px_out[j][0][1:]), (np.nanmedian(py_out[j], axis=0)[1:]    ), c=c, lw=2, label=survey)
		plt.loglog((px_out[j][0][1:]), (np.nanmedian(py_err_out[j], axis=0)[1:]),     c=c, lw=2, ls='--')#, label='SDSS')

if plottype=='snr':
	for j in np.arange(len(surveys)):
		survey =surveys[j]
		c= colors[j]

		plt.fill_between((px_out[j][0][1:]), np.nanpercentile((py_out[j]), (25), axis=0)[1:]/(np.nanmedian(py_err_out[j], axis=0)[1:]),  y2= np.nanpercentile((py_out[j]), (75), axis=0)[1:]/(np.nanmedian(py_err_out[j], axis=0)[1:]), color=c, alpha=0.2)
		
		#plt.loglog(px_out[0], np.median(py_out, axis=0),     c='k', lw=2, label='LSST',nonposx='clip', nonposy='clip')
		#plt.loglog(px_out[0], np.median(py_err_out, axis=0),     c='k', lw=2,nonposx='clip', nonposy='clip')#, label='SDSS')

		#plt.plot(np.log10(px_out[j][0][1:]), np.log10(np.nanmedian(py_out[j], axis=0)[1:]    ), c=c, lw=2, label=survey)
		#plt.plot(np.log10(px_out[j][0][1:]), np.log10(np.nanmedian(py_err_out[j], axis=0)[1:]),     c=c, lw=2, ls='--')#, label='SDSS')

		plt.plot((px_out[j][0][1:]), (np.nanmedian(py_out[j], axis=0)[1:]    )/(np.nanmedian(py_err_out[j], axis=0)[1:]), c=c, lw=2, label=survey)
		#plt.loglog((px_out[j][0][1:]), (np.nanmedian(py_err_out[j], axis=0)[1:]),     c=c, lw=2, ls='--')#, label='SDSS')
		ax = plt.gca()
		ax.set_xscale('log')

if pstype == 'dens': 
	plt.scatter(np.log10(px_out[0,0,1:]), np.log10(px_out[0,0,1:])*0+-1.5, c='k', alpha=1, s=4)
	plt.ylabel(r'$\sqrt{\delta \delta}$')
if pstype == 'dist': 
	plt.scatter(np.log10(px_out[0,0,1:]), np.log10(px_out[0,0,1:])*0+-4.25, c='k', alpha=1, s=4)
	plt.ylabel(r'$\sqrt{dd}$')
if pstype == 'pos': 
	plt.scatter(np.log10(px_out[0,0,1:]), np.log10(px_out[0,0,1:])*0+-4.25, c='k', alpha=1, s=4)
	plt.ylabel(r'$\sqrt{bb}$')

plt.xlim([-.1,20])
plt.xlabel(r'$\mathrm{log10\ 1/k_\xi (deg)}$')
plt.legend(loc='upper left')



def analysis(key, survey='LSST', maglim=28., nbins=140, magerror_mod=1., plot=False, which = 'dens', filt='g', verbose=False):
	spdata = lbr[key][0:2].T
	spdata_rs = lbr[key][2].T

	ms, gstars, rstars, istars = assign_mass_and_color(spdata, spdata_rs, 
	min_g_mag = maglim, max_g_mag = 14.6, slope = -0.5)


	realmaglim = getMagLimit(filt, survey=survey)

	if maglim > realmaglim:
		if verbose==True: print 'warning: maglim fainter than survey depth, overriding'
	maglim = realmaglim

	spdata = spdata[gstars<maglim]
	spdata_rs = spdata_rs[gstars<maglim]

	distfit = Polynomial.fit(spdata[:,0], spdata_rs, deg=1)
	def dm_from_xi(xi):
		return 5*np.log10(distfit(xi)*1e3)-5 

	errormodel = errormodels[survey]
	bg_interp = bg_interps[survey]

	#calculate background for each bin
	xis = np.linspace(-20,0, nbins)
	dxi = xis[1]-xis[0]
	midpoints = (xis[1:]+xis[:-1])/2.
	bgs = midpoints*0.
	for i in np.arange(len(midpoints)):
		bgs[i] = bg_interp((dm_from_xi(midpoints[i]), magerror_mod, maglim))
		bgs[i] = np.round(bgs[i]*dxi) # per sq deg


	if which == 'dens':
		tdata, terr, px, py, py_err=dens_and_power(spdata, bkg = bgs, nbins=nbins, col='k', plot=False)
	if which == 'dist':
		tdata, terr, px, py, py_err=dist_and_power(spdata, spdata_rs, bkg = bgs, nbins=nbins, col='k', plot=False)
	if which == 'pos':
		tdata, terr, px, py, py_err=pos_and_power(spdata, spdata_rs, bkg = bgs, nbins=nbins, col='k', plot=False)

	return tdata, terr, px, py, py_err







plt.figure()
for i in np.arange(20):
	ls = lbr['lbr'+str(i)][0]
	plt.hist(ls, bins=np.linspace(-20,0,100),histtype='step')


#####################################################
#####################################################
#####################################################
#create density histograms for further use

import pickle
from numpy.polynomial import Polynomial
from scipy import signal
import iso_handling
import pal5_mock_ps
model = pal5_mock_ps.pal5_stream_instance()
model.add_smooth()
#lbrs = pickle.load(open('./data/lbr500.pkl', "rb" ))
def assign_masses(nstars, slope = -0.5):
    maxmass = np.max(.944)
    minmass = np.min(.15)
    masses = np.zeros(nstars)

    for i in np.arange(nstars):
        x1=maxmass #np.minimum(maxmass, maxmass_from_g[i])
        x0=minmass #np.maximum(minmass, minmass_from_g[i])
        y=np.random.uniform(size=1)
        masses[i] = ((x1**(slope+1) - x0**(slope+1))*y + x0**(slope+1))**(1/(slope+1))
    return masses

#surveys=['SDSS', 'CFHT', 'LSST', 'LSST10','CASTOR', 'WFIRST']
surveys=['CFHT', 'LSST10', 'WFIRST']
#surveys=['CFHT']
#

nsample = 10000
nsets = 180
restart=False
if restart ==True: pepperdict = {}
if restart ==True: smoothdict = {}
if restart ==True: errordict = {}

for i, key in enumerate(surveys):
    if restart ==True: pepperdict[key]=[]
    if restart ==True: smoothdict[key]=[]
    if restart ==True: errordict[key]=[]
    for j in np.arange(nsets):
    	print '%2.2f pct'%(float(j)/nsets*100)
    	model.sample(nsample=nsample)
        lbr = np.vstack((model.spdata.T,np.array(model.spdata_rs).T))
        
        masses = assign_masses(lbr.shape[1])
        if 'LSST' in key:
            mag1 = isodict['lsst_g-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror1 = errormodels[key]['g'](mag1)
            mag2 = isodict['lsst_r-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror2 = errormodels[key]['r'](mag2)
        elif key == 'CASTOR':
            mag1 = isodict['castor_u-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror1 = errormodels[key]['u'](mag1)
            mag2 = isodict['castor_g-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror2 = errormodels[key]['g'](mag2)
        elif key == 'WFIRST':
            mag1 = isodict['Z087-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror1 = errormodels[key]['z'](mag1)
            mag2 = isodict['H158-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror2 = errormodels[key]['h'](mag2)
        else:
            mag1 = isodict['sdss_g-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror1 = errormodels[key]['g'](mag1)
            mag2 = isodict['sdss_r-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror2 = errormodels[key]['r'](mag2)

        omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1))]
        omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1))]

        #print '%3i stream stars'%(len(omag1))
        bgdens = bgdict[key]((16.8,np.max(omag2)))
        #print '%3i background stars'%(bgdens)

        pcounts, pbins = np.histogram(lbr[0][((magerror1<.1)&(magerror2<.1))], bins=np.linspace(-14.5,0,145))
        pepperdict[key]=pepperdict[key]+[pcounts]

        lbr = model.smooth.sample(n=nsample, lb=True)[0:3]
        masses = assign_masses(lbr.shape[1])
        if 'LSST' in key:
            mag1 = isodict['lsst_g-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror1 = errormodels[key]['g'](mag1)
            mag2 = isodict['lsst_r-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror2 = errormodels[key]['r'](mag2)
        elif key == 'CASTOR':
            mag1 = isodict['castor_u-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror1 = errormodels[key]['u'](mag1)
            mag2 = isodict['castor_g-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror2 = errormodels[key]['g'](mag2)
        elif key == 'WFIRST':
            mag1 = isodict['Z087-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror1 = errormodels[key]['z'](mag1)
            mag2 = isodict['H158-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror2 = errormodels[key]['h'](mag2)
        else:
            mag1 = isodict['sdss_g-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror1 = errormodels[key]['g'](mag1)
            mag2 = isodict['sdss_r-10.00-0.0001'](masses) + (5*np.log10(lbr[2]*1e3)-5)
            magerror2 = errormodels[key]['r'](mag2)

        omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1))]
        omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1))]

        lbr[0]=lbr[0]-360.
        lbr[1]=lbr[1]-np.median(lbr[1][(lbr[0]<0.)&(lbr[0]>-20.)])
        scounts, sbins = np.histogram(lbr[0][((magerror1<.1)&(magerror2<.1))], bins=np.linspace(-14.5,0,145))
        smoothdict[key]=smoothdict[key]+[scounts]

        errs = np.random.poisson(bgdens*.1*.2, size=len(scounts))
        errordict[key]=errordict[key]+[errs]

#calculate PS for one of the dicts above
if restart ==True: ps = {}
if restart ==True: ps_err = {}
for survey in pepperdict.keys():
	smoothdens = np.median(smoothdict[survey],axis=0)
	if restart ==True: ps[survey]=[]
	if restart ==True: ps_err[survey]=[]
	for i in np.arange(len(pepperdict[survey])):
		print '%2.2f pct'%(float(i)/len(pepperdict[survey])*100)
		bins=np.linspace(-14.5,0,145)
		centroids = (bins[1:] + bins[:-1]) / 2.
		tdata = (pepperdict[survey][i]+errordict[survey][i])/smoothdens
		px, py= signal.csd(tdata,tdata, fs=1./(centroids[1]-centroids[0]), scaling='spectrum', nperseg=len(centroids))
		py= py.real
		px= 1./px
		py= numpy.sqrt(py*(centroids[-1]-centroids[0]))
		ps[survey] = ps[survey] + [py]


		nerrsim= 1000
		ppy_err= numpy.empty((nerrsim,len(px)))
		terr = np.sqrt(errordict[survey][i]+ np.sqrt(pepperdict[survey][i])+10)/((smoothdens))

		for ii in range(nerrsim):
			tmock= terr*numpy.random.normal(size=len(centroids))
			ppy_err[ii]= signal.csd(tmock,tmock,
									fs=1./(centroids[1]-centroids[0]),scaling='spectrum',
									nperseg=len(centroids))[1].real
		ps_err[survey] = ps_err[survey] + [numpy.sqrt(numpy.median(ppy_err,axis=0)*(centroids[-1]-centroids[0]))]








# Convert track to xi, eta
def convert_dens_to_obs(sdf_pepper,apars,
                        dens,mO,dens_smooth,minxi=0.25,maxxi=14.35):
    """
    NAME:
        convert_dens_to_obs
    PURPOSE:
        Convert track to observed coordinates
    INPUT:
        sdf_pepper - streampepperdf object
        apars - parallel angles
        dens - density(apars)
        dens_smooth - smooth density(apars)
        mO= (None) mean parallel frequency (1D) 
            [needs to be set to get density on same grid as track]
        minxi= (0.25) minimum xi to consider
    OUTPUT:
        (xi,dens/smooth)
    """
    mT= sdf_pepper.meanTrack(apars,_mO=mO,coord='lb')
    mradec= bovy_coords.lb_to_radec(mT[0],mT[1],degree=True)
    mxieta= pal5_util.radec_to_pal5xieta(mradec[:,0],mradec[:,1],degree=True)
    outll= numpy.arange(minxi,maxxi,0.1)
    # Interpolate density
    ipll= interpolate.InterpolatedUnivariateSpline(mxieta[:,0],apars)
    ipdens= interpolate.InterpolatedUnivariateSpline(apars,dens/dens_smooth)
    return (outll,ipdens(ipll(outll)))



apar = np.linspace(0,3,100)
pdens, pomega = np.array([model.sp._densityAndOmega_par_approx(a) for a in apar]).T
sdens = np.array([model.smooth._density_par(a) for a in apar]).T
def convert_dens_to_obs_radec(sdf_pepper,apars,
                        dens,mO,dens_smooth,minra=0.25,maxra=14.35):
    """
    NAME:
        convert_dens_to_obs
    PURPOSE:
        Convert track to observed coordinates
    INPUT:
        sdf_pepper - streampepperdf object
        apars - parallel angles
        dens - density(apars)
        dens_smooth - smooth density(apars)
        mO= (None) mean parallel frequency (1D) 
            [needs to be set to get density on same grid as track]
        minxi= (0.25) minimum xi to consider
    OUTPUT:
        (xi,dens/smooth)
    """
    mT= sdf_pepper.meanTrack(apars,_mO=mO,coord='lb')
    mradec= bovy_coords.lb_to_radec(mT[0],mT[1],degree=True)
    mxieta= pal5_util.radec_to_pal5xieta(mradec[:,0],mradec[:,1],degree=True)
    outll= numpy.arange(minxi,maxxi,0.1)
    # Interpolate density
    ipll= interpolate.InterpolatedUnivariateSpline(mxieta[:,0],apars)
    ipdens= interpolate.InterpolatedUnivariateSpline(apars,dens/dens_smooth)
    return (outll,ipdens(ipll(outll)))







