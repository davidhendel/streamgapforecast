from astropy import table
from scipy.interpolate import interp1d
import numpy as np
import sys 
if sys.version_info[0] < 3:
    from __future__ import print_function

def build_isos_and_errors():
    castorfile = '/Users/hendel/projects/castor/iso_castor_0.00010_1e10.dat'
    compfile = '/Users/hendel/projects/castor/iso_comp_0.00010_1e10.dat'
    castoruv, castoruvd, castoru, castoruw, castorg, euclid_VIS, euclid_y, euclid_J, euclid_H, lsst_u, lsst_g,  lsst_r, lsst_i, lsst_z, lsst_y, wf_r062, wf_z087, wf_w146, wf_y106, wf_j129, wf_h158, wf_f184 = scipy.genfromtxt(compfile, skip_header=2, unpack=True)
    Age_yr, Mini_Mo, Mins_Mo, LogL_cgs, LogT_cgs, LogG_cgs, IMFweigth, lib, i_0, i_1, i_2, i_3, i_4, Vmag, Vnew, Cas_uv, Cas_uvD, Cas_u, Cas_uW, Cas_g = scipy.genfromtxt(castorfile, skip_header=2, unpack=True)

    zp = (Vmag+2.5*np.log10(Vnew))[0]
    Cas_uv  = -2.5*np.log10(Cas_uv)  + zp
    Cas_uvD = -2.5*np.log10(Cas_uvD) + zp
    Cas_u   = -2.5*np.log10(Cas_u)   + zp
    Cas_uW  = -2.5*np.log10(Cas_uW)  + zp
    Cas_g   = -2.5*np.log10(Cas_g)   + zp

    isotable=table.Table(
        data=[
        Age_yr[0:151], Mini_Mo[0:151], Mins_Mo[0:151], LogL_cgs[0:151], LogT_cgs[0:151], LogG_cgs[0:151], IMFweigth[0:151], 
        lib[0:151], i_0[0:151], i_1[0:151], i_2[0:151], i_3[0:151], i_4[0:151], Vmag[0:151], Vnew[0:151], 
        Cas_uv[0:151], Cas_uvD[0:151], Cas_u[0:151], Cas_uW[0:151], Cas_g[0:151], 
        castoruv, castoruvd, castoru, castoruw, castorg, 
        euclid_VIS, euclid_y, euclid_J, euclid_H, 
        lsst_u, lsst_g,  lsst_r, lsst_i, lsst_z, lsst_y, 
        wf_r062, wf_z087, wf_w146, wf_y106, wf_j129, wf_h158, wf_f184],
        names=[
        'Age_yr', 'Mini_Mo', 'Mins_Mo', 'LogL_cgs', 'LogT_cgs', 'LogG_cgs', 'IMFweigth', 
        'lib', 'i_0', 'i_1', 'i_2', 'i_3', 'i_4', 'Vmag', 'Vnew', 
        'Cas_uv', 'Cas_uvD', 'Cas_u', 'Cas_uW', 'Cas_g',
        'castoruv', 'castoruvd', 'castoru', 'castoruw', 'castorg', 
        'euclid_VIS', 'euclid_y', 'euclid_J', 'euclid_H', 
        'lsst_u', 'lsst_g',  'lsst_r', 'lsst_i', 'lsst_z', 'lsst_y', 
        'wf_r062', 'wf_z087', 'wf_w146', 'wf_y106', 'wf_j129', 'wf_h158', 'wf_f184']
        )

    isointerpdict = dict()
    for key in isotable.keys()[15:]:
        interp = interp1d(isotable['Mini_Mo'],isotable[key], bounds_error=False, fill_value='extrapolate')
        isointerpdict[key]=interp


    # https://www.eso.org/~ohainaut/ccd/sn.html - error on mag = 1/SNR
    errormodels = dict()
    thisfilter='g'
    errormodels['LSST'] = interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), thisfilter, survey='LSST'), bounds_error=False, fill_value=Inf)
    errormodels['LSST10'] = interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), thisfilter, survey='LSST10'), bounds_error=False, fill_value=Inf)
    errormodels['CFHT'] = interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), thisfilter, survey='CFHT'), bounds_error=False, fill_value=Inf)
    errormodels['SDSS'] = interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), thisfilter, survey='SDSS'), bounds_error=False, fill_value=Inf)

    castormag, castorsnr = scipy.genfromtxt('/Users/hendel/projects/castor/snrs/snexptime_1335_g.out', skip_header=0, unpack=True, usecols=(0,2))
    errormodels['CASTOR'] = interp1d(castormag, 1/castorsnr, bounds_error=False, fill_value=Inf)

    return isointerpdict, errormodels


def process_galaxia(name=None):
    import ebf
    import astropy.table as atpy
    import numpy as np
    import string

    if name == None:
        name = '/Users/hendel/modules/Galaxia/GalaxiaData/Examples/test_gap_mock.ebf'
    namebase = string.split(name,'.')[0]
    F = ebf.read(name)
    dm = 5*np.log10(F['rad']*1e3)-5  
    u,g,r,i,z = [F['sdss_%s'%_]+dm for _ in ['u','g','r','i','z']]

    tab = atpy.Table()
    #for filt in ['u','g','r','i','z']:
    #    tab.add_column(atpy.Column(eval(filt),filt))
    tab.add_column(atpy.Column(eval('g'),'g'))
    tab.add_column(atpy.Column(eval('r'),'r'))
    tab.add_column(atpy.Column(eval('g')-eval('r'),'g-r'))
    tab.write(namebase+'.fits',overwrite=True)



class isoinstance():
    def __init__(self,isoname="iso_a12.0_z0.00020.dat"):
        self.iso = read_girardi.read_girardi(isoname)
        xind = self.iso['stage'] <= 3  # 3 = cut the horizontal branch, 6 = cut AGB
        self.iso_g = self.iso['DES-g'][xind]
        self.iso_r = self.iso['DES-r'][xind]
        self.iso_color_interp = interp1d(self.iso_g,self.iso_g-self.iso_r, bounds_error=False, fill_value=Inf)


class bginstance():
    def __init__(self,bg_name = '/Users/hendel/modules/Galaxia/GalaxiaData/Examples/test_gap_mock.fits'):
        data = fits.open(bg_name)
        self.g, self.r, self.gr = data[1].data['g'], data[1].data['r'], data[1].data['g-r']
        data.close()


def bg_counts_fits(dm, magerror_mod=1., filter='g', survey='LSST10', iso=None, errormodel=None, maxerr=0.1, bg=None, maglim=28., area=100.):
    if bg==None:
        bg = bginstance()
    if iso==None:
        iso = isoinstance()

    if errormodel==None:
        print("warning: not using pre-calculated error model")
        magerror_interp = interp1d(np.linspace(14,28,100),
            getMagErrVec(np.linspace(14,28,100), filter, survey=survey), bounds_error=False, fill_value=Inf)
    else: magerror_interp=errormodel


    bg_sel = ((abs((iso.iso_color_interp(bg.g-dm) - (bg.gr))/(magerror_interp(bg.g)*magerror_mod)) < 2.) & (magerror_interp(bg.g)<maxerr) & (bg.g<maglim))
    #per sq degree
    realmaglim = getMagLimit(filter, survey=survey)
    if maglim > realmaglim:
        print('warning: maglim fainter than survey depth')
    return np.sum(bg_sel)/area


def gen_bg_counts_interp(dms = np.linspace(15,19,25), magerror_mods=[1], survey='LSST10', filter='g', maxerr=0.1):
    from scipy.interpolate import RegularGridInterpolator

    maglim = getMagLimit(filter, survey=survey)
    maglims = np.linspace(maglim-5,maglim, 25)

    bg_name = '/Users/hendel/modules/Galaxia/GalaxiaData/Examples/test_gap_mock.fits'

    data = fits.open(bg_name)
    bg_g, bg_r, bg_gr = data[1].data['g'], data[1].data['r'], data[1].data['g-r']
    data.close()

    isoname = "iso_a12.0_z0.00020.dat"
    iso = read_girardi.read_girardi(isoname)
    xind = iso['stage'] <= 3  # 3 = cut the horizontal branch, 6 = cut AGB
    iso_g = iso['DES-g'][xind]
    iso_r = iso['DES-r'][xind]

    iso_color_interp = interp1d(iso_g,iso_g-iso_r, bounds_error=False, fill_value=Inf)

    magerror_interp = interp1d(np.linspace(14,28,100),
        getMagErrVec(np.linspace(14,28,100), filter, survey=survey), bounds_error=False, fill_value=Inf)

    def bg_count_func(dm, magerror_mod, maglim, area=100.):
        bg_sel = ((abs((iso_color_interp(bg_g-dm) - (bg_gr))/(magerror_interp(bg_g)*magerror_mod)) < 2.) & (magerror_interp(bg_g)<maxerr) & (bg_g<maglim))
        return np.sum(bg_sel)/area

    data = np.zeros((len(dms), len(magerror_mods), len(maglims)))
    for ii in np.arange(len(dms)):
        for jj in np.arange(len(magerror_mods)):
            for kk in np.arange(len(maglims)):
                data[ii,jj,kk] = bg_count_func(dms[ii], magerror_mods[jj], maglims[kk])

    interp = RegularGridInterpolator((array(dms), array(magerror_mods), array(maglims)),data)
    return interp


#def bg_counts(dm, magerror_mod=1., filter='g', survey='LSST10', maxerr=0.1, bg_name=None, maglim=28., area=100.):
#	if bg_name==None:
#		bg_name = '/Users/hendel/modules/Galaxia/GalaxiaData/Examples/test_gap_mock.ebf'
#	import ebf
#	F = ebf.read(bg_name)
#	dms = 5*np.log10(F['rad']*1e3)-5  
#	bg_u,bg_g,bg_r,bg_i,bg_z = [F['sdss_%s'%_]+dms for _ in ['u','g','r','i','z']]
#
#	magerror_interp = interp1d(np.linspace(14,28,100),
#		getMagErrVec(np.linspace(14,28,100), filter, survey=survey), bounds_error=False, fill_value=Inf)
#
#
#	isoname = "iso_a12.0_z0.00020.dat"
#	iso = read_girardi.read_girardi(isoname)
#	xind = iso['stage'] <= 3  # 3 = cut the horizontal branch, 6 = cut AGB
#	iso_g = iso['DES-g'][xind]
#	iso_r = iso['DES-r'][xind]
#
#	iso_color_interp = interp1d(iso_g,iso_g-iso_r, bounds_error=False, fill_value=Inf)
#
#	bg_sel = ((abs((iso_color_interp(bg_g-dm) - (bg_g-bg_r))/(magerror_interp(bg_g)*magerror_mod)) < 2.) & (magerror_interp(bg_g)<maxerr) & (bg_g<maglim))
#	#per sq degree
#	realmaglim = getMagLimit(filter, survey=survey)
#	if maglim > realmaglim:
#		print 'warning: maglim fainter than survey depth'
#	return np.sum(bg_sel)/area


def dist_and_power(data, data_rs, err=None, bkg=0., degree=3, nbins=50, xirange=[-20,0], scaling='spectrum', col='k', plot=False, verbose=False):
    #data[:,0] = data[:,0]*180./np.pi
    #plt.subplot(1,2,1)
    t0 = time.time()
    #counts, bins, patches = plt.hist(data[:,0], bins=np.linspace(xirange[0], xirange[1], nbins), histtype='step', lw=2, color='k')
    counts, bins = np.histogram(data[:,0], bins=np.linspace(xirange[0], xirange[1], nbins))

    #counts = counts + np.random.poisson(bkg, size=len(counts))
    centroids = (bins[1:] + bins[:-1]) / 2.
    dists = centroids*0.
    dist_errs= centroids*0.

    for i in np.arange(len(bins)-1):
        mask = ((data[:,0]>bins[i])&(data[:,0]<bins[i+1]))
        dists[i] = np.mean(data_rs[mask])
        dist_errs[i] = np.std(data_rs[mask])/np.sqrt(np.sum(mask))


    #newmask = (isfinite(dists) & isfinite(dist_errs) & (dist_errs>0))
    #centroids = centroids[newmask]
    #dists = dists[newmask]
    #dist_errs = dist_errs[newmask]

    #if err==None:
    #    err = np.sqrt(counts)
    #plt.errorbar(centroids, counts, yerr=err, capthick=0, c='k')

    #x = numpy.linspace(min(bins), max(bins), 100)
    pp= Polynomial.fit(centroids, dists, deg=degree, w=1./dist_errs)
    #plt.plot(x,pp(x),'-', c=col)

    tdata= (dists)/pp(centroids)
    terr = dist_errs/pp(centroids)

    t1 = time.time()
    if verbose==True: print("time for tdata and terr:", t1-t0)

    t0 = time.time()
    from scipy import signal
    px, py= signal.csd(tdata,tdata,
                        fs=1./(centroids[1]-centroids[0]),scaling=scaling,
                        nperseg=len(centroids))
    py= py.real
    px= 1./px
    py= numpy.sqrt(py*(centroids[-1]-centroids[0]))

    t1 = time.time()
    if verbose==True: print("time for power spectrum:", t1-t0)

    # Perform simulations of the noise to determine the power in the noise
    t0 = time.time()
    nerrsim= 10000
    ppy_err= numpy.empty((nerrsim,len(px)))
    for ii in range(nerrsim):
        tmock= terr*numpy.random.normal(size=len(centroids))
        ppy_err[ii]= signal.csd(tmock,tmock,
                                fs=1./(centroids[1]-centroids[0]),scaling=scaling,
                                nperseg=len(centroids))[1].real

    py_err= numpy.sqrt(numpy.median(ppy_err,axis=0)*(centroids[-1]-centroids[0]))
    pcut= 0.0 # Only trust points above this, then remove noise - Jo has 0.4
    t1 = time.time()
    if verbose==True: print("time for error analysis", t1-t0)

    if plot==True:
        plt.subplot(111)
        loglog(px[py>pcut],numpy.sqrt(py[py>pcut]**2.-py_err[py>pcut]**2.),
               marker='o',zorder=0,ls='none', markersize=5, color=col)
        errorbar(px[(py<pcut)],numpy.amax(numpy.array([py,py_err]),axis=0)[py<pcut],
                 yerr=numpy.array([.1+0.*px[(py<pcut)],.1+0.*px[(py<pcut)]]),
                 uplims=True,capthick=2.,ls='none',color='k',zorder=0)
        loglog(px,py_err,lw=2.,color=col, alpha=0.5,zorder=-2)

    return tdata, terr, px, py, py_err


def pos_and_power(data, data_rs, err=None, bkg=0., degree=3, nbins=50, xirange=[-20,0], scaling='spectrum', col='k', plot=False, verbose=False):
    #data[:,0] = data[:,0]*180./np.pi
    #plt.subplot(1,2,1)
    t0 = time.time()
    #counts, bins, patches = plt.hist(data[:,0], bins=np.linspace(xirange[0], xirange[1], nbins), histtype='step', lw=2, color='k')
    counts, bins = np.histogram(data[:,0], bins=np.linspace(xirange[0], xirange[1], nbins))

    #counts = counts + np.random.poisson(bkg, size=len(counts))
    centroids = (bins[1:] + bins[:-1]) / 2.
    pos = centroids*0.
    pos_errs= centroids*0.

    for i in np.arange(len(bins)-1):
        mask = ((data[:,0]>bins[i])&(data[:,0]<bins[i+1]))
        pos[i] = np.mean(data[:,1][mask])
        pos_errs[i] = np.std(data[:,1])/np.sqrt(np.sum(mask))


    #newmask = (isfinite(dists) & isfinite(dist_errs) & (dist_errs>0))
    #centroids = centroids[newmask]
    #dists = dists[newmask]
    #dist_errs = dist_errs[newmask]

    #if err==None:
    #    err = np.sqrt(counts)
    #plt.errorbar(centroids, counts, yerr=err, capthick=0, c='k')

    #x = numpy.linspace(min(bins), max(bins), 100)
    pp= Polynomial.fit(centroids, pos, deg=degree, w=1./pos_errs)
    #plt.plot(x,pp(x),'-', c=col)

    tdata= (pos)/pp(centroids)
    terr = pos_errs/pp(centroids)

    t1 = time.time()
    if verbose==True: print("time for tdata and terr:", t1-t0)

    t0 = time.time()
    from scipy import signal
    px, py= signal.csd(tdata,tdata,
                        fs=1./(centroids[1]-centroids[0]),scaling=scaling,
                        nperseg=len(centroids))
    py= py.real
    px= 1./px
    py= numpy.sqrt(py*(centroids[-1]-centroids[0]))

    t1 = time.time()
    if verbose==True: print("time for power spectrum:", t1-t0)

    # Perform simulations of the noise to determine the power in the noise
    t0 = time.time()
    nerrsim= 10000
    ppy_err= numpy.empty((nerrsim,len(px)))
    for ii in range(nerrsim):
        tmock= terr*numpy.random.normal(size=len(centroids))
        ppy_err[ii]= signal.csd(tmock,tmock,
                                fs=1./(centroids[1]-centroids[0]),scaling=scaling,
                                nperseg=len(centroids))[1].real

    py_err= numpy.sqrt(numpy.median(ppy_err,axis=0)*(centroids[-1]-centroids[0]))
    pcut= 0.0 # Only trust points above this, then remove noise - Jo has 0.4
    t1 = time.time()
    if verbose==True: print("time for error analysis", t1-t0)

    if plot==True:
        plt.subplot(111)
        loglog(px[py>pcut],numpy.sqrt(py[py>pcut]**2.-py_err[py>pcut]**2.),
               marker='o',zorder=0,ls='none', markersize=5, color=col)
        errorbar(px[(py<pcut)],numpy.amax(numpy.array([py,py_err]),axis=0)[py<pcut],
                 yerr=numpy.array([.1+0.*px[(py<pcut)],.1+0.*px[(py<pcut)]]),
                 uplims=True,capthick=2.,ls='none',color='k',zorder=0)
        loglog(px,py_err,lw=2.,color=col, alpha=0.5,zorder=-2)

    return tdata, terr, px, py, py_err



def dens_and_power(data, err=None, bkg=0., degree=3, nbins=50, xirange=[-20,0], scaling='spectrum', col='k', plot=False, verbose=False):
    #data[:,0] = data[:,0]*180./np.pi
    #plt.subplot(1,2,1)
    t0 = time.time()
    #counts, bins, patches = plt.hist(data[:,0],bins=np.linspace(xirange[0], xirange[1], nbins), histtype='step', lw=2, color='k')
    counts, bins = np.histogram(data[:,0],bins=np.linspace(xirange[0], xirange[1], nbins))
    counts = counts + np.random.poisson(bkg, size=len(counts))
    centroids = (bins[1:] + bins[:-1]) / 2.

    if err==None:
    	err = np.sqrt(counts)
    #plt.errorbar(centroids, counts, yerr=err, capthick=0, c='k')

    #x = numpy.linspace(min(bins), max(bins), 100)
    pp= Polynomial.fit(centroids,counts-bkg, deg=degree, w=1./err)
    #plt.plot(x,pp(x),'-', c=col)

    tdata= (counts-bkg)/pp(centroids)
    terr = err/pp(centroids)

    t1 = time.time()
    if verbose==True: print("time for tdata and terr:", t1-t0)

    t0 = time.time()
    from scipy import signal
    px, py= signal.csd(tdata,tdata,
    					fs=1./(centroids[1]-centroids[0]),scaling=scaling,
    					nperseg=len(centroids))
    py= py.real
    px= 1./px
    py= numpy.sqrt(py*(centroids[-1]-centroids[0]))

    t1 = time.time()
    if verbose==True: print("time for power spectrum:", t1-t0)

    # Perform simulations of the noise to determine the power in the noise
    t0 = time.time()
    nerrsim= 10000
    ppy_err= numpy.empty((nerrsim,len(px)))
    for ii in range(nerrsim):
        tmock= terr*numpy.random.normal(size=len(centroids))
        ppy_err[ii]= signal.csd(tmock,tmock,
                                fs=1./(centroids[1]-centroids[0]),scaling=scaling,
                                nperseg=len(centroids))[1].real

    py_err= numpy.sqrt(numpy.median(ppy_err,axis=0)*(centroids[-1]-centroids[0]))
    pcut= 0.0 # Only trust points above this, then remove noise - Jo has 0.4
    t1 = time.time()
    if verbose==True: print("time for error analysis", t1-t0)

    if plot==True:
    	plt.subplot(111)
    	loglog(px[py>pcut],numpy.sqrt(py[py>pcut]**2.-py_err[py>pcut]**2.),
    	       marker='o',zorder=0,ls='none', markersize=5, color=col)
    	errorbar(px[(py<pcut)],numpy.amax(numpy.array([py,py_err]),axis=0)[py<pcut],
    	         yerr=numpy.array([.1+0.*px[(py<pcut)],.1+0.*px[(py<pcut)]]),
    	         uplims=True,capthick=2.,ls='none',color='k',zorder=0)
    	loglog(px,py_err,lw=2.,color=col, alpha=0.5,zorder=-2)

    return tdata, terr, px, py, py_err


def sp_stream_samples(sp, nsample =20000, lb=True, massexp=-2, GMmod=1., massrange = [6,9], cutoff =5., ratemod = 3., do_sample=True):
    massexp=massexp
    sample_GM= lambda: GMmod*powerlaw_wcutoff(massrange,cutoff)
    rate_range= numpy.arange(massrange[0]+0.5,massrange[1]+0.5,1)
    rate = ratemod*numpy.sum([dNencdm(sp,10.**r,Xrs=3.,plummer=False,rsfac=1.,sigma=120.) for r in rate_range])
    sample_rs= lambda x: rs(x*bovy_conversion.mass_in_1010msol(V0,R0)*10.**10.,plummer=False,rsfac=1.)
    ns= 0
    sp.simulate(rate=rate,sample_GM=sample_GM,sample_rs=sample_rs,Xrs=3.,sigma=120./220.)
    sp._useInterp=True

    if do_sample==True:
        sp_sample= sp.sample(n=nsample,lb=lb)
        spc = SkyCoord(sp_sample[0]*u.deg,sp_sample[1]*u.deg,distance=sp_sample[2]*u.kpc,frame='galactic')
        spxi = radec_to_pal5xieta(spc.icrs.ra, spc.icrs.dec)
        return sp_sample, spxi, spc


def assign_mass_and_color(xydata, rdata, min_g_mag = 28., max_g_mag = 14.6, slope = -0.5):

	isoname = "iso_a12.0_z0.00020.dat"
	iso = read_girardi.read_girardi(isoname)
	xind = iso['stage'] <= 3  # 3 = cut the horizontal branch, 6 = cut AGB
	maxmass = np.max(iso['M_ini'][xind])
	minmass = np.min(iso['M_ini'][xind])

	minmass_from_g = np.interp((min_g_mag-(5*np.log10(rdata*1e3)-5)),iso['DES-g'][xind][::-1],iso['M_ini'][xind][::-1]) 
	#not needed unless doing AGB
	#maxmass_from_g = np.interp((max_g_mag-(5*np.log10(rdata*1e3)-5)),iso['DES-g'][xind][::-1],iso['M_ini'][xind][::-1]) 

	nstars = len(xydata)
	ms = np.zeros(nstars)
	gstars = np.zeros(nstars)
	rstars = np.zeros(nstars)
	istars = np.zeros(nstars)

	for i in np.arange(nstars):
		x1=maxmass #np.minimum(maxmass, maxmass_from_g[i])
		x0=np.maximum(minmass, minmass_from_g[i])
		y=np.random.uniform(size=1)
		ms[i] = ((x1**(slope+1) - x0**(slope+1))*y + x0**(slope+1))**(1/(slope+1))
		gstars[i] = np.interp(ms[i],iso['M_ini'],iso['DES-g'])+(5*np.log10(rdata[i]*1e3)-5)
		rstars[i] = np.interp(ms[i],iso['M_ini'],iso['DES-r'])+(5*np.log10(rdata[i]*1e3)-5)
		istars[i] = np.interp(ms[i],iso['M_ini'],iso['DES-i'])+(5*np.log10(rdata[i]*1e3)-5)

	return ms, gstars, rstars, istars


#timpacts= parse_times(options.timpacts,options.age)
#if options.timpacts == '64sampling':
#    # We've cached this one
#    with open('pal5_64sampling.pkl','rb') as savefile:
#        sdf_smooth= pickle.load(savefile)
#        sdf_pepper= pickle.load(savefile)
def parse_times(times,age):
    if 'sampling' in times:
        nsam= int(times.split('sampling')[0])
        return [float(ti)/bovy_conversion.time_in_Gyr(V0,R0)
                for ti in numpy.arange(1,nsam+1)/(nsam+1.)*age]
    return [float(ti)/bovy_conversion.time_in_Gyr(V0,R0)
            for ti in times.split(',')]

####################################
####################################
####################################
#mag error functions



import astropy.table as atpy
import read_girardi
import numpy as np
import scipy.spatial
import astropy.units as auni
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize


def betw(x, x1, x2): return (x >= x1) & (x <= x2)


def getMagErr(mag, filt, survey='LSST', calibration_err=0.01):
    """
    Parameters
    ----------
    mag: float
        Magnitude
    filt: str
        Filter [g or r]
    survey: str
        Survey
    calibration_err: float
        Assumed systematic, in mag
    nYrObs: float
        parameter to be passed to the error model; default is 1 year

    Returns:
    -------
        err: float
        Magnitude uncertainty
    """
    if survey == 'CASTOR':
        #1335 sec exposures, from primary survey
        castormag, castorsnr = scipy.genfromtxt('/Users/hendel/projects/castor/snrs/snexptime_1335_'+filt+'.out', skip_header=0, unpack=True, usecols=(0,2))
        magerr = np.interp(mag, castormag, 1./castorsnr)
        return magerr
    if survey == 'LSST':
        import photErrorModel as pem
        lem = pem.LSSTErrorModel()
        lem.nYrObs = 1.
        #minmagerr = 0.01
        magerr = lem.getMagError(mag, 'LSST_' + filt)
        return magerr
    if survey == 'LSST10':
        import photErrorModel as pem
        lem = pem.LSSTErrorModel()
        lem.nYrObs = 10.
        #minmagerr = 0.01
        magerr = lem.getMagError(mag, 'LSST_' + filt)
        return magerr
    if survey == 'CFHT':
        g, g_err, r, r_err = np.loadtxt('CFHT_photoerr.txt', skiprows=1, unpack=True)
        if filt == 'g':
            magerr = np.interp(mag, g, g_err)
        if filt == 'r':
            magerr = np.interp(mag, r, r_err)
        return np.sqrt(magerr**2 + calibration_err**2)
    if survey == 'SDSS':
        # this is DR9 photometry
        g, g_err, r, r_err = np.loadtxt('SDSS_photoerr.txt', skiprows=1, unpack=True)
        if filt == 'g':
            magerr = np.interp(mag, g, g_err)
        if filt == 'r':
            magerr = np.interp(mag, r, r_err)
        return np.sqrt(magerr**2 + calibration_err**2)
    else:
        print("No error model for this survey")


def getMagErrVec(mag, filt, survey='LSST'):
    """ 
    Parameters:
    ----------
    mag: numpy
        The array of magnitudes
    filt: str
        The filter of observations
    Returns:
    err: numpy array
        The magnitude uncertainty

    """
    maggrid = np.linspace(14, 28, 1000)
    res = [getMagErr(m, filt, survey) for m in maggrid]
    res = scipy.interpolate.UnivariateSpline(maggrid, res, s=0)(mag)
    return res


def getMagLimit(filt, survey='LSST', maxerr=0.1):
    "A sophisticated calculation of LSST magntude limit"
    xgrid = np.linspace(14, 28, 1000)
    err = getMagErrVec(xgrid, filt, survey)
    xid = np.argmax(err * (err < maxerr))
    return xgrid[xid]




######################################################################
######################################################################
######################################################################
R0,V0= 8., 220.
save_figures= False
import galpy.orbit as Orbit
import numpy
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.orbit import Orbit
from galpy.df import streamdf, streamgapdf
from streampepperdf import streampepperdf
from galpy.util import bovy_conversion, bovy_coords

#prog= Orbit([229.018,-0.124,23.2,-2.296,-2.257,-58.7],radec=True,ro=R0,vo=V0,
#            solarmotion=[-11.1,24.,7.25])
#aAI= actionAngleIsochroneApprox(pot=MWPotential2014,b=0.8)

sigv= 0.5
_RAPAL5= 229.018/180.*numpy.pi
_DECPAL5= -0.124/180.*numpy.pi
_TPAL5= numpy.dot(numpy.array([[numpy.cos(_DECPAL5),0.,numpy.sin(_DECPAL5)],
                               [0.,1.,0.],
                               [-numpy.sin(_DECPAL5),0.,numpy.cos(_DECPAL5)]]),
                  numpy.array([[numpy.cos(_RAPAL5),numpy.sin(_RAPAL5),0.],
                               [-numpy.sin(_RAPAL5),numpy.cos(_RAPAL5),0.],
                               [0.,0.,1.]]))
@bovy_coords.scalarDecorator
@bovy_coords.degreeDecorator([0,1],[0,1])
def radec_to_pal5xieta(ra,dec,degree=False):
    XYZ= numpy.array([numpy.cos(dec)*numpy.cos(ra),
                      numpy.cos(dec)*numpy.sin(ra),
                      numpy.sin(dec)])
    phiXYZ= numpy.dot(_TPAL5,XYZ)
    phi2= numpy.arcsin(phiXYZ[2])
    phi1= numpy.arctan2(phiXYZ[1],phiXYZ[0])
    return numpy.array([phi1,phi2]).T

def parse_times(times,age):
    if 'sampling' in times:
        nsam= int(times.split('sampling')[0])
        return [float(ti)/bovy_conversion.time_in_Gyr(V0,R0)
                for ti in numpy.arange(1,nsam+1)/(nsam+1.)*age]
    return [float(ti)/bovy_conversion.time_in_Gyr(V0,R0)
            for ti in times.split(',')]
def parse_mass(mass):   
    return [float(m) for m in mass.split(',')]

# Functions to sample
def nsubhalo(m):
    return 0.3*(10.**6.5/m)
def rs(m,plummer=False,rsfac=1.):
    if plummer:
        return 1.62*rsfac/R0*(m/10.**8.)**0.5
    else:
        return 1.05*rsfac/R0*(m/10.**8.)**0.5
def dNencdm(sdf_pepper,m,Xrs=3.,plummer=False,rsfac=1.,sigma=120.):
    return sdf_pepper.subhalo_encounters(\
        sigma=sigma/V0,nsubhalo=nsubhalo(m),
        bmax=Xrs*rs(m,plummer=plummer,rsfac=rsfac))
def powerlaw_wcutoff(massrange,cutoff):
    accept= False
    while not accept:
        prop= (10.**-(massrange[0]/2.)+(10.**-(massrange[1]/2.)\
                         -10.**(-massrange[0]/2.))\
                   *numpy.random.uniform())**-2.
        if numpy.random.uniform() < numpy.exp(-10.**cutoff/prop):
            accept= True
    return prop/bovy_conversion.mass_in_msol(V0,R0)



def getIsoCurve(iso, magstep=0.01):
    """
    Returns the list of points sampling along the isochrone

    Arguments:
    ---------
    iso: dict
        Dictionary with the Girardi isochrone
    magstep: float(optional)
        The step in magntidues along the isochrone
    Returns:
    -------
    gcurve,rcurve: Tuple of numpy arrays
        The tupe of arrays of magnitudes in g and r going along the isochrone
    """
    mini = iso['M_ini']
    g = iso['DES-g']
    r = iso['DES-r']
    res_g, res_r = [], []
    for i in range(len(mini) - 1):
        l_g, l_r = g[i], r[i]
        r_g, r_r = g[i + 1], r[i + 1]
        npt = max(abs(r_g - l_g), abs(r_r - l_r)) / magstep + 2
        ggrid = np.linspace(l_g, r_g, npt)
        rgrid = np.linspace(l_r, r_r, npt)
        res_g.append(ggrid)
        res_r.append(rgrid)
    res_g, res_r = [np.concatenate(_) for _ in [res_g, res_r]]
    return res_g, res_r



