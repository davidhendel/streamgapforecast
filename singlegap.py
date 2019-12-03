#do single gaps better
import pal5_mock_ps
import iso_handling
import astropy.table as atpy
import read_girardi
import numpy as np
import scipy.spatial
import astropy.units as auni
import simple_stream_model as sss
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize


errormodels = iso_handling.load_errormodels()
isodict = iso_handling.load_iso_interps(remake=True)
try: 
    bgdict = np.load('/Users/hendel/projects/streamgaps/streampepper/bgdict.npy').item()
except:
    bgdict = iso_handling.gen_bg_counts_interp(isodict=isodict,errormodels=errormodels, verbose=True)



def find_gap_fill_time(mass, dist, **kwargs):
    """
    Arguments:
    ---------
    mass:
        subhalos mass in units of 1e7 Msun
    dist:
        distance of stream in kpc

    Returns:
    ---
    time:
        time to fill gap
    """
    def F(x):
        len_gap_kpc = np.deg2rad(sss.gap_size(mass, dist=dist, timpact=x, **kwargs)) / auni.rad * dist
        vel = 1  # kms
        len_gap_km = 3.086e16 * len_gap_kpc
        time_fill_gap = len_gap_km / vel / 3.15e7 / 1e9  # in gyr
        return time_fill_gap / x - 1

    R = scipy.optimize.root(F, 0.1)
    time = R['x'][0]  # time to fill gap
    return time


def find_gap_size_depth(mass, dist, maxt=1, **kwargs):
    # if gap_fill = True
    """
    Arguments:
    ---------
    mass:
        subhalos mass in units of 1e7 Msun
    dist:
        distance of stream in kpc

    Returns:
    ---
    (len_gap_deg, depth_gap):
        length of gap in degrees
        gap depth
    """

    # time to fill the gap created by subhalo of the given mass and impact parameter
    time = find_gap_fill_time(mass, dist, **kwargs)
    # cap time after the impact to the time required to fill the gap
    time = min(time, maxt)  # time to fill gap if less than max t (default 0.5 Gyr)
    #print ('x',F(0.5),F(0.001),F(10),time,maxt)
    print('time', time, mass)  # ,maxt)

    len_gap_deg = sss.gap_size(mass, dist=dist, timpact=float(time), **kwargs) / auni.deg
    depth_gap = 1 - sss.gap_depth(mass, timpact=time, **kwargs)

    return len_gap_deg, depth_gap

def predict_gap_depths(multpal5, distance_kpc, survey='LSST', bands='gr', width_pc=20., maglim=None, errfactor=1.,
	                   timpact=1, gap_fill=True, bgdict=None, galdict=None, model=None):
	"""
	Arguments:
	---------
	mu: real
	    Surface brightness of the stream in mag/sq.arcsec^2
	distance_kpc: real
	    Distance to the stream in kpc
	survey: str
	    Name of the survey (LSST, LSST10, CFHT, SDSS)
	width_pc: real
	    The width of the stream in pc
	timpact: real
	    The time of impact in Gyr
	gap_fill: bool
	    If true we take into account the filling of the gaps. I.e. we 
	    use the depth of the gap and the size of the gap up to a point
	    in time when the gap is supposed to be filled (assuming that it 
	    fills with 1km/s velocity)
	mockfile: string
	    Galaxia background file; l=90,b=30, 60, 90 are included
	Returns:
	---
	(masses,tdepths,odepths): Tuple of 3 numpy arrays
	    The array of halo masses
	    The array of theoretically predicted gap depths
	    The array of potentially observable gap depths
	"""
	if survey == 'WFIRST': bands = 'zh'
	elif survey == 'CASTOR': bands = 'ug'
	if model == None:
		model = pal5_mock_ps.pal5_stream_instance()
		model.sample(nsample=int(10000*multpal5))
		model.assign_masses()

	width_deg = np.rad2deg(float(width_pc) / float(distance_kpc) / 1e3)
	mgrid = 10**np.linspace(3., 10, 100)
	mgrid7 = mgrid / 1e7
	gap_depths = np.zeros(len(mgrid))
	gap_sizes_deg = np.zeros(len(mgrid))
	for i, curm in enumerate(mgrid7):
		gap_sizes_deg[i], gap_depths[i] = find_gap_size_depth(curm, dist=distance_kpc, maxt=timpact)

	if maglim is None:
	    maglim_g = iso_handling.getMagLimit(bands[0], survey)
	    maglim_r = iso_handling.getMagLimit(bands[1], survey)
	else:
	    maglim_g, maglim_r = [maglim] * 2
	#dens_stream = snc.nstar_cal(mu, distance_kpc, maglim_g=maglim_g,
	#                            maglim_r=maglim_r)
	#dens_bg = get_mock_density(distance_kpc, isoname, survey, errfactor=errfactor,
	#                           mockfile=mockfile, mockarea=mockarea,
	#                           maglim_g=maglim_g, maglim_r=maglim_r)
	key=survey
	if 'LSST' in key:
	    mag1 = isodict['lsst_g-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
	    magerror1 = errormodels[key]['g'](mag1)
	    mag2 = isodict['lsst_r-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
	    magerror2 = errormodels[key]['r'](mag2)
	elif key == 'CASTOR':
	    mag1 = isodict['castor_u-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
	    magerror1 = errormodels[key]['u'](mag1)
	    mag2 = isodict['castor_g-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
	    magerror2 = errormodels[key]['g'](mag2)
	elif key == 'WFIRST':
	    mag1 = isodict['Z087-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
	    magerror1 = errormodels[key]['z'](mag1)
	    mag2 = isodict['H158-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
	    magerror2 = errormodels[key]['h'](mag2)
	else:
	    mag1 = isodict['sdss_g-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
	    magerror1 = errormodels[key]['g'](mag1)
	    mag2 = isodict['sdss_r-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
	    magerror2 = errormodels[key]['r'](mag2)


	omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1))]
	omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1))]

	print np.max(omag2)
	if galdict==None:
		dens_bg = bgdict[key]((5*np.log10(distance_kpc*1e3)-5,np.max(omag2)))
	else: dens_bg = bgdict[key]((5*np.log10(distance_kpc*1e3)-5,np.max(omag2))) + galdict[key]((5*np.log10(distance_kpc*1e3)-5,np.max(omag2)))
	dens_stream = len(omag1)/(20*.1)*multpal5

	print('Background/stream density [stars/sq.deg]', dens_bg, dens_stream)
	max_gap_deg = 10  # this the maximum gap length that we consider reasonable
	N = len(gap_sizes_deg)
	detfracs = np.zeros(N)
	for i in range(N):
	    area = 2 * width_deg * gap_sizes_deg[i]
	    # twice the width and the length of the gap
	    nbg = dens_bg * area
	    nstr = dens_stream * area
	    detfrac = 5 * np.sqrt(nbg + nstr) / nstr
	    print('Nstream', nstr, 'Nbg', nbg)#, 'Detfrac', detfrac, 'Area', area )
	    # this is smallest gap depth that we could detect
	    # we the poisson noise on density is sqrt(nbg+nstr)
	    # and the stream density (per bin) is nstr
	    detfracs[i] = detfrac
	    # if gap_sizes_deg[i] > max_gap_deg:
	    #     detfracs[i] = np.nan
	return (mgrid, gap_depths, detfracs)


def wdm_mass(mhalo, h=0.7):
    # input solar mass, return keV
    # return (mhalo * (h / 1e11))**-0.25 # 1512.05349
    #return (mhalo / 5.5e10) ** (1 / -3.33)  # 1707.04256

    #Ethan's new relation
    #result in kev
    mwdm = 2.9 * (mhalo/1e9)**-0.264
    return mwdm


def halo_mass(mwdm, h=0.7):
    #return 5.5e10 * (mwdm)**(-3.33)

    #Ethan's new relation
    mhalo = ((mwdm/2.9)**(1/-0.264))*1e9
    return mhalo


def final_plot(filename=None, mus=[.25,.5,.75,1.,1.25], surveys=['LSST','LSST10','WFIRST'], w=150., b=1., maglim=None, lat=60., gap_fill=True):
    #output = np.genfromtxt('output.txt', unpack=True, delimiter=', ', dtype=None, names=['dist', 'w', 'b', 'maglim', 'lat', 'gap_fill', 'survey', 'mu', 'mass'], encoding='bytes')

    colors = ['seagreen', 'orange', 'darkslateblue']
    markers = ['o','X','d']
    plt.figure()
    ret = np.zeros((3,5))

    model = pal5_mock_ps.pal5_stream_instance()
    model.sample(nsample=int(10000))
    model.assign_masses()

    for i, survey in enumerate(surveys):
        if survey == 'WFIRST':  iso_handling.getMagLimit('z', survey)
        elif survey == 'CASTOR': iso_handling.getMagLimit('u', survey)
        else:maglim = iso_handling.getMagLimit('g', survey)
        for k, mu in enumerate(mus):


            for j, distance in enumerate([20.,40.,60.]):
            	print survey, mu, distance
                mass, gapt, gapo = predict_gap_depths(mu, distance, survey, width_pc=20, maglim=None,
                                                      timpact=0.5, gap_fill=gap_fill, bgdict=bgdict, model=model)
                xind = np.isfinite(gapo / gapt)
                II1 = scipy.interpolate.UnivariateSpline(np.log10(mass)[xind], (gapo / gapt - 1)[xind], s=0)
                R = scipy.optimize.root(II1, 6)
                ret[j,k] = (10**R['x'])

        label = r'$\mathrm{%s}$' % survey
        plt.semilogy(mus, ret[1,:], 'o-', label=label, c=colors[i],marker = markers[i], zorder=1)  # label='d = %d, w = %d, b = %d' % (distance, w, b)
        plt.fill_between(mus, ret[0,:],ret[2,:], alpha=0.2, color=colors[i],zorder=0)

    plt.legend(loc='upper right', fontsize=12)
    plt.ylabel(r'$M_{\mathrm{vir}}(z=0)\ \mathrm{[M_{\odot}]}$',)
    plt.xlabel(r'$\mathrm{Stream\ mass\ [Pal 5]}$',)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.set_ylim(1e5,5e9)
    mn, mx = ax1.get_ylim()
    ax2.set_ylim(mn, mx)
    ax2.set_yscale('log')
    ax2.minorticks_off()

    ticks = ax1.get_yticks()
    wdm = wdm_mass(np.asarray(ticks))
    labels = [r'$%.1f$' % t for t in wdm]
    ax2.set_yticklabels(labels)
    ax2.set_ylabel(r'$m_{\mathrm{WDM}\ \mathrm{[keV]}}$')
    mn2, mx2 = ax2.get_ylim()
    ax2.fill_between([0,1.5],[halo_mass(2.95),halo_mass(2.95)],[3e8,3e8], facecolor= 'none', edgecolor='k', alpha = 0.3, hatch ='/', lw=2, zorder=4) # MW satellite constraint
    #plt.plot([29.5, 33.5], [halo_mass(2.95), halo_mass(2.95)], c='0.5', lw=2, linestyle='-')#, label=r'$\mathrm{MW\ satellites}$')
    plt.text(.5,6e8,r'$\mathrm{MW\ satellites}$', horizontalalignment='center', verticalalignment='center', size=10,bbox=dict(facecolor='white', alpha=1, edgecolor='none'),zorder=-1)


    plt.plot([0,1.5], [halo_mass(5.30), halo_mass(5.30)], c='0.5', lw=2, linestyle='-',zorder=1)#, label=r'$\mathrm{Lyman}\ \alpha$')
    
    plt.text(.5,halo_mass(5.30),r'$\mathrm{Lyman}\ \alpha$', horizontalalignment='center', verticalalignment='center', size=10,bbox=dict(facecolor='white', alpha=1, ec='none'),zorder=2)
    #plt.plot([31.9,31.9], [mn2,mx2], c='0.5', lw=2, linestyle='--')#, label=r'$\mathrm{MW\ satellites}$')
    #plt.plot([33.0,33.0], [mn2,mx2], c='0.5', lw=2, linestyle='--')#, label=r'$\mathrm{Lyman}\ \alpha$')
    #plt.text(31,3e4,r'$\mathrm{GD-1}$',rotation=90., horizontalalignment='center', verticalalignment='bottom', size=10,bbox=dict(facecolor='white', alpha=0, ec='none'),zorder=-10)
    #plt.text(32,3e4,r'$\mathrm{Indus}$',rotation=90., horizontalalignment='center', verticalalignment='bottom', size=10,bbox=dict(facecolor='white', alpha=1, ec='none'),zorder=-1)
    #plt.text(33,3e4,r'$\mathrm{ATLAS}$',rotation=90., horizontalalignment='center', verticalalignment='bottom', size=10,bbox=dict(facecolor='white', alpha=1, ec='none'),zorder=-1)


    plt.xlim(0.2,1.3)
    plt.title(r'$\mathrm{Minimum\ Detectable\ Halo\ Mass}$')
    plt.tight_layout()
    if filename is not None:
        plt.savefig('%s.png' % filename)
    plt.show()

    return ret

