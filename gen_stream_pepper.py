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
#%pylab inline
R0,V0= 8., 220.
save_figures= True

prog= Orbit([229.018,-0.124,23.2,-2.296,-2.257,-58.7],radec=True,ro=R0,vo=V0,
            solarmotion=[-11.1,24.,7.25])
aAI= actionAngleIsochroneApprox(pot=MWPotential2014,b=0.8)
sigv= 0.5

######################################################################
######################################################################
######################################################################

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
        print "No error model for this survey"


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
    # if survey == 'LSST'
    #    maggrid = np.linspace(15, 25.3, 1000)
    # if survey == 'LSST10'
    #    maggrid = np.linspace(15, 26.6, 1000)
    # if survey == 'SDSS'
    #    maggrid = np.linspace(15, 22.3, 1000)
    # if survey == 'CFHT'
    maggrid = np.linspace(15, 28, 1000)
    res = [getMagErr(m, filt, survey) for m in maggrid]
    res = scipy.interpolate.UnivariateSpline(maggrid, res, s=0)(mag)
    return res


def getMagLimit(filt, survey='LSST', maxerr=0.1):
    "A sophisticated calculation of LSST magntude limit"
    xgrid = np.linspace(15, 28, 1000)
    err = getMagErrVec(xgrid, filt, survey)
    xid = np.argmax(err * (err < maxerr))
    return xgrid[xid]


#apar= numpy.arange(options.amin,options.amax,options.dapar)

#sdf_smooth= pal5_util.setup_pal5model(age=9.)
#sdf_pepper= pal5_util.setup_pal5model(singleImpact=True)
#
#

if 0:
  sdf_pepper_t= pal5_util.setup_pal5model(timpact=[1.])
  sdf_pepper_l= pal5_util.setup_pal5model(timpact=[1.],leading=True)
  massrange=[7.,9.]
  massexp=-2.
  sp = sdf_pepper_l
  sample_GM= lambda: (10.**((massexp+1.5)*massrange[0])\
                          +(10.**((massexp+1.5)*massrange[1])\
                                -10.**((massexp+1.5)*massrange[0]))\
                          *numpy.random.uniform())**(1./(massexp+1.5))\
                          /bovy_conversion.mass_in_msol(V0,R0)
  rate_range= numpy.arange(massrange[0]+0.5,massrange[1]+0.5,1)
  rate= numpy.sum([dNencdm(sp,10.**r,Xrs=3.,
                          plummer=False,rsfac=1.,
                          sigma=120.)
                  for r in rate_range])


for i in np.arange(10):
  sample_GM= lambda: powerlaw_wcutoff(massrange,7.)
  rate_range= numpy.arange(massrange[0]+0.5,massrange[1]+0.5,1)
  rate = 3*numpy.sum([dNencdm(sp,10.**r,Xrs=3.,plummer=False,rsfac=1.,sigma=120.) for r in rate_range])
  sample_rs= lambda x: rs(x*bovy_conversion.mass_in_1010msol(V0,R0)*10.**10.,plummer=False,rsfac=1.)
  ns= 0
  sp.simulate(rate=rate,sample_GM=sample_GM,sample_rs=sample_rs,Xrs=3.,sigma=120./220.)
  sp._useInterp=True
  sp_sample= sp.sample(n=10000,lb=True)
  spc = SkyCoord(sp_sample[0]*u.deg,sp_sample[1]*u.deg,frame='galactic')
  spxi = radec_to_pal5xieta(spc.icrs.ra, spc.icrs.dec)
  #plt.scatter(spxi[:,0], spxi[:,1]-.05, c='k', alpha=0.05, s=1)
  #plt.hist(spxi[:,1], bins=np.linspace(-1.4,0,200), histtype='step')
  a,b = np.histogram(spxi[:,1], bins=np.linspace(-1.4,0,100))
  f = np.polyfit(b[:-1],a,3)
  d = a - (f[0]*np.linspace(-1.4,0,100)**3 + f[1]*np.linspace(-1.4,0,100)**2 + f[2]*np.linspace(-1.4,0,100) + f[3])[:-1]
  plt.scatter(np.linspace(-1.4,0,100)[:-1], d)



a,b,c=  plt.hist(spxi[:,1], bins=np.linspace(-1.4,0,100), histtype='step')
f = np.polyfit(b[:-1],a,3)
d = a - (f[0]*np.linspace(-1.4,0,100)**3 + f[1]*np.linspace(-1.4,0,100)**2 + f[2]*np.linspace(-1.4,0,100) + f[3])[:-1]

plt.hist(spxi[:,1], bins=np.linspace(-1.4,0,100), histtype='step')


sdf_pepper_l.plotTrack(d1='ll',d2='dist',interp=True,color='k',spread=2,overplot=False,lw=2.)
sdf_pepper_l.plotTrack(d1='ll',d2='dist',interp=False,color='k',spread=0,overplot=True,ls='none',marker='o')
sdf_pepper_l.plotProgenitor(d1='ll',d2='dist',color='r',overplot=True,ls='--')

#sdf_smooth_sample= sdf_smooth.sample(n=1000,lb=True)
#
#apar = np.linspace(0,5,100)
#densOmega= numpy.array([sdf_pepper._densityAndOmega_par_approx(a) for a in apar]).T
#plt.plot(apar,densOmega[0])
#
#
#plt.scatter(sdf_smooth_sample[0],5+sdf_smooth_sample[1],c='k',s=.1, alpha = 0.2)
#plt.scatter(sdf_pepper_sample[0],sdf_pepper_sample[1]-5.,c='k',s=.1, alpha = 0.2)


######################################################################
######################################################################
######################################################################
#generate smooth + single impact stream

from astropy.coordinates import SkyCoord

sdf_smooth_l = pal5_util.setup_pal5model(age=5., leading=True)
sdf_smooth_t = pal5_util.setup_pal5model(age=5., leading=False)
sdf_smooth_l_sample= sdf_smooth_l.sample(n=24000,lb=True)
sdf_smooth_t_sample= sdf_smooth_t.sample(n=24000,lb=True)

reload(pal5_util)
sdf_impact_l = pal5_util.setup_pal5model(age=5., leading=True, singleImpact=True, timpact=.3)
sdf_impact_t = pal5_util.setup_pal5model(age=5., leading=False)
sdf_impact_l_sample= sdf_impact_l.sample(n=24000,lb=True)
sdf_impact_t_sample= sdf_impact_t.sample(n=24000,lb=True)

lsc = SkyCoord(sdf_smooth_l_sample[0]*u.deg,sdf_smooth_l_sample[1]*u.deg,frame='galactic')
tsc = SkyCoord(sdf_smooth_t_sample[0]*u.deg,sdf_smooth_t_sample[1]*u.deg,frame='galactic')
lic = SkyCoord(sdf_impact_l_sample[0]*u.deg,sdf_impact_l_sample[1]*u.deg,frame='galactic')
tic = SkyCoord(sdf_impact_t_sample[0]*u.deg,sdf_impact_t_sample[1]*u.deg,frame='galactic')

lsxi = radec_to_pal5xieta(lsc.icrs.ra, lsc.icrs.dec)
tsxi = radec_to_pal5xieta(tsc.icrs.ra, tsc.icrs.dec)
lixi = radec_to_pal5xieta(lic.icrs.ra, lic.icrs.dec)
tixi = radec_to_pal5xieta(tic.icrs.ra, tic.icrs.dec)

plt.scatter(lsxi[:,0],lsxi[:,1],c='k',s=.1, alpha = 0.2)
plt.scatter(tsxi[:,0],tsxi[:,1],c='r',s=.1, alpha = 0.2)
plt.scatter(lixi[:,0],lixi[:,1]+.1,c='k',s=.1, alpha = 0.2)
plt.scatter(tixi[:,0],tixi[:,1]+.1,c='r',s=.1, alpha = 0.2)


smoothdata = lsxi[(lsxi[:,0]>-.20)&(lsxi[:,1]>-.20)]
impactdata = lixi[(lixi[:,0]>-.20)&(lixi[:,1]>-.20)]
smooth_rs = sdf_smooth_l_sample[2,:][(lsxi[:,0]>-.20)&(lsxi[:,1]>-.20)]
impact_rs = sdf_impact_l_sample[2,:][(lixi[:,0]>-.20)&(lixi[:,1]>-.20)]
#plt.scatter(smoothdata[:,0],smoothdata[:,1],c='k',s=.1, alpha = 0.2)
#plt.scatter(impactdata[:,0],impactdata[:,1],c='k',s=.1, alpha = 0.2)

#sample masses
import read_girardi
isoname = "iso_a12.0_z0.00020.dat"
iso = read_girardi.read_girardi(isoname)
xind = iso['stage'] <= 3  # 3 = cut the horizontal branch, 6 = cut AGB
maxmass = np.max(iso['M_ini'][xind])
minmass = np.min(iso['M_ini'][xind])

n=-.5
x1=maxmass
x0=minmass
nstars = len(impactdata)
y=np.random.uniform(size=nstars)
ms = ((x1**(n+1) - x0**(n+1))*y + x0**(n+1))**(1/(n+1))

ogstars = np.interp(ms,iso['M_ini'],iso['DES-g'])+(5*np.log10(impact_rs*1e3)-5)
orstars = np.interp(ms,iso['M_ini'],iso['DES-r'])+(5*np.log10(impact_rs*1e3)-5)
oistars = np.interp(ms,iso['M_ini'],iso['DES-i'])+(5*np.log10(impact_rs*1e3)-5)

gstars = ogstars[(ogstars>14.6)&(ogstars<28.)&(orstars<28.)&(oistars<28.)]
rstars = orstars[(ogstars>14.6)&(ogstars<28.)&(orstars<28.)&(oistars<28.)]
istars = oistars[(ogstars>14.6)&(ogstars<28.)&(orstars<28.)&(oistars<28.)]
impactdata_cut = impactdata[(ogstars>14.6)&(ogstars<28.)&(orstars<28.)&(oistars<28.)]
#smoothdata_cut = smoothdata[(ogstars>14.6)&(ogstars<28.)&(orstars<28.)&(oistars<28.)]


bg_name = '/Users/hendel/modules/Galaxia/GalaxiaData/Examples/test_gap_mock_big.ebf'
import ebf
F = ebf.read(bg_name)
dm = 5*np.log10(F['rad']*1e3)-5  
bg_u,bg_g,bg_r,bg_i,bg_z = [F['sdss_%s'%_]+dm for _ in ['u','g','r','i','z']]
bg_b, bg_l = [F['glat'], F['glon']]
bg_og = bg_g
bg_u = bg_u[bg_og>14.6]
bg_g = bg_g[bg_og>14.6]
bg_r = bg_r[bg_og>14.6]
bg_i = bg_i[bg_og>14.6]
bg_z = bg_z[bg_og>14.6]
bg_l = bg_l[bg_og>14.6]
bg_b = bg_b[bg_og>14.6]


bgc = SkyCoord(bg_l*u.deg,bg_b*u.deg,frame='galactic')
bgxi = radec_to_pal5xieta(bgc.icrs.ra, bgc.icrs.dec)


plt.subplot(131,aspect='equal')
plt.title('background')
plt.hist2d(bgxi[:,0], bgxi[:,1], 
  bins=np.linspace(-.17,-.04,100), vmin=0,vmax=300)

plt.subplot(132,aspect='equal')
plt.title('background+streamgap')
plt.hist2d(np.hstack((bgxi[:,0],impactdata[:,0])), np.hstack((bgxi[:,1],impactdata[:,1])), 
  bins=np.linspace(-.17,-.04,100), vmin=0,vmax=300)


gcurve, rcurve = getIsoCurve(iso)
mean_dm = np.mean(5*np.log10(impact_rs*1e3)-5)
gcurve, rcurve = [_ + mean_dm for _ in [gcurve, rcurve]]
col_func_g = interp1d(gcurve,gcurve-rcurve)
thresh = 2.
dm = .35

garr = np.linspace(15,28,100)
gd = np.hstack((bg_g,gstars))
g =gd[gd>14.6]
col = (np.hstack((bg_g,gstars))- np.hstack((bg_r,rstars)))
col =col[gd>14.6]
sel = (col < col_func_g(g)+.05)&(col > col_func_g(g)-.05)

plt.subplot(133,aspect='equal')
plt.hist2d(np.hstack((bgxi[:,0],impactdata_cut[:,0]))[sel], np.hstack((bgxi[:,1],impactdata_cut[:,1]))[sel], bins=np.linspace(-.17,-.04,100), vmin=0,vmax=60)
plt.title('with CMD cut')


smoothfit = np.polyfit(smoothdata[:,0],smoothdata[:,1],1)
impactfit = np.polyfit(impactdata[:,0],impactdata[:,1],1)

def rot(arr, theta):
  theta = theta*np.pi/180.
  c, s = np.cos(theta), np.sin(theta)
  R = np.array(((c,-s), (s, c)))
  return np.dot(R,arr.T)

smoothrot = rot(smoothdata,90.+np.rad2deg(np.arctan(impactfit[0])))
impactrot = rot(impactdata,90.+np.rad2deg(np.arctan(impactfit[0])))

bgrot = rot((np.vstack((bg_l,bg_b)).T),np.rad2deg(np.arctan(impactfit[0])))


from scipy import signal


plt.subplot(121)
ns,binss,patchess = plt.hist(smoothrot[0]*180./np.pi,histtype='step',bins=100, label = 'smooth')
ni,binsi,patchesi = plt.hist(impactrot[0]*180./np.pi,histtype='step',bins=100, label = 'impact')
plt.legend()
plt.ylabel('counts')
plt.xlabel('angle along stream [deg]')

fs,pxys=signal.csd(ns,ns,scaling='spectrum')
fi,pxyi=signal.csd(ni,ni,scaling='spectrum')
f, pxy = signal.csd(ni/ns,ni/ns,scaling='density',fs=6.666)

plt.subplot(122)
#plt.loglog(1/fs,np.sqrt(pxys), label = 'smooth')
#plt.loglog(1/fi,np.sqrt(pxyi), label = 'impact')
plt.loglog(1./f,np.sqrt(pxy))
plt.xlabel('1/freq')
















