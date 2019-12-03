import streampepperdf
import os, os.path
import numpy as np
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
#import read_girardi
import astropy.io.fits as fits
from galpy.util import save_pickles

#0.1 degre bins ~ 700 stars/sq deg
#pal 5 has 68k stars in the CFHT field, ~14 deg, ~3160 stars/deg

remodel = False


def sp_stream_samples(sp, nsample =10000, lb=True, massexp=-2, GMmod=1., massrange = [6,9], cutoff =5., ratemod = 1., do_sample=False):
    massexp=massexp
    sample_GM= lambda: GMmod*powerlaw_wcutoff(massrange,cutoff)
    rate_range= numpy.arange(massrange[0]+0.5,massrange[1]+0.5,1)
    rate = ratemod*numpy.sum([dNencdm(sp,10.**r,Xrs=5.,plummer=False,rsfac=1.,sigma=120.) for r in rate_range])
    sample_rs= lambda x: rs(x*bovy_conversion.mass_in_1010msol(V0,R0)*10.**10.,plummer=False,rsfac=1.)
    ns= 0
    print rate
    sp.simulate(rate=rate,sample_GM=sample_GM,sample_rs=sample_rs,Xrs=3.,sigma=120./220.)

    if do_sample==True:
        sp_sample= sp.sample(n=nsample,lb=lb)
        spc = SkyCoord(sp_sample[0]*u.deg,sp_sample[1]*u.deg,distance=sp_sample[2]*u.kpc,frame='galactic')
        spxi = radec_to_pal5xieta(spc.icrs.ra.value, spc.icrs.dec.value, degree=True)
        return sp_sample, spxi, spc

class pal5_stream_instance:

	def __init__(self, remodel=False, save=False, leading=False):
		if remodel==True:
			from streampepper_utils import parse_times
			self.sp = pal5_util.setup_pal5model(timpact=parse_times('64sampling',5),leading=leading)
			if save:
				self.pepperfilename= './data/pal5_64sample.pkl'
				save_pickles(self.pepperfilename,self.sp)

		if remodel==False:
			self.pepperfilename= 'data/pal5_64sampling_trailing.pkl'
			with open(self.pepperfilename,'rb') as savefile:
				self.sp= pickle.load(savefile)
		self.sp._useInterp=True

	def add_smooth(self, nsample=10000):
		self.smooth = pal5_util.setup_pal5model(n=nsample, timpact=None, leading=False)

	def densApproxSample(self, apar = np.linspace(0,4,40)):
		dens = numpy.array([self.sp._density_par_approx(a, self.sp._tdisrupt,False,False) for a in apar]).T
		return dens

	def sample(self, nsample = 10000, lrange = [340,360], brange = [37.5,47.5],ratemod=1.):
		self.sp_sample, self.spxi, self.spc = sp_stream_samples(self.sp, nsample = nsample,ratemod=ratemod,do_sample=True)
		sel = ((self.spc.l.value<lrange[1])&(self.spc.l.value>lrange[0]) & (self.spc.b.value<brange[1])&(self.spc.b.value>brange[0]))
		self.spdata = np.array([self.spc.l[sel].value-360., self.spc.b[sel].value-np.median(self.spc.b[sel].value)]).T
		self.spdata_rs = self.spc.distance.value[sel]
		#self.spdata = self.spxi[(self.spxi[:,0]>area[0])&(self.spxi[:,0]<area[1])&(self.spxi[:,1]>area[2])&(self.spxi[:,1]<area[3])]
		#self.spdata_rs = self.sp_sample[2,:][(self.spxi[:,0]>area[0])&(self.spxi[:,0]<area[1])&(self.spxi[:,1]>area[2])&(self.spxi[:,1]<area[3])]
		#self.ms, self.gstars, self.rstars, self.istars = assign_mass_and_color(self.spdata,self.spdata_rs, min_g_mag = 28., max_g_mag = 14.6, slope = -0.5)

	def assign_masses(self, n=None, slope = -0.5, maxmass=0.82):
		maxmass = np.max(maxmass)
		minmass = np.min(.15)
		if n==None:
			nstars = len(self.spdata)
		else: nstars=n
		self.masses = np.zeros(nstars)

		for i in np.arange(nstars):
			x1=maxmass #np.minimum(maxmass, maxmass_from_g[i])
			x0=minmass #np.maximum(minmass, minmass_from_g[i])
			y=np.random.uniform(size=1)
			self.masses[i] = ((x1**(slope+1) - x0**(slope+1))*y + x0**(slope+1))**(1/(slope+1))

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


