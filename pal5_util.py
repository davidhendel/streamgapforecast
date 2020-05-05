import numpy
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.orbit import Orbit
from galpy.df import streamdf, streamgapdf
from streampepperdf import streampepperdf
from galpy.util import bovy_conversion, bovy_coords
R0,V0= 8., 220.
# Coordinate transformation routines
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

def radec_to_pal5phi1phi2(ra,dec,degree=True):
    #from Erkal, Koposov, & Belokurov 2017 MNRAS 470, 60
    XYZ= numpy.array([numpy.cos(dec*np.pi/180.)*numpy.cos(ra*np.pi/180.),
                      numpy.cos(dec*np.pi/180.)*numpy.sin(ra*np.pi/180.),
                      numpy.sin(dec*np.pi/180.)])
    rot = numpy.array([[-0.656057, -0.754711, 0.000636],
                       [ 0.609115, -0.528995, 0.590883],
                       [-0.445608,  0.388045, 0.806751]])
    phiXYZ= numpy.dot(rot,XYZ)
    phi2= numpy.arcsin(phiXYZ[2])
    phi1= numpy.arctan2(phiXYZ[1],phiXYZ[0])
    return numpy.array([phi1*180./np.pi,phi2*180./np.pi])

def setup_pal5model(leading=False,
                    timpact=None,
                    hernquist=True,
                    age=5.,
                    singleImpact=False,
                    length_factor=1.,
                    **kwargs):
    obs= Orbit([229.018,-0.124,23.2,-2.296,-2.257,-58.7],
               radec=True,ro=R0,vo=V0,
               solarmotion=[-11.1,24.,7.25])
    aAI= actionAngleIsochroneApprox(pot=MWPotential2014,b=0.81)
    sigv= 0.5*(5./age) #km/s, adjust for diff. age
    if timpact is None:
        sdf= streamdf(sigv/V0,progenitor=obs,
                      pot=MWPotential2014,aA=aAI,
                      leading=leading,nTrackChunks=11,
                      tdisrupt=age/bovy_conversion.time_in_Gyr(V0,R0),
                      ro=R0,vo=V0,R0=R0,
                      vsun=[-11.1,V0+24.,7.25],
                      custom_transform=_TPAL5)
    elif singleImpact:
        sdf= streamgapdf(sigv/V0,progenitor=obs,
                         pot=MWPotential2014,aA=aAI,
                         leading=leading,nTrackChunks=11,
                         tdisrupt=age/bovy_conversion.time_in_Gyr(V0,R0),
                         ro=R0,vo=V0,R0=R0,
                         vsun=[-11.1,V0+24.,7.25],
                         custom_transform=_TPAL5,
                         timpact= 0.3/bovy_conversion.time_in_Gyr(V0,R0),
                         spline_order=3,
                         hernquist=hernquist,
                         impact_angle=0.7,
                         impactb=0.,
                         GM= 10.**-2./bovy_conversion.mass_in_1010msol(V0,R0),
                         rs= 0.625/R0,
                         subhalovel=numpy.array([6.82200571,132.7700529,14.4174464])/V0,
                         **kwargs)
    else:
        sdf= streampepperdf(sigv/V0,progenitor=obs,
                            pot=MWPotential2014,aA=aAI,
                            leading=leading,nTrackChunks=101,
                            tdisrupt=age/bovy_conversion.time_in_Gyr(V0,R0),
                            ro=R0,vo=V0,R0=R0,
                            vsun=[-11.1,V0+24.,7.25],
                            custom_transform=_TPAL5,
                            timpact=timpact,
                            spline_order=1,
                            hernquist=hernquist,
                            length_factor=length_factor)
    sdf.turn_physical_off()
    return sdf