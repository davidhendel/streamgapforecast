# script to run simulations of stream peppering
import os, os.path, sys
import csv
import time
import pickle
import numpy
from scipy import integrate
from optparse import OptionParser
from galpy.util import bovy_conversion
from galpy.util import bovy_coords
#import gd1_util
import pal5_util
from pal5_util import R0,V0
_DATADIR= '/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/' #os.getenv('DATADIR')


#python ../simulate_streampepper.py --outdens=$DATADIR/pal5_t64sampling_X3_6-9_dens.dat --outomega=$DATADIR/pal5_t64sampling_X3_6-9_omega.dat -t 64sampling -M 6,9 -n 100 -X 3. -s pal5like --age=5. --amax=1.5 --da=0.01

#export DATADIR=/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/
#python ../simulate_streampepper_sample.py --outsamp=$DATADIR/pal5_t64sampling_X3_7-9_samp.dat -nsamples 1000 -nbg 100 -t 64sampling -M 7,9 -n 100 -X 3. -s pal5like --age=5.

def get_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    # stream
    parser.add_option("-s","--stream",dest='stream',default='gd1like',
                      help="Stream to consider")
    # savefilenames
    #parser.add_option("--outdens",dest='outdens',default=None,
    #                  help="Name of the output file for the density")
    #parser.add_option("--outomega",dest='outomega',default=None,
    #                  help="Name of the output file for the mean Omega")
    parser.add_option("--outsamp",dest='outsamp',default=None,
                      help="Name of the output file for the sampled density")
    # Parameters of this simulation
    parser.add_option("-t","--timpacts",dest='timpacts',default=None,
                      help="Impact times in Gyr to consider; should be a comma separated list")
    parser.add_option("-X",dest='Xrs',default=3.,
                      type='float',
                      help="Number of times rs to consider for the impact parameter")
    parser.add_option("-l",dest='length_factor',default=1.,
                      type='float',
                      help="length_factor input to streampepperdf (consider impacts to length_factor x length)")
    parser.add_option("-M",dest='mass',default='6.5',
                      help="Mass or mass range to consider; given as log10(mass)")
    parser.add_option("--cutoff",dest='cutoff',default=None,type='float',
                      help="Log10 mass cut-off in power-spectrum")
    parser.add_option("--massexp",dest='massexp',default=-2.,type='float',
                      help="Exponent of the mass spectrum (doesn't work with cutoff)")
    parser.add_option("--timescdm",dest='timescdm',default=1.,type='float',
                      help="Use a rate that is timescdm times the CDM prediction")
    parser.add_option("--rsfac",dest='rsfac',default=1.,type='float',
                      help="Use a r_s(M) relation that is a factor of rsfac different from the fiducial one")
    parser.add_option("--plummer",action="store_true", 
                      dest="plummer",default=False,
                      help="If set, use a Plummer DM profile rather than Hernquist")
    parser.add_option("--age",dest='age',default=9.,type='float',
                      help="Age of the stream in Gyr")
    parser.add_option("--sigma",dest='sigma',default=120.,type='float',
                      help="Velocity dispersion of the population of DM subhalos")
    # Parallel angles at which to compute stuff
    parser.add_option("--ximin",dest='axi',default=-15.,
                      type='float',
                      help="Minimum parallel angle to consider")
    parser.add_option("--ximax",dest='axi',default=0.,
                      type='float',
                      help="Maximum parallel angle to consider (default: 2*meandO*mintimpact)")
    parser.add_option("--nxi",dest='nxi',default=150,
                      type='float',
                      help="Steps in xi to use")
    parser.add_option("--dt",dest='dt',default=60.,
                      type='float',
                      help="Number of minutes to run simulations for")
    parser.add_option("-n","--nsims",dest='nsims',default=None,
                      type='int',
                      help="Number of simulations to run")
    parser.add_option("--nsamples",dest='nsamples',default=10000,
                      type='int',
                      help="Number of sample stars to draw")
    parser.add_option("--nbg",dest='nbg',default=100,
                      type='int',
                      help="Number of background stars per square degree")
    return parser

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



# Function to run the simulations
def run_simulations(sdf_pepper,sdf_smooth,options):
    # Setup apar grid
    bins= numpy.linspace(options.ximin,options.ximax,options.dxi)
    # Check whether the output files already exist and if so, get the amin, amax, da from them
    if os.path.exists(options.outsamp):
        # First read the file to check apar
        bins_file= numpy.genfromtxt(options.outsamp,delimiter=',',max_rows=1)
        print numpy.amax(numpy.fabs(bins_file-bins))
        assert numpy.amax(numpy.fabs(bins_file-bins)) < 10.**-5., 'bins according to options does not correspond to bins already in outsamp'
        csvsamp= open(options.outsamp,'a')          
        sampwriter= csv.writer(csvsamp,delimiter=',')
    else:
        csvsamp= open(options.outsamp,'w')
        sampwriter= csv.writer(csvsamp,delimiter=',')
        # First write bins
        sampwriter.writerow([b for b in bins])
        csvsamp.flush()
    # Parse mass
    massrange= parse_mass(options.mass)
    if len(massrange) == 1:
        sample_GM= lambda: 10.**(massrange[0]-10.)\
            /bovy_conversion.mass_in_1010msol(V0,R0)
        rate= options.timescdm*dNencdm(sdf_pepper,
                                       10.**massrange[0],Xrs=options.Xrs,
                                       plummer=options.plummer,
                                       rsfac=options.rsfac,
                                       sigma=options.sigma)
    elif len(massrange) == 2:
        # Sample from power-law
        if not options.cutoff is None:
            sample_GM= lambda: powerlaw_wcutoff(massrange,options.cutoff)
        elif numpy.fabs(options.massexp+1.5) < 10.**-6.:
            sample_GM= lambda: 10.**(massrange[0]\
                                         +(massrange[1]-massrange[0])\
                                         *numpy.random.uniform())\
                                         /bovy_conversion.mass_in_msol(V0,R0)
        else:
            sample_GM= lambda: (10.**((options.massexp+1.5)*massrange[0])\
                                    +(10.**((options.massexp+1.5)*massrange[1])\
                                          -10.**((options.massexp+1.5)*massrange[0]))\
                                    *numpy.random.uniform())**(1./(options.massexp+1.5))\
                                    /bovy_conversion.mass_in_msol(V0,R0)
        rate_range= numpy.arange(massrange[0]+0.5,massrange[1]+0.5,1)
        rate= options.timescdm\
            *numpy.sum([dNencdm(sdf_pepper,10.**r,Xrs=options.Xrs,
                                plummer=options.plummer,rsfac=options.rsfac,
                                sigma=options.sigma)
                        for r in rate_range])
        if not options.cutoff is None:
            rate*= integrate.quad(lambda x: x**-1.5\
                                      *numpy.exp(-10.**options.cutoff/x),
                                  10.**massrange[0],10.**massrange[1])[0]\
                                  /integrate.quad(lambda x: x**-1.5,
                                                  10.**massrange[0],
                                                  10.**massrange[1])[0]
    print "Using an overall rate of %f" % rate
    sample_rs= lambda x: rs(x*bovy_conversion.mass_in_1010msol(V0,R0)*10.**10.,
                            plummer=options.plummer,rsfac=options.rsfac)
    # Simulate
    start= time.time()
    ns= 0
    while True:
        if options.nsims is None and time.time() >= (start+options.dt*60.):
            break
        elif not options.nsims is None and ns > options.nsims:
            break
        ns+= 1
        sdf_pepper.simulate(rate=rate,sample_GM=sample_GM,sample_rs=sample_rs,
                            Xrs=options.Xrs,sigma=options.sigma/V0)
        # Compute density and meanOmega and save
        try:
            densOmega= numpy.array([sdf_pepper._densityAndOmega_par_approx(a)
                                    for a in apar]).T
        except IndexError: # no hit
            dens_unp= [sdf_smooth._density_par(a) for a in apar]
            omega_unp= [sdf_smooth.meanOmega(a,oned=True) for a in apar]
            denswriter.writerow(dens_unp)
            omegawriter.writerow(omega_unp)
        else:
            denswriter.writerow(list(densOmega[0]))
            omegawriter.writerow(list(densOmega[1]))
        csvdens.flush()
        csvomega.flush()
    csvdens.close()
    csvomega.close()
    return None
        
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    # Setup 
    if options.stream.lower() == 'gd1like':
        timpacts= parse_times(options.timpacts,options.age)
        sdf_pepper= gd1_util.setup_gd1model(timpact=timpacts,
                                            hernquist=not options.plummer,
                                            age=options.age,
                                            length_factor=options.length_factor)
    elif options.stream.lower() == 'pal5like':
        timpacts= parse_times(options.timpacts,options.age)
        if options.timpacts == '64sampling':
            # We've cached this one
            #with open('/Users/hendel/projects/streamgaps/streampepper/data/pal5_64sampling.pkl','rb') as savefile:
            #   sdf_smooth= pickle.load(savefile)
            with open('/Users/hendel/projects/streamgaps/streampepper/data/pal5_64sampling.pkl','rb') as savefile:
                sdf_pepper= pickle.load(savefile)
        else:
            sdf_pepper= pal5_util.setup_pal5model(timpact=timpacts,
                                                  hernquist=not options.plummer,
                                                  age=options.age,
                                                  length_factor=options.length_factor)
    # Need smooth?
    #if options.amax is None or options.amin is None:
    #    if options.stream.lower() == 'gd1like':
    #        sdf_smooth= gd1_util.setup_gd1model(age=options.age)
    #    else:
    #        sdf_smooth= pal5_util.setup_pal5model(age=options.age)
    #    sdf_smooth.turn_physical_off()
    #else:
    #    sdf_smooth= None
    #if options.amax is None:
    #    options.amax= sdf_smooth.length()+options.dapar
    #if options.amin is None:
    #    options.amin= 2.*sdf_smooth.meanOmega(0.1,oned=True)\
    #        *numpy.amin(numpy.array(timpacts))
    run_simulations(sdf_pepper,options)#,sdf_smooth,options)