# run_pal5_abc.py: simple ABC method for constraining Nsubhalo from Pal 5 data
from __future__ import print_function
import os, os.path
import glob
import csv
import time
import pickle
from optparse import OptionParser
import numpy as np
from numpy.polynomial import Polynomial
from scipy import interpolate, signal
from galpy.util import save_pickles, bovy_conversion, bovy_coords
import simulate_streampepper
import bispectrum
import pal5_util
from gd1_util import R0,V0
if os.uname()[1] == 'hendel':
    _DATADIR = '/Users/hendel/projects/streamgaps/streampepper/data/'
elif os.uname()[1] == 'yngve':
    _DATADIR = '/epsen_data/scr/hendel/streamgapforecast/'
#_DATADIR= os.getenv('DATADIR')
_BISPECIND= 2


#new
#python3 run_pal5_abc_sample.py -s pal5_64sampling_trailing.pkl --outsamp abcsamples/samp_test.dat  --abcfile abcsamples/abc_test.dat  -M 7,9 --nsamples=3200  --nbg 13 --nsims=1000 --nerrsim 100

# python run_pal5_abc_sample.py -s pal5_64sampling.pkl --outsamp abcsamples/outsamp.dat --abcfile abcsamples/abcsamp.dat -M 6,9 -m dens_30000 --nsamples=30000 --nbg 300

def get_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    # stream
    parser.add_option("-s",dest='streamsavefilename',
                      default=None,
                      help="Filename to save the streampepperdf object in")
    # savefilenames
    parser.add_option("--datadir",dest='datadir',default=_DATADIR,
                      help="Name of the data directory, prefixed to outsamp and abcfile")
    parser.add_option("--outsamp",dest='outsamp',default=None,
                      help="Name of the output file for the sampled density")
    parser.add_option("-o","--abcfile",dest='abcfile',default=None,
                      help="Name of the output file for the ABC")
    parser.add_option("-b","--batch",dest='batch',default=None,
                      type='int',
                      help="If running batches of ABC simulations, batch number")
    # Parameters of the subhalos simulation
    parser.add_option("-t","--timpacts",dest='timpacts',default='64sampling',
                      help="Impact times in Gyr to consider; should be a comma separated list")
    parser.add_option("-X",dest='Xrs',default=5.,
                      type='float',
                      help="Number of times rs to consider for the impact parameter")
    parser.add_option("-l",dest='length_factor',default=1.,
                      type='float',
                      help="length_factor input to streampepperdf (consider impacts to length_factor x length)")
    parser.add_option("-M",dest='mass',default='5,9',
                      help="Mass or mass range to consider; given as log10(mass)")
    parser.add_option("--rsfac",dest='rsfac',default=1.,type='float',
                      help="Use a r_s(M) relation that is a factor of rsfac different from the fiducial one")
    parser.add_option("--plummer",action="store_true", 
                      dest="plummer",default=False,
                      help="If set, use a Plummer DM profile rather than Hernquist")
    parser.add_option("--age",dest='age',default=5.,type='float',
                      help="Age of the stream in Gyr")
    # Parallel angles at which to compute stuff
    parser.add_option("--ximin",dest='ximin',default=0.,
                      type='float',
                      help="Minimum parallel angle to consider")
    parser.add_option("--ximax",dest='ximax',default=15.,
                      type='float',
                      help="Maximum parallel angle to consider (default: 2*meandO*mintimpact)")
    parser.add_option("--nxi",dest='nxi',default=151,
                      type='int',
                      help="Steps in xi to use")
    # Data handling and continuum normalization
    parser.add_option("--polydeg",dest='polydeg',default=1,
                      type='int',
                      help="Polynomial order to fit to smooth stream density")
    parser.add_option("--minxi",dest='minxi',default=0.,
                      type='float',
                      help="Minimum xi to consider")   
    parser.add_option("--maxxi",dest='maxxi',default=15.,
                      type='float',
                      help="Maximum xi to consider")   
    parser.add_option("--nerrsim",dest='nerrsim',default=10,
                      type='int',
                      help="Simulate this many realizations of the errors per rate simulation")   
    parser.add_option("-m",dest='mockfilename',
                      default=None,
                      help="If set, filename of a mock Pal 5 simulation to use instead of real data")
    # Parameters of the ABC simulation
    parser.add_option("--ratemin",dest='ratemin',default=-1.5,
                      type='float',
                      help="Minimum rate compared to CDM expectation; in log10")
    parser.add_option("--ratemax",dest='ratemax',default=1.,
                      type='float',
                      help="Maximum rate compared to CDM expectation; in log10")
    parser.add_option("-n","--nsims",dest='nsims',default=100,
                      type='int',
                      help="Number of simulations to run")
    parser.add_option("-r","--recompute",action="store_true", 
                      dest="recompute",default=False,
                      help="If set, do not run simulations, but recompute the statistics for existing densities")
    parser.add_option("--recomputeall",action="store_true", 
                      dest="recomputeall",default=False,
                      help="If set, do not run simulations, but recompute the statistics for existing densities for *all* existing batches")
    parser.add_option("--nsamples",dest='nsamples',default=1000,
                      type='int',
                      help="Number of sample stars to draw")
    parser.add_option("--nbg",dest='nbg',default=10,
                      type='int',
                      help="Number of background stars per bin")
    parser.add_option("--summarize",dest='summarize',default=None,
                      type='int',
                      help="Compute summary statistics from the abcfile")
    parser.add_option("--fixcdmrate",dest='fixcdmrate',default=None,
                      type='int',
                      help="Force impact rate to match CDM")
    return parser

def load_abc(filename):
    """
    NAME:
       load_abc
    PURPOSE:
       Load all ABC runs for a given filename (all batches)
    INPUT:
       filename - filename w/o batch
    OUTPUT:
       array with ABC outputs
    HISTORY:
       2016-04-10 - Written - Bovy (UofT)
    """
    allfilenames= glob.glob(filename.replace('.dat','.*.dat'))
    out= np.loadtxt(filename,delimiter=',')
    for fname in allfilenames:
        out= np.vstack((out,np.loadtxt(fname,delimiter=',')))
    return out

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
    outll= np.arange(minxi,maxxi,0.1)
    # Interpolate density
    ipll= interpolate.InterpolatedUnivariateSpline(mxieta[:,0],apars)
    ipdens= interpolate.InterpolatedUnivariateSpline(apars,dens/dens_smooth)
    return (outll,ipdens(ipll(outll)))

def setup_densOmegaWriter(apar,options):
    outdens= options.outdens
    outomega= options.outomega
    if not options.batch is None:
        outdens= outdens.replace('.dat','.%i.dat' % options.batch)
    if not options.batch is None:
        outomega= outomega.replace('.dat','.%i.dat' % options.batch)
    if os.path.exists(outdens):
        # First read the file to check apar
        apar_file= np.genfromtxt(outdens,delimiter=',',max_rows=1)
        assert np.amax(np.fabs(apar_file-apar)) < 10.**-5., 'apar according to options does not correspond to apar already in outdens'
        apar_file= np.genfromtxt(outomega,delimiter=',',max_rows=1)
        assert np.amax(np.fabs(apar_file-apar)) < 10.**-5., 'apar according to options does not correspond to apar already in outomega'
        csvdens= open(outdens,'a')
        csvomega= open(outomega,'a')       
        denswriter= csv.writer(csvdens,delimiter=',')
        omegawriter= csv.writer(csvomega,delimiter=',')
    else:
        csvdens= open(outdens,'w')
        csvomega= open(outomega,'w')
        denswriter= csv.writer(csvdens,delimiter=',')
        omegawriter= csv.writer(csvomega,delimiter=',')
        # First write apar
        denswriter.writerow([a for a in apar])
        omegawriter.writerow([a for a in apar])
        csvdens.flush()
        csvomega.flush()
    return (denswriter,omegawriter,csvdens,csvomega)

def process_pal5_densdata(options):
    # Read and prep data
    backg= 400.
    data= np.loadtxt('data/ibata_fig7b_raw.dat',delimiter=',')
    sindx= np.argsort(data[:,0])
    data= data[sindx]
    data_lowerr= np.loadtxt('data/ibata_fig7b_rawlowerr.dat',delimiter=',')
    sindx= np.argsort(data_lowerr[:,0])
    data_lowerr= data_lowerr[sindx]
    data_uperr= np.loadtxt('data/ibata_fig7b_rawuperr.dat',delimiter=',')
    sindx= np.argsort(data_uperr[:,0])
    data_uperr= data_uperr[sindx]
    data_err= 0.5*(data_uperr-data_lowerr)
    # CUTS
    indx= (data[:,0] > options.minxi-0.05)*(data[:,0] < options.maxxi)
    data= data[indx]
    data_lowerr= data_lowerr[indx]
    data_uperr= data_uperr[indx]
    data_err= data_err[indx]
    # Compute power spectrum
    tdata= data[:,1]-backg
    pp= Polynomial.fit(data[:,0],tdata,deg=options.polydeg,w=1./data_err[:,1])
    tdata/= pp(data[:,0])
    ll= data[:,0]
    py= signal.csd(tdata,tdata,fs=1./(ll[1]-ll[0]),scaling='spectrum',
                   nperseg=len(ll))[1]
    py= py.real
    # Also compute the bispectrum
    Bspec, Bpx= bispectrum.bispectrum(np.vstack((tdata,tdata)).T,
                                      nfft=len(tdata),wind=7,nsamp=1,overlap=0)
    ppyr= np.fabs(Bspec[len(Bspec)//2+_BISPECIND,len(Bspec)//2:].real)
    ppyi= np.fabs(Bspec[len(Bspec)//2+_BISPECIND,len(Bspec)//2:].imag)
    return (np.sqrt(py*(ll[-1]-ll[0])),data_err[:,1]/pp(data[:,0]),
            ppyr,ppyi)

def process_mock_densdata(options):
    print("Using mock Pal 5 data from %s" % options.mockfilename)
    simn=2
    dat = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/fakeobs/' + options.mockfilename, delimiter=',')
    #fix seed for testing
    if 0:
      print('warning: Poisson seed fixed')
      np.random.seed(42)
    h = dat[simn] + np.random.poisson(options.nbg, size=len(dat[simn]))
    h = np.maximum(h - options.nbg, np.zeros_like(h))
    #h,e= np.histogram(xieta[:,0],range=[0.2,14.3],bins=141)
    bins = np.linspace(options.ximin,options.ximax,options.nxi)
    xdata = (bins[1:] + bins[:-1]) / 2.
    # Compute power spectrum
    tdata= h-0.
    pp= Polynomial.fit(xdata,tdata,deg=options.polydeg,w=1./np.sqrt(h+1.))
    tdata = tdata/pp(xdata)
    ll= xdata
    px, py= signal.csd(tdata,tdata,fs=1./(ll[1]-ll[0]),scaling='spectrum', nperseg=len(ll))
    px= 1./px
    py= py.real
    py= np.sqrt(py*(ll[-1]-ll[0]))

    #get ps error level
    nerrsim= 1000
    ppy_err= np.empty((nerrsim,len(px)))
    terr = np.sqrt(h+1.+options.nbg)/pp(xdata)
    for ii in range(nerrsim):
        tmock= terr*np.random.normal(size=len(ll))
        ppy_err[ii]= signal.csd(tmock,tmock,
                                fs=1./(ll[1]-ll[0]),scaling='spectrum',
                                nperseg=len(ll))[1].real
    py_err= np.sqrt(np.median(ppy_err,axis=0)*(ll[-1]-ll[0]))

    np.save('/Users/hendel/Desktop/pscrosscheck_abc.npy',(dat[simn],bins,h,tdata,terr,px,py,py_err))

    # Also compute the bispectrum
    Bspec, Bpx= bispectrum.bispectrum(np.vstack((tdata,tdata)).T,
                                      nfft=len(tdata),wind=7,nsamp=1,overlap=0)
    ppyr= np.fabs(Bspec[len(Bspec)//2+_BISPECIND,len(Bspec)//2:].real)
    ppyi= np.fabs(Bspec[len(Bspec)//2+_BISPECIND,len(Bspec)//2:].imag)
    return (py,terr,
            ppyr,ppyi,py_err)

def get_star_dx(pepperdf, n=1000, returnapar=False, returnxi=False):
    (Omega,angle,dt) = pepperdf.sample(n=n, returnaAdt=True)
    RvR = pepperdf._approxaAInv(Omega[0],Omega[1],Omega[2],angle[0],angle[1],angle[2])
    vo= pepperdf._vo
    ro= pepperdf._ro
    R0= pepperdf._R0
    Zsun= pepperdf._Zsun
    vsun= pepperdf._vsun
    XYZ= bovy_coords.galcencyl_to_XYZ(RvR[0]*ro,
                                      RvR[5],
                                      RvR[3]*ro,
                                      Xsun=R0,Zsun=Zsun).T
    slbd=bovy_coords.XYZ_to_lbd(XYZ[0],XYZ[1],XYZ[2],
                                degree=True)
    sradec=bovy_coords.lb_to_radec(slbd[:,0],slbd[:,1],degree=True)
    xieta=pal5_util.radec_to_pal5xieta(sradec[:,0],sradec[:,1],degree=True)
    l= slbd[:,0]
    b= slbd[:,1]
    r= slbd[:,2]
    if returnapar:
      closesttrackindexes=np.zeros(len(r))
      for i in np.arange(len(r)):
          closesttrackindexes[i]=pepperdf.find_closest_trackpoint(RvR[0][i],RvR[1][i],RvR[2][i],RvR[3][i],RvR[4][i],RvR[5][i],interp=True)
      starapar = pepperdf._interpolatedThetasTrack[(closesttrackindexes).astype(int)]
      return starapar
    if returnxi:return xieta[:,0]
    else: return None


def pal5_abc(sdf_pepper,options):
    """
    """
    # Setup apar grid
    bins = np.linspace(options.ximin,options.ximax,options.nxi)
    if options.recompute:
        # Load density and omega from file
        outsamp= options.outsamp
        if not options.batch is None:
            outsamp= outsamp.replace('.dat','.%i.dat' % options.batch)
        sampdata= np.genfromtxt(outsamp,delimiter=',',skip_header=1)
        nd= 0
    else:
        # Setup saving 
        if os.path.exists(options.outsamp):
            # First read the file to check apar
            print('does ' + options.outsamp + ' exist?', os.path.exists(options.outsamp))
            bins_file= np.genfromtxt(options.outsamp,delimiter=',',max_rows=1)
            print(np.amax(np.fabs(bins_file-bins)))
            assert np.amax(np.fabs(bins_file-bins)) < 10.**-5., 'bins according to options does not correspond to bins already in outsamp'
            csvsamp= open(options.outsamp,'a')          
            sampwriter= csv.writer(csvsamp,delimiter=',')
        else:
            csvsamp= open(options.outsamp,'w')
            sampwriter= csv.writer(csvsamp,delimiter=',')
            # First write bins
            sampwriter.writerow([b for b in bins])
            csvsamp.flush()
        # Setup sampling
        massrange= simulate_streampepper.parse_mass(options.mass)
        rs= simulate_streampepper.rs
        sample_GM= lambda: (10.**((-0.5)*massrange[0])\
                            +(10.**((-0.5)*massrange[1])\
                              -10.**((-0.5)*massrange[0]))\
                            *np.random.uniform())**(1./(-0.5))\
            /bovy_conversion.mass_in_msol(V0,R0)
        sample_rs= lambda x: rs(x*bovy_conversion.mass_in_1010msol(V0,R0)*10.**10.,
                                plummer=options.plummer)
        rate_range= np.arange(massrange[0]+0.5,massrange[1]+0.5,1)
        cdmrate= np.sum([simulate_streampepper.\
                            dNencdm(sdf_pepper,10.**r,Xrs=options.Xrs,
                                    plummer=options.plummer,
                                    rsfac=options.rsfac)
                            for r in rate_range])
        print("Using an overall CDM rate of %f" % cdmrate)
    # Load Pal 5 data to compare to
    if options.mockfilename is None:
        power_data, data_err, data_ppyr, data_ppyi=\
                                    process_pal5_densdata(options)
    else:
        power_data, data_err, data_ppyr, data_ppyi, data_py_err=\
                                    process_mock_densdata(options)
    # Run ABC
    while True:
        if not options.recompute:
            # Simulate a rate
            l10rate= (np.random.uniform()*(options.ratemax-options.ratemin)
                      +options.ratemin)
            #### fix to CDM for testing
            if options.fixcdmrate:
              print('warning: using only CDM rate')
              l10rate=0.
            rate= 10.**l10rate*cdmrate
            print(l10rate, rate)
            # Simulate
            sdf_pepper.simulate(rate=rate,sample_GM=sample_GM,sample_rs=sample_rs,
                                Xrs=options.Xrs)
            # Compute density along stream
            try: samp,binn = np.histogram(get_star_dx(sdf_pepper,n=options.nsamples,returnxi=True),bins=bins)
            except: continue
            write_samp= [l10rate]
            write_samp.extend(list(samp))
            sampwriter.writerow(write_samp)
            csvsamp.flush()
        else:
            if nd >= len(densdata): break
            l10rate= densdata[nd,0]
            dens = densdata[nd,1:]
            omega= omegadata[nd,1:]
            nd+= 1
        # Convert density to observed density
        xixi = (bins[1:] + bins[:-1]) / 2.
        dens = samp
        # Add errors (Rao-Blackwellize...)
        for ee in range(options.nerrsim):
            tdens = dens + np.random.poisson(options.nbg, size=len(dens))
            tdens = np.maximum(tdens - options.nbg, np.zeros_like(tdens))
            pp= Polynomial.fit(xixi,tdens,deg=options.polydeg,w=1./np.sqrt(tdens+1.))
            tdens = tdens/pp(xixi)
            # Compute power spectrum
            tcsd= signal.csd(tdens,tdens,fs=1./(xixi[1]-xixi[0]),
                             scaling='spectrum',nperseg=len(xixi))[1].real
            power= np.sqrt(tcsd*(xixi[-1]-xixi[0]))
            # Compute bispectrum
            Bspec, Bpx= bispectrum.bispectrum(np.vstack((tdens,tdens)).T,
                                              nfft=len(tdens),wind=7,
                                              nsamp=1,overlap=0)
            ppyr= np.fabs(Bspec[len(Bspec)//2+_BISPECIND,
                                   len(Bspec)//2:].real)
            ppyi= np.fabs(Bspec[len(Bspec)//2+_BISPECIND,
                                   len(Bspec)//2:].imag)
            yield (l10rate, power, ppyr, ppyi,
                   #np.fabs(power[1]-power_data[1]),
                   #np.fabs(power[2]-power_data[2]),
                   #np.fabs(power[3]-power_data[3]),
                   #np.fabs(np.log(np.mean(tdens[7:17])\
                   #                         /np.mean(tdens[107:117]))),
                   #np.fabs(ppyr-data_ppyr)[_BISPECIND],
                   #np.fabs(ppyi-data_ppyi)[_BISPECIND],
                   ee)

def abcsims(sdf_pepper,options):
    """
    NAME:
       abcsims
    PURPOSE:
       Run a bunch of ABC simulations
    INPUT:
       sdf_pepper - streampepperdf object to compute peppering
       sdf_smooth - streamdf object for smooth stream
       options - the options dictionary
    OUTPUT:
       (none; just saves the simulations to a file)
    HISTORY:
       2016-04-08 - Written - Bovy (UofT)
    """
    print("Running ABC sims ...")
    abcfile= options.abcfile
    if not options.batch is None:
        abcfile= abcfile.replace('.dat','.%i.dat' % options.batch)
    if os.path.exists(abcfile):
        # First read the file to check apar
        csvabc= open(abcfile,'a')
        abcwriter= csv.writer(csvabc,delimiter=',')
    else:
        csvabc= open(abcfile,'w')
        abcwriter= csv.writer(csvabc,delimiter=',')
    nit= 0
    for sim in pal5_abc(sdf_pepper,options):
        np.random.seed((nit+3)*42)
        abcwriter.writerow(list([nit,sim[0]]+[p for p in list(sim)[1]]))
        csvabc.flush()
        abcwriter.writerow(list([sim[0]]+[p for p in list(sim)[2]]))
        csvabc.flush()
        abcwriter.writerow(list([sim[0]]+[p for p in list(sim)[3]]))
        csvabc.flush()
        print(nit)
        nit+= 1
        if nit >= options.nerrsim*options.nsims: break
    return None

def recompute(sdf_pepper,options):
    """
    NAME:
       recompute
    PURPOSE:
       Recompute the ABC summaries for existing simulations
    INPUT:
       sdf_pepper - streampepperdf object to compute peppering
       sdf_smooth - streamdf object for smooth stream
       options - the options dictionary
    OUTPUT:
       (none; just saves the simulations to a file)
    HISTORY:
       2016-04-14 - Written - Bovy (UofT)
    """
    print("Recomputing ABC sims ...")
    abcfile= options.abcfile
    if not options.batch is None:
        abcfile= abcfile.replace('.dat','.%i.dat' % options.batch)
    if os.path.exists(abcfile):
        raise IOError("ERROR: abcfile already exists, would be overridden...")
    else:
        csvabc= open(abcfile,'w')
        abcwriter= csv.writer(csvabc,delimiter=',')
    for sim in pal5_abc(sdf_pepper,options):
        abcwriter.writerow(list(sim)[:-1])
        csvabc.flush()
    return None

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    options.outsamp = options.datadir+options.outsamp
    options.abcfile = options.datadir+options.abcfile
    print(options.abcfile,options.outsamp)
    # Setup the streampepperdf object
    print(_DATADIR+options.streamsavefilename, os.path.exists(_DATADIR+options.streamsavefilename))
    if not os.path.exists(_DATADIR+options.streamsavefilename):
        print('rebuilding pepper sampling')
        timpacts= simulate_streampepper.parse_times(\
            options.timpacts,options.age)
        sdf_smooth= pal5_util.setup_pal5model(age=options.age)
        sdf_pepper= pal5_util.setup_pal5model(timpact=timpacts,
                                              hernquist=not options.plummer,
                                              age=options.age,
                                              length_factor=options.length_factor)
        save_pickles(_DATADIR+options.streamsavefilename,sdf_pepper) #, sdf_smooth)
    else:
        with open(_DATADIR+options.streamsavefilename,'rb') as savefile:
            print('loading streampepper pickle')
            #print options.streamsavefilename
            #sdf_smooth= pickle.load(savefile)
            if os.uname()[1] == 'yngve':
                sdf_pepper = pickle.load(savefile, encoding='latin1')
            if os.uname()[1] == 'hendel':
                sdf_pepper = pickle.load(savefile)
    if options.recomputeall:
        options.recompute= True
        # recompute basic
        recompute(sdf_pepper,options)
        # Find and recompute batches
        allfilenames= glob.glob(options.outdens.replace('.dat','.*.dat'))
        batches= np.array([int(fn.split('.dat')[0].split('.')[-1]) 
                              for fn in allfilenames],
                             dtype='int')
        for batch in batches:
            options.batch= batch
            recompute(sdf_pepper,options)
    elif options.recompute:
        recompute(sdf_pepper,options)
    else:
        abcsims(sdf_pepper,options)