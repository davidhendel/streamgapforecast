#compare error models
import iso_handling
import pal5_mock_ps
model = pal5_mock_ps.pal5_stream_instance()
model.add_smooth()
model.sample()
model.assign_masses()

errormodels = iso_handling.load_errormodels()
isodict = iso_handling.load_iso_interps(remake=True)
try: 
    bgdict = np.load('/Users/hendel/projects/streamgaps/streampepper/bgdict.npy').item()
except:
    bgdict = iso_handling.gen_bg_counts_interp(isodict=isodict,errormodels=errormodels, verbose=True)
    #bg_interp = iso_handling.gen_bg_counts_interp(surveys=['SDSS','CFHT','LSST','LSST10','CASTOR'], 
    # bands = ['gr','gr','gr','gr','ug'], isodict=isodict, errormodels=errormodels)


#####################################################
#####################################################
#####################################################
#Check sampling effect on power spectrum

from galpy.util import bovy_coords
from scipy import signal
from numpy.polynomial import Polynomial
import pal5_util
#pepperdf = model.smooth
pepperdf = model.sp

def sp_stream_samples(sp, nsample =10000, lb=True, massexp=-2, GMmod=1., massrange = [6,9], cutoff =5., ratemod = 1., do_sample=False):
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

def get_star_dx(pepperdf,n=1000, returnapar=False, returnxi=False):
    (Omega,angle,dt) = pepperdf.sample(n=n, returnaAdt=True)
    RvR = model.sp._approxaAInv(Omega[0],Omega[1],Omega[2],angle[0],angle[1],angle[2])
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
    closesttrackindexes=np.zeros(len(r))
    for i in np.arange(len(r)):
        closesttrackindexes[i]=pepperdf.find_closest_trackpoint(RvR[0][i],RvR[1][i],RvR[2][i],RvR[3][i],RvR[4][i],RvR[5][i],interp=True)
    starapar = pepperdf._interpolatedThetasTrack[(closesttrackindexes).astype(int)]
    if returnapar:return starapar
    if returnxi:return xieta[:,0]
    return None


def xi_csd(d, bins=np.linspace(-15,0,150), nbg=0., binned=False):
    from scipy import signal
    from numpy.polynomial import Polynomial
    np.random.seed(42)
    if binned==False:
        actcounts, bins = np.histogram(d, bins=bins)
        counts = actcounts + np.random.poisson(nbg, size=len(d))
    else:counts = d + np.random.poisson(nbg, size=len(d))
    counts = np.maximum(counts - nbg, np.zeros_like(counts))
    centroids = (bins[1:] + bins[:-1]) / 2.
    err = np.sqrt(counts+nbg)#*(counts/actcounts)
    bkg=0
    degree=1
    pp=Polynomial.fit(centroids,counts,degree,w=1./numpy.sqrt(counts+1.))
    tdata= counts/pp(centroids)
    terr = err/pp(centroids)
    px, py= signal.csd(tdata,tdata, fs=1./(centroids[1]-centroids[0]),scaling='spectrum',nperseg=len(centroids))
    py= py.real
    px= 1./px
    py= numpy.sqrt(py*(centroids[-1]-centroids[0]))

    nerrsim= 1000
    ppy_err= numpy.empty((nerrsim,len(px)))
    for ii in range(nerrsim):
        tmock= terr*numpy.random.normal(size=len(centroids))
        ppy_err[ii]= signal.csd(tmock,tmock,
                                fs=1./(centroids[1]-centroids[0]),scaling='spectrum',
                                nperseg=len(centroids))[1].real
    py_err= numpy.sqrt(numpy.median(ppy_err,axis=0)*(centroids[-1]-centroids[0]))
    np.save('/Users/hendel/Desktop/pscrosscheck_xi_csd.npy',(d,bins,counts,tdata,terr,px,py,py_err))
    return px,py,py_err




#####################################################
#####################################################
#####################################################
#Check ABC data
dat = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/fakeobs/dens_10000', delimiter=',')
px,py,py_err = xi_csd(dat[2], nbg=00, bins=np.linspace(-15,0,150), binned=True)
d = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abctest.dat',delimiter=',')
plt.loglog(px,py*5., color = 'k', label = 'simulated observations, n=1000', marker='o') #fake 'observed' data
plt.loglog(px,py_err*5., color = 'k') #fake 'observed' data
plt.loglog(px,np.nanmedian(d[:,1:][d[:,0] > .5],axis=0), color = 'r',               label = r'$\mathrm{lograte > .5}$'      ) 
plt.loglog(px,np.nanmedian(d[:,1:][d[:,0] < -.5],axis=0), color = 'g',              label = r'$\mathrm{lograte < -.5}$'     ) 
plt.loglog(px,np.nanmedian(d[:,1:][(d[:,0]> -.5)&(d[:,0]<.5)],axis=0), color = 'b', label = r'$\mathrm{-.5 < lograte < .5}$') 
plt.legend(loc='upper left')
plt.ylabel('power')
plt.xlabel('xi')


#Compare PS from dat file and mockdata in run_pal5_abc
#fdat = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/fakeobs/dens_10000', delimiter=',')
fpx,fpy,fpy_err = xi_csd(fdat[2], nbg=1000, bins=np.linspace(-15,0,150), binned=True)
d,bins,counts,tdata,terr,px,py,py_err = np.load('/Users/hendel/Desktop/pscrosscheck_xi_csd.npy',allow_pickle=True)
plt.loglog(px,py, color = 'k',lw=4 ,label = 'outside', marker='o') #fake 'observed' data
plt.loglog(px,py_err, color = 'k',lw=4) #fake 'observed' data
ad,abins,acounts,atdata,aterr,apx,apy,apy_err = np.load('/Users/hendel/Desktop/pscrosscheck_abc.npy',allow_pickle=True)
plt.loglog(apx,apy, color = 'g', label = 'in abc', marker='o',markersize=5) #fake 'observed' data
plt.loglog(apx,apy_err, color = 'g') #fake 'observed' data
plt.legend(loc='upper left')
plt.ylabel('power')
plt.xlabel('xi')

#Check that background addition is working properly
fdat = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/fakeobs/dens_10000', delimiter=',')
fpx,fpy,fpy_err = xi_csd(fdat[2], nbg=0, bins=np.linspace(-15,0,150), binned=True)
plt.loglog(fpx,fpy_err, color = 'k',lw=2) #fake 'observed' data
fpx,fpy,fpy_err = xi_csd(fdat[2], nbg=100, bins=np.linspace(-15,0,150), binned=True)
plt.loglog(fpx,fpy_err, color = 'k',lw=2) #fake 'observed' data
fpx,fpy,fpy_err = xi_csd(fdat[2], nbg=300, bins=np.linspace(-15,0,150), binned=True)
plt.loglog(fpx,fpy_err, color = 'k',lw=2) #fake 'observed' data

d = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abctest_7-9_10k_0bg.dat',delimiter=',')
plt.loglog(px,np.nanmedian(d[:,1:][d[:,0] > .5],axis=0), color = 'r',               label = r'$\mathrm{lograte > .5}$'      ) 
plt.loglog(px,np.nanmedian(d[:,1:][d[:,0] < -.5],axis=0), color = 'g',              label = r'$\mathrm{lograte < -.5}$'     ) 
plt.loglog(px,np.nanmedian(d[:,1:][(d[:,0]> -.5)&(d[:,0]<.5)],axis=0), color = 'b', label = r'$\mathrm{-.5 < lograte < .5}$') 

d = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abctest_7-9_10k_100bg.dat',delimiter=',')
plt.loglog(px,np.nanmedian(d[:,1:][d[:,0] > .5],axis=0), color = 'r',               ) 
plt.loglog(px,np.nanmedian(d[:,1:][d[:,0] < -.5],axis=0), color = 'g',              ) 
plt.loglog(px,np.nanmedian(d[:,1:][(d[:,0]> -.5)&(d[:,0]<.5)],axis=0), color = 'b', ) 

d = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abctest_7-9_10k_300bg.dat',delimiter=',')
plt.loglog(px,np.nanmedian(d[:,1:][d[:,0] > .5],axis=0), color = 'r',               ) 
plt.loglog(px,np.nanmedian(d[:,1:][d[:,0] < -.5],axis=0), color = 'g',              ) 
plt.loglog(px,np.nanmedian(d[:,1:][(d[:,0]> -.5)&(d[:,0]<.5)],axis=0), color = 'b', ) 

plt.legend(loc='upper left')
plt.ylabel('power')
plt.xlabel('xi')






























#########################
#compar apar with analytic
sdens_apar = np.zeros(100)
for j in np.arange(len(sdens_apar)):
    tapar = np.linspace(0,4,100)[j]
    sdens_apar[j] = pepperdf.density_par(tapar, approx=True)

plt.plot(np.linspace(0,4,100),sdens_apar/np.sum(sdens_apar*(4/100.)),lw=2, label='analytic')
plt.ylabel('normalized counts')
plt.xlabel('apar')

for i in [10000,30000,100000]:
    print i
    starapar = get_star_dx(pepperdf,n=i,returnapar=True)
    plt.hist(starapar,bins=np.linspace(0,4,100), histtype='step',lw=2, normed=True, label = 'n=%8i'%i)
plt.legend()


#########################
#compare noise level as f(n)
colors = ['r','g','b','orange']
pepperdf = model.sp
for i, n in enumerate([1000,3000,10000,30000]):
    print n
    xi = get_star_dx(pepperdf,n=n,returnxi=True)
    px,py,py_err = xi_csd(xi)
    plt.loglog(px,py, color = colors[i], lw=2)
    plt.loglog(px,py_err, color = colors[i], label = 'n=%8i'%n)
    #plt.hist(xi,bins=np.linspace(-15,0,150), histtype='step',lw=2, normed=True, label = 'n=%8i'%i)
plt.legend()
plt.ylabel('power')
plt.xlabel('xi')



#########################
#simulate a bunch of data as f(n)
import csv
nruns = 100
for j in np.arange(nruns):
    pal5_mock_ps.sp_stream_samples(pepperdf,massrange = [7,9],do_sample=False)
    for i, n in enumerate([10000,30000]):#,10000,30000]):
        print j, n
        csvdens = open('dens_7-9_%i'%n, 'a')
        denswriter= csv.writer(csvdens,delimiter=',')
        xi = get_star_dx(pepperdf,n=n,returnxi=True)
        counts, bins = np.histogram(xi, bins=np.linspace(-15,0,150))
        denswriter.writerow(counts)
        csvdens.flush()
        csvdens.close()

#read and plot ps
nruns = 100
colors = ['r','g','b','orange']
for i, n in enumerate([1000,3000,10000,30000]):
        print n
        dat = np.loadtxt('dens_%i'%n, delimiter=',')
        px = np.zeros((nruns, (dat.shape[1]+1)/2))
        py = np.zeros((nruns, (dat.shape[1]+1)/2))
        py_err = np.zeros((nruns, (dat.shape[1]+1)/2))
        for j in np.arange(nruns):
            px[j],py[j],py_err[j] = xi_csd(dat[j], bins=np.linspace(-15,0,150), binned=True)
        plt.loglog(px[0],np.nanmedian(py,axis=0), color = colors[i])
        plt.loglog(px[0],np.nanmedian(py_err,axis=0), color = colors[i], label = 'n=%i'%n)
        plt.fill_between(px[0], np.nanpercentile(py, (25), axis=0),  y2= np.nanpercentile(py, (75), axis=0), color=colors[i], alpha=0.2)

plt.legend()
plt.ylabel('power')
plt.xlabel('xi')


#what about with Poisson errors?
nruns = 100
colors = ['r','g','b','orange']
for i, n in enumerate([10000]):
        dat = np.loadtxt('dens_%i'%n, delimiter=',')
        px = np.zeros((nruns, (dat.shape[1]+1)/2))
        py = np.zeros((nruns, (dat.shape[1]+1)/2))
        py_err = np.zeros((nruns, (dat.shape[1]+1)/2))

        print n
        for j in np.arange(nruns):
            px[j],py[j],py_err[j] = xi_csd(dat[j], bins=np.linspace(-15,0,150), binned=True)
        plt.loglog(px[0],np.nanmedian(py,axis=0), color = colors[i])
        plt.loglog(px[0],np.nanmedian(py_err,axis=0), color = colors[i], label = r'$\mathrm{no\ background}$')
        plt.fill_between(px[0], np.nanpercentile(py, (25), axis=0),  y2= np.nanpercentile(py, (75), axis=0), color=colors[i], alpha=0.2)

        for j in np.arange(nruns):
            px[j],py[j],py_err[j] = xi_csd(dat[j]+np.random.poisson(50, size=len(dat[j])), nbg=50, bins=np.linspace(-15,0,150), binned=True)
        plt.loglog(px[0],np.nanmedian(py,axis=0), color = colors[i+1])
        plt.loglog(px[0],np.nanmedian(py_err,axis=0), color = colors[i+1], label = r'$\mathrm{SNR \sim 1}$')
        plt.fill_between(px[0], np.nanpercentile(py, (25), axis=0),  y2= np.nanpercentile(py, (75), axis=0), color=colors[i+1], alpha=0.2)

        for j in np.arange(nruns):
            px[j],py[j],py_err[j] = xi_csd(dat[j]+np.random.poisson(200, size=len(dat[j])), nbg=200, bins=np.linspace(-15,0,150), binned=True)
        plt.loglog(px[0],np.nanmedian(py,axis=0), color = colors[i+2])
        plt.loglog(px[0],np.nanmedian(py_err,axis=0), color = colors[i+2], label = r'$\mathrm{SNR \sim 0.25}$')
        plt.fill_between(px[0], np.nanpercentile(py, (25), axis=0),  y2= np.nanpercentile(py, (75), axis=0), color=colors[i+2], alpha=0.2)

        for j in np.arange(nruns):
            px[j],py[j],py_err[j] = xi_csd(dat[j]+np.random.poisson(500, size=len(dat[j])), nbg=500, bins=np.linspace(-15,0,150), binned=True)
        plt.loglog(px[0],np.nanmedian(py,axis=0), color = colors[i+3])
        plt.loglog(px[0],np.nanmedian(py_err,axis=0), color = colors[i+3], label = r'$\mathrm{SNR \sim 0.1}$')
        plt.fill_between(px[0], np.nanpercentile(py, (25), axis=0),  y2= np.nanpercentile(py, (75), axis=0), color=colors[i+3], alpha=0.2)

plt.legend(loc='upper left')
plt.ylabel('power')
plt.xlabel('xi')

#now do in terms of SNR
nruns = 100
colors = ['r','g','b','orange']
for i, n in enumerate([10000]):
        dat = np.loadtxt('dens_%i'%n, delimiter=',')
        px = np.zeros((nruns, (dat.shape[1]+1)/2))
        py = np.zeros((nruns, (dat.shape[1]+1)/2))
        py_err = np.zeros((nruns, (dat.shape[1]+1)/2))

        print n
        for j in np.arange(nruns):
            px[j],py[j],py_err[j] = xi_csd(dat[j], bins=np.linspace(-15,0,150), binned=True)
        plt.plot(px[0],np.nanmedian(py,axis=0)/np.nanmedian(py_err,axis=0), color = colors[i])
        plt.plot(px[0],np.nanmedian(py_err,axis=0), color = colors[i], label = r'$\mathrm{no\ background}$')
        plt.fill_between(px[0], np.nanpercentile(py, (25), axis=0)/np.nanmedian(py_err,axis=0),  y2= np.nanpercentile(py, (75), axis=0)/np.nanmedian(py_err,axis=0), color=colors[i], alpha=0.2)

        for j in np.arange(nruns):
            px[j],py[j],py_err[j] = xi_csd(dat[j]+np.random.poisson(50, size=len(dat[j])), nbg=50, bins=np.linspace(-15,0,150), binned=True)
        plt.plot(px[0],np.nanmedian(py,axis=0)/np.nanmedian(py_err,axis=0), color = colors[i+1])
        plt.plot(px[0],np.nanmedian(py_err,axis=0), color = colors[i+1], label = r'$\mathrm{SNR \sim 1}$')
        plt.fill_between(px[0], np.nanpercentile(py, (25), axis=0)/np.nanmedian(py_err,axis=0),  y2= np.nanpercentile(py, (75), axis=0)/np.nanmedian(py_err,axis=0), color=colors[i+1], alpha=0.2)

        for j in np.arange(nruns):
            px[j],py[j],py_err[j] = xi_csd(dat[j]+np.random.poisson(200, size=len(dat[j])), nbg=200, bins=np.linspace(-15,0,150), binned=True)
        plt.plot(px[0],np.nanmedian(py,axis=0)/np.nanmedian(py_err,axis=0), color = colors[i+2])
        plt.plot(px[0],np.nanmedian(py_err,axis=0), color = colors[i+2], label = r'$\mathrm{SNR \sim 0.25}$')
        plt.fill_between(px[0], np.nanpercentile(py, (25), axis=0)/np.nanmedian(py_err,axis=0),  y2= np.nanpercentile(py, (75), axis=0)/np.nanmedian(py_err,axis=0), color=colors[i+2], alpha=0.2)

        for j in np.arange(nruns):
            px[j],py[j],py_err[j] = xi_csd(dat[j]+np.random.poisson(500, size=len(dat[j])), nbg=500, bins=np.linspace(-15,0,150), binned=True)
        plt.plot(px[0],np.nanmedian(py,axis=0)/np.nanmedian(py_err,axis=0), color = colors[i+3])
        plt.plot(px[0],np.nanmedian(py_err,axis=0), color = colors[i+3], label = r'$\mathrm{SNR \sim 0.1}$')
        plt.fill_between(px[0], np.nanpercentile(py, (25), axis=0)/np.nanmedian(py_err,axis=0),  y2= np.nanpercentile(py, (75), axis=0)/np.nanmedian(py_err,axis=0), color=colors[i+3], alpha=0.2)

plt.legend(loc='upper left')
plt.title('n=10000')
plt.ylabel('SNR')
plt.xlabel('xi')




























#####################################################
#####################################################
#####################################################
#CMDs + sky plots + hists
model.sample()
model.assign_masses()

surveys=['SDSS', 'CFHT', 'LSST', 'LSST10','CASTOR', 'WFIRST']

for i, key in enumerate(surveys):
    plt.subplot(3,6,i+1)
    if 'LSST' in key:
        mag1 = isodict['lsst_g-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror1 = errormodels[key]['g'](mag1)
        mag2 = isodict['lsst_r-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror2 = errormodels[key]['r'](mag2)
        plt.xlabel('g-r')
        plt.ylabel('g')
        plt.xlim(0,1.25)
        #plt.errorbar(errx,erry, yerr = errormodels[key]['g'](erry), xerr= np.sqrt(errormodels[key]['g'](erry)**2 + errormodels[key]['r'](erry)**2),c='k',linestyle='')
    elif key == 'CASTOR':
        mag1 = isodict['castor_u-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror1 = errormodels[key]['u'](mag1)
        mag2 = isodict['castor_g-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror2 = errormodels[key]['g'](mag2)
        plt.xlabel('u-g')
        plt.ylabel('u')
        plt.xlim(.75,2)
        #plt.errorbar(errx,erry, yerr = errormodels[key]['u'](erry), xerr= np.sqrt(errormodels[key]['g'](erry)**2 + errormodels[key]['u'](erry)**2), c='k', linestyle='')
    elif key == 'WFIRST':
        mag1 = isodict['Z087-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror1 = errormodels[key]['z'](mag1)
        mag2 = isodict['H158-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror2 = errormodels[key]['h'](mag2)
        plt.xlabel('Z-H')
        plt.ylabel('Z')
        plt.xlim(.25,1.75)
    else:
        mag1 = isodict['sdss_g-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror1 = errormodels[key]['g'](mag1)
        mag2 = isodict['sdss_r-10.00-0.0001'](model.masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror2 = errormodels[key]['r'](mag2)
        plt.xlabel('g-r')
        plt.ylabel('g')
        plt.xlim(-.25,.75)

    omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1))]
    omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1))]
    print key, len(omag1)
    plt.scatter(omag1-omag2, omag1, c='k', alpha=0.3,s=1)
    plt.ylim(28,14)
    plt.title(key)

    plt.subplot(3,6,i+7)
    plt.scatter(model.spdata[:,0][((magerror1<.1)&(magerror2<.1))], model.spdata[:,1][((magerror1<.1)&(magerror2<.1))], c='k', alpha=0.3,s=1)
    plt.xlabel('RA [deg]')
    plt.xlim(-20,0)
    print i
    if i == 0: plt.ylabel('Dec [deg]')
    plt.subplot(3,6,i+13)
    plt.hist(model.spdata[:,0][((magerror1<.1)&(magerror2<.1))], histtype='step', lw=1, color='k',bins=np.linspace(-20,0,200))
    
    plt.xlabel('RA [deg]')
    if i == 0: plt.ylabel('counts')
    plt.text(-19,55,'%3i stream stars'%(len(omag1)))
    plt.text(-19,45,'%3i background stars'%(bgdict[key]((16.8,np.max(omag2)))))
    plt.ylim(0,70)
    plt.xlim(-20,0)

#

#######################################################
#######################################################
#######################################################
#for pal5-like streams, compute PS from new ps dict 
# in densiyt_calcs

colors = ['seagreen', 'orange', 'darkslateblue']
for i,survey in enumerate(np.sort(pepperdict.keys())):
    #plt.fill_between(px[:-1], np.nanpercentile(ps[survey], (25), axis=0)[1:],  
    #        y2= np.nanpercentile(ps[survey], (75), axis=0)[1:], color=colors[i], alpha=0.2)
    plt.loglog(px[:-1], (np.nanmedian(ps[survey], axis=0)[1:]), c=colors[i], lw=4, label=survey)
    plt.loglog(px[:-1], (np.nanmedian(ps_err[survey], axis=0)[1:]), c=colors[i], lw=2, ls='--')
plt.ylabel(r'$\sqrt{\delta \delta}$')
plt.xlim([-.1,30])
plt.xlabel(r'$\mathrm{log10\ 1/k_\xi (deg)}$')
plt.legend(loc='upper left')


#######################################################
#######################################################
#######################################################
#for pal5-like streams, compute PS and count signal
import pickle
from numpy.polynomial import Polynomial
from scipy import signal
lbrs = pickle.load(open('./data/lbr500.pkl', "rb" ))
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
surveys=['SDSS', 'LSST', 'WFIRST']
colors = ['k', 'b', 'r', 'c']
outdict = {}
errdict = {}

for i, key in enumerate(surveys):
    outdict[key]=[]
    errdict[key]=[]
    for j in np.arange(50):
        lbr = lbrs['lbr'+str(j)][:,::2]
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

        print '%3i stream stars'%(len(omag1))
        bgdens = bgdict[key]((16.8,np.max(omag2)))
        print '%3i background stars'%(bgdens)

        counts, bins = np.histogram(lbr[0][((magerror1<.1)&(magerror2<.1))], bins=np.linspace(-20,0,200))
        counts = counts + np.random.poisson(bgdens*.1*.2, size=len(counts))
        centroids = (bins[1:] + bins[:-1]) / 2.
        err = np.sqrt(counts)
        bkg=0
        degree=1
        
        tdata= (counts-bkg)#/pp(centroids)
        terr = err#/pp(centroids)
        px, py= signal.csd(tdata,tdata, fs=1./(centroids[1]-centroids[0]),scaling='spectrum',nperseg=len(centroids))
        py= py.real
        px= 1./px
        py= numpy.sqrt(py*(centroids[-1]-centroids[0]))

        nerrsim= 10000
        ppy_err= numpy.empty((nerrsim,len(px)))
        for ii in range(nerrsim):
            tmock= terr*numpy.random.normal(size=len(centroids))
            ppy_err[ii]= signal.csd(tmock,tmock,
                                    fs=1./(centroids[1]-centroids[0]),scaling='spectrum',
                                    nperseg=len(centroids))[1].real
        py_err= numpy.sqrt(numpy.median(ppy_err,axis=0)*(centroids[-1]-centroids[0]))

        outdict[key] = outdict[key]+[py]
        errdict[key] = errdict[key]+[py_err]


plt.subplot(121)
for i,survey in enumerate(['WFIRST','LSST', 'SDSS']):
    plt.fill_between(px[:-1], np.nanpercentile(outdict[survey], (25), axis=0)[1:],  y2= np.nanpercentile(outdict[survey], (75), axis=0)[1:], color=colors[i], alpha=0.2)
    plt.loglog(px[:-1], (np.nanmedian(outdict[survey], axis=0)[1:]), c=colors[i], lw=2, label=survey)
    plt.loglog(px[:-1], (np.nanmedian(errdict[survey], axis=0)[1:]), c=colors[i], lw=2, ls='--')#, label='SDSS')

ax = plt.gca()
ax.set_xscale('log')
#plt.xlim([-.5,1.5])
plt.xlabel(r'$\mathrm{log10\ 1/k_\xi\ [deg]}$')
plt.ylabel(r'$\sqrt{\delta \delta}$')
plt.legend(loc='upper left')
from matplotlib.ticker import FuncFormatter
#ax.axis([0.4, 30, .03,.7])
ax.loglog()
for axis in [ax.xaxis, ax.yaxis]:
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    axis.set_major_formatter(formatter)


plt.subplot(122)
thresh = 2
offsets = [-.2,0,.2]
for i,survey in enumerate(['WFIRST','LSST', 'SDSS']):
    for j in np.arange(len(outdict[survey])):
        snrs = (outdict[survey][j]/errdict[survey][j])[:20]
        if j == 0:
            plt.scatter(sum(snrs>thresh)*np.ones(sum(snrs>thresh))-offsets[i], (snrs[snrs>thresh]), c=colors[i], s=10, alpha=.5, label=survey)
        else:
            plt.scatter(sum(snrs>thresh)*np.ones(sum(snrs>thresh))-offsets[i], (snrs[snrs>thresh]), c=colors[i], s=10, alpha=.5)

plt.legend()
plt.ylabel(r'SNR of each element')
plt.xlabel(r'Elements with SNR $>$ 2')


#######################################################
#######################################################
#######################################################
#make some mock pal5 data in ibata-like format
import iso_handling
import pal5_mock_ps
from numpy.polynomial import Polynomial
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
model = pal5_mock_ps.pal5_stream_instance()
model.add_smooth()

errormodels = iso_handling.load_errormodels()
isodict = iso_handling.load_iso_interps(remake=True)
try: 
    bgdict = np.load('/Users/hendel/projects/streamgaps/streampepper/bgdict.npy').item()
except:
    bgdict = iso_handling.gen_bg_counts_interp(isodict=isodict,errormodels=errormodels, verbose=True)

model.sample(ratemod=1.)
model.assign_masses()
masses = model.masses
#surveys=['SDSS', 'CFHT', 'LSST', 'LSST10','CASTOR', 'WFIRST']
surveys=['SDSS', 'CFHT','LSST', 'WFIRST']
for i, key in enumerate(surveys):
    if 'LSST' in key:
        mag1 = isodict['lsst_g-10.00-0.0001'](masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror1 = errormodels[key]['g'](mag1)
        mag2 = isodict['lsst_r-10.00-0.0001'](masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror2 = errormodels[key]['r'](mag2)
    elif key == 'CASTOR':
        mag1 = isodict['castor_u-10.00-0.0001'](masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror1 = errormodels[key]['u'](mag1)
        mag2 = isodict['castor_g-10.00-0.0001'](masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror2 = errormodels[key]['g'](mag2)
    elif key == 'WFIRST':
        mag1 = isodict['Z087-10.00-0.0001'](masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror1 = errormodels[key]['z'](mag1)
        mag2 = isodict['H158-10.00-0.0001'](masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror2 = errormodels[key]['h'](mag2)
    else:
        mag1 = isodict['sdss_g-10.00-0.0001'](masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror1 = errormodels[key]['g'](mag1)
        mag2 = isodict['sdss_r-10.00-0.0001'](masses) + (5*np.log10(model.spdata_rs*1e3)-5)
        magerror2 = errormodels[key]['r'](mag2)
    magsel = ((magerror1<.1)&(magerror2<.1))
    omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[magsel]
    omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[magsel]
    bgdens = bgdict[key]((16.8,np.max(omag2)))
    print bgdens
    #counts, bins = np.histogram(model.spdata[:,0], bins=np.linspace(-20,0,200))
    counts, bins,patch =plt.hist(model.spdata[:,0][magsel], bins=np.linspace(-20,0,200),histtype='step',lw=(3-i))
    counts = counts + np.random.poisson(bgdens*.1*.1, size=len(counts))
    centroids = (bins[1:] + bins[:-1]) / 2.
    err_upper = np.sqrt(counts)*2.
    err_lower = np.zeros(len(counts))
    np.savetxt('./data/fakeobs/'+key,np.vstack((abs(centroids),counts/.1)).T,delimiter=',')
    np.savetxt('./data/fakeobs/'+key+'_lower',np.vstack((abs(centroids),err_lower/.1)).T,delimiter=',')
    np.savetxt('./data/fakeobs/'+key+'_upper',np.vstack((abs(centroids),err_upper/.1)).T,delimiter=',')




