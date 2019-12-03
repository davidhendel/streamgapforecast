import iso_handling
import pal5_mock_ps
import scipy
model = pal5_mock_ps.pal5_stream_instance()
#model.add_smooth()
#model.sample()
#model.assign_masses()

errormodels = iso_handling.load_errormodels()
isodict = iso_handling.load_iso_interps(remake=True)
try: 
	bgdict = np.load('/Users/hendel/projects/streamgaps/streampepper/bgdict.npy').item()
except:
	bgdict = iso_handling.gen_bg_counts_interp(isodict=isodict,errormodels=errormodels, verbose=True)
	#bg_interp = iso_handling.gen_bg_counts_interp(surveys=['SDSS','CFHT','LSST','LSST10','CASTOR'], 
	# bands = ['gr','gr','gr','gr','ug'], isodict=isodict, errormodels=errormodels)

######################################################
######################################################
######################################################
#Figure 1: compare error models
plt.figure(figsize=(4,4))
mags = np.linspace(18,28,1000)
surveys=['CFHT', 'LSST', 'LSST10', 'WFIRST']
bands={'CFHT':['g','r'],'LSST':['g','r'],'LSST10':['g','r'],'WFIRST':['z','h']}
colors=['seagreen','firebrick', 'slateblue', 'darkorange']
linestyles = ['-',':']
linewidths = [2,2,2,2]
for i, key in enumerate(surveys):
	for j, band in enumerate(bands[key]):
		errs = errormodels[key][band](mags)
		plt.plot(mags[errs<.2], errs[errs<.2], label=(key+' '+band),color=colors[i], linestyle=linestyles[j], linewidth=linewidths[i])
plt.plot([18,28],[.2,.2],c='k', lw=1, linestyle='--')
plt.text(21.5,.2,r'$\mathrm{5\sigma\ detection}$', horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', edgecolor='none',alpha=1))
plt.plot([18,28],[.1,.1],c='k', lw=1, linestyle='--')
plt.text(21.5,.1,r'$\mathrm{10\sigma\ detection}$', horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', edgecolor='none',alpha=1))
plt.xlim(20,28)
plt.ylim(0,.28)
plt.xlabel('Magnitude [mag]')
plt.ylabel('Magnitude uncertainty [mag]')
plt.legend(loc='upper right', ncol=2, fontsize='small')
plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/fig_photerr.png',dpi=300,bbox_inches='tight')

######################################################
######################################################
######################################################
#Figure 1: fraction of stars with xi<12.6
plt.figure(figsize=(4,4))
d = np.loadtxt('./data/abcsamples/samp_10k_5-9_000bg_cdm.dat',skiprows=0,delimiter=',', max_rows=1)
counts = np.loadtxt('./data/abcsamples/samp_10k_5-9_000bg_cdm.dat',skiprows=1,delimiter=',')
plt.hist(np.sum(counts[:,0:127],axis=1)/10000.,bins=np.linspace(0,1,100),density=True,color='0.1')
plt.xlim(0.4,0.9)
plt.xlabel(r'$\mathrm{Fraction\ of\ stars\ with\ \xi<12.6^\circ}$')
plt.ylabel(r'$\mathrm{PDF}$')
plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/fig_fstars.png',dpi=300,bbox_inches='tight')


######################################################
######################################################
######################################################
#Stream and background counts
#g = gSDSS - 0.185(gSDSS - rSDSS)
import scipy
import iso_handling
errormodels = iso_handling.load_errormodels()
isodict = iso_handling.load_iso_interps(remake=False, maxlabel=2)
dm = (5*np.log10(23.2*1e3)-5)
ms = np.linspace(0,.8,1000)
gSDSS=isodict['sdss_g-10.06-0.0008'](ms) + dm
rSDSS=isodict['sdss_r-10.06-0.0008'](ms) + dm
gCFHT = gSDSS - 0.185*(gSDSS - rSDSS)
mag_lim_gSDSS = gSDSS[np.argmax(gCFHT<23.5)]
m_lim_CFHT = ms[np.argmax(gCFHT<23.5)]

maglim_gLSST  =iso_handling.getMagLimit('g',survey='LSST')
maglim_rLSST  =iso_handling.getMagLimit('r',survey='LSST')
maglim_gLSST10=iso_handling.getMagLimit('g',survey='LSST10')
maglim_rLSST10=iso_handling.getMagLimit('r',survey='LSST10')
maglim_hWFIRST=iso_handling.getMagLimit('h',survey='WFIRST')
maglim_zWFIRST=iso_handling.getMagLimit('z',survey='WFIRST')

gLSST  =isodict['lsst_g-10.06-0.0008'](ms) + dm
rLSST  =isodict['lsst_r-10.06-0.0008'](ms) + dm
gLSST10=isodict['lsst_g-10.06-0.0008'](ms) + dm
rLSST10=isodict['lsst_r-10.06-0.0008'](ms) + dm
hWFIRST=isodict['wfirst_h-10.06-0.0008'](ms) + dm
zWFIRST=isodict['wfirst_z-10.06-0.0008'](ms) + dm

m_lim_gLSST  = ms[np.argmax(gLSST   < maglim_gLSST  )]
m_lim_rLSST  = ms[np.argmax(rLSST   < maglim_rLSST  )]
m_lim_gLSST10= ms[np.argmax(gLSST10 < maglim_gLSST10)]
m_lim_rLSST10= ms[np.argmax(rLSST10 < maglim_rLSST10)]
m_lim_hWFIRST= ms[np.argmax(hWFIRST < maglim_hWFIRST)]
m_lim_zWFIRST= ms[np.argmax(zWFIRST < maglim_zWFIRST)]

m_lim_LSST  =np.max((m_lim_gLSST,m_lim_rLSST))
m_lim_LSST10=np.max((m_lim_gLSST10,m_lim_rLSST10))
m_lim_WFIRST=np.max((m_lim_hWFIRST,m_lim_zWFIRST))

CFHT_integral   = scipy.integrate.quad(lambda x: x**(-0.5),m_lim_CFHT,  0.8273221254)[0]
LSST_integral   = scipy.integrate.quad(lambda x: x**(-0.5),m_lim_LSST,  0.8273221254)[0]
LSST10_integral = scipy.integrate.quad(lambda x: x**(-0.5),m_lim_LSST10,0.8273221254)[0]
WFIRST_integral = scipy.integrate.quad(lambda x: x**(-0.5),m_lim_WFIRST,0.8273221254)[0]

galbg = iso_handling.gen_gal_counts_interp(dms =[16.,dm],surveys=['CFHT', 'LSST', 'LSST10', 'WFIRST'],
	bands=['gr', 'gr', 'gr', 'zh'], isodict=isodict, errormodels=errormodels,verbose=True)
starbg = iso_handling.gen_bg_counts_interp(dms=[16.,dm],surveys=['CFHT', 'LSST', 'LSST10', 'WFIRST'], 
	bands=['gr', 'gr', 'gr', 'zh'], isodict=isodict, errormodels=errormodels,verbose=True)

#isodict_max2 = iso_handling.load_iso_interps(remake=True, maxlabel=2)
#galbg_max2 = iso_handling.gen_gal_counts_interp(dms =[16.,dm],surveys=['CFHT', 'LSST', 'LSST10', 'WFIRST'],
#	bands=['gr', 'gr', 'gr', 'zh'], isodict=isodict_max2, errormodels=errormodels,verbose=True)
#starbg_max2 = iso_handling.gen_bg_counts_interp(dms=[16.,dm],surveys=['CFHT', 'LSST', 'LSST10', 'WFIRST'], 
#	bands=['gr', 'gr', 'gr', 'zh'], isodict=isodict_max2, errormodels=errormodels,verbose=True)


data = [
('CFHT',   2000*1.6*CFHT_integral/CFHT_integral  , np.round(starbg['CFHT']((dm,23.5          ))+galbg['CFHT']((dm,23.5          )))),
('LSST',   2000*1.6*LSST_integral/CFHT_integral  , np.round(starbg['LSST']((dm,maglim_gLSST  ))+galbg['LSST']((dm,maglim_gLSST  )))),
('LSST 10',2000*1.6*LSST10_integral/CFHT_integral, np.round(starbg['LSST10']((dm,maglim_gLSST10))+galbg['LSST10']((dm,maglim_gLSST10)))),
('WFIRST', 2000*1.6*WFIRST_integral/CFHT_integral, np.round(starbg['WFIRST']((dm,maglim_zWFIRST))+galbg['WFIRST']((dm,maglim_zWFIRST))))
]

from astropy.table import Table
table = Table(rows=data,names = ('Survey', 'Samples', 'Background'), dtype=('S','i','i'))
table.write('/Users/hendel/projects/streamgaps/streampepper/paper_figures/samptable_cmd33.tex',format='latex',overwrite=True)




#####################################################
#####################################################
#####################################################
#plot CMDs for CFHT, LSST, LSST10, WFIRST
from scipy.interpolate import interp1d
isodata = scipy.genfromtxt('/Users/hendel/data/isochrones/newsdsspal5.txt', names = True, skip_header = 11)
sel = ((isodata['logAge']==10.06005))#&(isodata['label']<=maxlabel))
pal5sdssiso_g = interp1d(isodata['Mini'][sel], isodata['gmag'][sel], fill_value=99, bounds_error=False)
pal5sdssiso_r = interp1d(isodata['Mini'][sel], isodata['rmag'][sel], fill_value=99, bounds_error=False)

surveys=['CFHT', 'LSST', 'LSST10', 'WFIRST']

model = pal5_mock_ps.pal5_stream_instance()
model.sample(nsample=13323)
model.assign_masses(n=13323, maxmass=pal5sdssiso_g.x[-2])

plt.figure(figsize=(9,5))
plt.subplot(1,4,1)
mlim = m_lim_CFHT
mag1 = isodict['cfht_g-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['CFHT']['g'](mag1)
mag2 = isodict['cfht_r-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['CFHT']['r'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
plt.scatter(omag1-omag2, omag1, c='k', alpha=0.3,s=1)
plt.ylim(28,14)
plt.xlabel('g-r')
plt.ylabel('g')
plt.xlim(-0,1.5)
plt.gca().set_title('CFHT')

plt.subplot(1,4,2)

mlim = m_lim_LSST
mag1 = isodict['lsst_g-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['LSST']['g'](mag1)
mag2 = isodict['lsst_r-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['LSST']['r'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
plt.scatter(omag1-omag2, omag1, c='k', alpha=0.3,s=1)
plt.ylim(28,14)
plt.xlabel('g-r')
plt.ylabel('g')
plt.xlim(0,1.5)
plt.gca().set_title('LSST')

plt.subplot(1,4,3)
mlim = m_lim_LSST10
mag1 = isodict['lsst_g-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['LSST10']['g'](mag1)
mag2 = isodict['lsst_r-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['LSST10']['r'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
plt.scatter(omag1-omag2, omag1, c='k', alpha=0.3,s=1)
plt.ylim(28,14)
plt.xlabel('g-r')
plt.ylabel('g')
plt.xlim(0,1.5)
plt.gca().set_title('LSST 10')

plt.subplot(1,4,4)
mlim = m_lim_WFIRST
mag1 = isodict['wfirst_h-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['WFIRST']['h'](mag1)
mag2 = isodict['wfirst_z-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['WFIRST']['z'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
plt.scatter(omag2-omag1, omag1, c='k', alpha=0.3,s=1)
plt.ylim(28,14)
plt.xlabel('z-h')
plt.ylabel('h')
plt.xlim(.25,1.75)
plt.gca().set_title('WFIRST')
plt.subplots_adjust(wspace=.3)

plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/fig_cmds.png',dpi=300,bbox_inches='tight')

#####################################################
#####################################################
#####################################################
#plot sky histograms for CFHT, LSST, LSST10, WFIRST

model.sample(nsample=13323)
model.assign_masses(n=13323, maxmass=pal5sdssiso_g.x[-2])
plt.figure(figsize=(4,8))
plt.subplot(4,1,1)
mlim = m_lim_CFHT
mag1 = isodict['cfht_g-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['CFHT']['g'](mag1)
mag2 = isodict['cfht_r-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['CFHT']['r'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
xi  = model.spxi[:,0][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
eta = model.spxi[:,1][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
print xi.shape
plt.hist2d(
	np.concatenate((xi,15*numpy.random.uniform(size=15*7*844))),
	np.concatenate((eta,7*numpy.random.uniform(size=15*7*844))),
	bins=[np.linspace(0,15,150),np.linspace(0,7,70)],
	vmin=0,vmax=20, cmap='gray')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.text(1,5,'CFHT',color='w', bbox=dict(facecolor='k', alpha=0.5))

plt.subplot(4,1,2)

mlim = m_lim_LSST
mag1 = isodict['lsst_g-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['LSST']['g'](mag1)
mag2 = isodict['lsst_r-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['LSST']['r'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
xi  = model.spxi[:,0][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
eta = model.spxi[:,1][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
print xi.shape
plt.hist2d(
	np.concatenate((xi,15*numpy.random.uniform(size=15*7*324))),
	np.concatenate((eta,7*numpy.random.uniform(size=15*7*324))),
	bins=[np.linspace(0,15,150),np.linspace(0,7,70)],
	vmin=0,vmax=20, cmap='gray')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.text(1,5,'LSST',color='w', bbox=dict(facecolor='k', alpha=0.5))

plt.subplot(4,1,3)
mlim = m_lim_LSST10
mag1 = isodict['lsst_g-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['LSST10']['g'](mag1)
mag2 = isodict['lsst_r-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['LSST10']['r'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
xi  = model.spxi[:,0][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
eta = model.spxi[:,1][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
print xi.shape
plt.hist2d(
	np.concatenate((xi,15*numpy.random.uniform(size=15*7*245))),
	np.concatenate((eta,7*numpy.random.uniform(size=15*7*245))),
	bins=[np.linspace(0,15,150),np.linspace(0,7,70)],
	vmin=0,vmax=20, cmap='gray')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.text(1,5,'LSST 10',color='w', bbox=dict(facecolor='k', alpha=0.5))

plt.subplot(4,1,4)
mlim = m_lim_WFIRST
mag1 = isodict['wfirst_h-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['WFIRST']['h'](mag1)
mag2 = isodict['wfirst_z-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['WFIRST']['z'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
xi  = model.spxi[:,0][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
eta = model.spxi[:,1][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
print xi.shape
h,xe,ye,im = plt.hist2d(
	np.concatenate((xi,15*numpy.random.uniform(size=15*7*187))),
	np.concatenate((eta,7*numpy.random.uniform(size=15*7*187))),
	bins=[np.linspace(0,15,150),np.linspace(0,7,70)],
	vmin=0,vmax=20, cmap='gray')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.text(1,5,'WFIRST',color='w', bbox=dict(facecolor='k', alpha=0.5))

plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/fig_hists.png',dpi=300,bbox_inches='tight')


#####################################################
#####################################################
#####################################################
#Power spectra

def xi_csd(d, bins=np.linspace(0,15,151), nbg=0., binned=False):
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
	#np.save('/Users/hendel/Desktop/pscrosscheck_xi_csd.npy',(d,bins,counts,tdata,terr,px,py,py_err))
	return px,py,py_err

nruns = 100
xi_cfht =   scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abc_3200_7-9_008bg.dat',delimiter=',', loose=True, invalid_raise=False)
xi_lsst =   scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abc_6144_7-9_006bg.dat',delimiter=',', loose=True, invalid_raise=False)
xi_lsst10 = scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abc_8245_7-9_004bg.dat',delimiter=',', loose=True, invalid_raise=False)
xi_wfirst = scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abc_15577_7-9_004bg.dat',delimiter=',',loose=True, invalid_raise=False)
samp_cfht = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/samp_3200_7-9_008bg.dat',delimiter=',', skiprows=1, max_rows=10000)
samp_lsst = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/samp_6144_7-9_006bg.dat',delimiter=',', skiprows=1, max_rows=10000)
samp_lsst10 = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/samp_8245_7-9_004bg.dat',delimiter=',', skiprows=1, max_rows=10000)
samp_wfirst = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/samp_15577_7-9_004bg.dat',delimiter=',', skiprows=1, max_rows=10000)

px,py,py_err = xi_csd(samp_cfht[3,1:],binned=True)

# Do a Epanechnikov KDE estimate of the PDF in the transformed y=(1+x)/(1-x) space
def kde_epanechnikov(x,h,ydata):
	"""ydata= ln[(1+xdata)/(1-xdata)]"""
	h= numpy.ones_like(x)*h
	h[x < -0.5]= h[x < -0.5]*(-2.*(x[x < -0.5]+0.5)+1.) # use slightly wider kernel at small values
	y= numpy.log((1.6+x)/(1.6-x))
	#r= numpy.fabs(numpy.tile(y,(len(ydata),1)).T-ydata)/h
	r= numpy.fabs(numpy.tile(y,(len(ydata),1)).T-ydata)/numpy.tile(h,(len(ydata),1)).T
	r[r > 1.]= 1. # Gets around multi-D slicing
	return numpy.sum(0.75*(1.-r**2.),axis=1)/h*(1./(1.6+x)+1./(1.6-x))

n=1
plt.figure(figsize=(4,4))

labels=['cfht','lsst','lsst10','wfirst']
colors=['seagreen','firebrick', 'slateblue', 'darkorange']
for i, d in enumerate([xi_cfht[:,1:],xi_lsst[:,1:],xi_lsst10[:,1:],xi_wfirst[:,1:]]):

	#px,py,py_err = xi_csd(samp_cfht[3], nbg=0, binned=True)
	#plt.loglog(px,np.nanmedian(d[0::3,1:][ d[0::3,0] > .5],axis=0), color = 'r',               ) 
	#plt.loglog(px,np.nanmedian(d[0::3,1:][ d[0::3,0] < -.5],axis=0), color = 'g',              ) 
	#plt.loglog(px,np.nanmedian(d[0::3,1:][(d[0::3,0]> -.5)&(d[0::3,0]<.5)],axis=0), color = 'b', ) 
	#plt.loglog(px,py_err,c='k')

	data = np.nanmedian(d[:,1:],axis=0)#d[n,1:]
	eps = 2*np.nanmedian(np.nanmedian(d[:,1:],axis=0)[20:-20])
	sindx= ((d[:,1]-data[0]) < eps)*((d[:,2]-data[1]) < eps)*((d[:,3]-data[2]) <eps)\
							*((d[:,4]-data[3]) < 2*eps)*((d[:,5]--data[4]) < 2*eps)


	scale=1.
	kernel_width=.3
	rate_full= d[sindx,0]
	xxs= numpy.linspace(-1.5,1.,151)
	kdey_full= kde_epanechnikov(xxs,kernel_width,numpy.log((1.6+rate_full)/(1.6-rate_full)))\
		+numpy.random.uniform(size=len(xxs))*0.000001
	kdey_full/= numpy.sum(kdey_full)*(xxs[1]-xxs[0])  
	plt.plot(xxs,kdey_full*scale,label=labels[i],color=colors[i])#,'-',lw=3.,color=color,zorder=zorder,overplot=overplot)

	# Get peak and 68% around the peak
	bf= xxs[numpy.argmax(kdey_full)]
	sindx= numpy.argsort(-kdey_full) # minus reverses sort
	cp= numpy.cumsum((kdey_full/numpy.sum(kdey_full))[sindx])
	m68= xxs[sindx][cp > 0.68]
	uplim68= numpy.amin(m68[m68 > bf])
	lowlim68= numpy.amax(m68[m68 < bf])

	m,h,l=(10.**bf,10.**uplim68-10.**bf,10.**bf-10.**lowlim68)
	plt.errorbar([bf],[(i+1)*.05],xerr=array([(bf-lowlim68),-(bf-uplim68)]).reshape(2,1),color=colors[i], marker='o',markersize=5)


plt.legend()
plt.xlim(-1.5,1)
plt.ylim(.0,1.2)
plt.xlabel('log10 rate')
plt.ylabel('PDF')
#lt.plot([d[n,0],d[n,0]],[-10,10],c='k')

plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/fig_abc1.png',dpi=300,bbox_inches='tight')








plt.loglog(px,np.nanmedian(xi_cfht[:,2:],axis=0),  marker='o',markersize=5, label ='CFHT', color = 'orange') 
eps = np.nanmedian(np.nanmedian(xi_cfht[:,2:],axis=0)[20:-20])
plt.plot([.001,100],[eps,eps],linestyle='--',  color = 'orange')
plt.fill_between(px, np.nanpercentile(xi_cfht[:,2:], (25), axis=0),  y2= np.nanpercentile(xi_cfht[:,2:], (75), axis=0), color='orange', alpha=0.2)
plt.loglog(px,np.nanmedian(xi_lsst[:,2:],axis=0),  marker='o',markersize=5, label ='LSST', color = 'seagreen') 
eps = np.nanmedian(np.nanmedian(xi_lsst[:,2:],axis=0)[20:-20])
plt.plot([.001,100],[eps,eps],linestyle='--',  color = 'seagreen')
plt.fill_between(px, np.nanpercentile(xi_lsst[:,2:], (25), axis=0),  y2= np.nanpercentile(xi_lsst[:,2:], (75), axis=0), color='seagreen', alpha=0.2)
plt.loglog(px,np.nanmedian(xi_lsst10[:,2:],axis=0),marker='o',markersize=5, label ='LSST 10', color = 'darkslateblue')
eps = np.nanmedian(np.nanmedian(xi_lsst10[:,2:],axis=0)[20:-20])
plt.plot([.001,100],[eps,eps],linestyle='--',  color = 'darkslateblue') 
plt.fill_between(px, np.nanpercentile(xi_lsst10[:,2:], (25), axis=0),  y2= np.nanpercentile(xi_lsst10[:,2:], (75), axis=0), color='darkslateblue', alpha=0.2)
plt.loglog(px,np.nanmedian(xi_wfirst[:,2:],axis=0),marker='o',markersize=5, label ='WFIRST', color = 'tomato') 
eps = np.nanmedian(np.nanmedian(xi_wfirst[:,2:],axis=0)[20:-20])
plt.plot([.001,100],[eps,eps],linestyle='--',  color = 'tomato')
plt.fill_between(px, np.nanpercentile(xi_wfirst[:,2:], (25), axis=0),  y2= np.nanpercentile(xi_wfirst[:,2:], (75), axis=0), color='tomato', alpha=0.2)
plt.legend(loc='upper left')
plt.ylabel('Density power')
plt.xlabel(r'$\xi$ [deg]')
plt.xlim(0.5,15)
plt.ylim(.06,.6)






