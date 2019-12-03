import matplotlib.path as mpltPath
import scipy

'''
Use HLF 2.0 to estimate the number of unresolved background galaxies
Need at least g or r and H - Sloan g (F475W), r (F625W), i (F775W), and z (F850LP)

Counts = 10**( (m_AB - zp_AB) / (-2.5) )
m_AB = -2.5*np.log10(counts)+zp

HLF Filter  Zero Point (ABMAG)
WFC3/UV
F225W   24.04
F275W   24.13
F336W   24.67
ACS/WFC
F435W   25.68
F606W   26.51
F775W   25.69
F814W   25.94
F850LP  24.87
WFC3/IR
F098M   25.68
F105W   26.27
F125W   26.23
F140W   26.45
F160W   25.94
'''

hlf = scipy.genfromtxt('/Users/hendel/data/hlsp_hlf_hst_60mas_goodss_v2.0_catalog.txt',names=True)

xdf_outline = [ (53.187414,-27.779152), (53.159507,-27.759633), (53.134517,-27.787144), (53.161906,-27.807208) ]
points = zip(hlf['ra'],hlf['dec'])
path = mpltPath.Path(xdf_outline)
inside = path.contains_points(points)
xhlf = hlf[inside]

xsel = ((xhlf['f_f160w']>0)&(xhlf['f_f850lp']>0))
xflags = xhlf['star_flag'][xsel]
xH160 =  -2.5*np.log10(xhlf['f_f160w'][xsel])+25.94
xZ085 =  -2.5*np.log10(xhlf['f_f850lp'][xsel])+24.87

gsel = ((hlf['f_f160w']>0)&(hlf['f_f850lp']>0))
gflags = hlf['star_flag'][gsel]
gH160 =  -2.5*np.log10(hlf['f_f160w'][gsel])+25.94
gZ085 =  -2.5*np.log10(hlf['f_f850lp'][gsel])+24.87


#####################################################
#####################################################
#####################################################
#Plot flux_radius vs H for XDF and GOODS-s
plt.figure(figsize=(9,5))
plt.subplot(122)
plt.title('XDF')
plt.scatter(xH160,xhlf['flux_radius'][xsel]*.06, s=1,c='k')
plt.scatter(xH160[xflags==0],(xhlf['flux_radius'][xsel]*.06)[xflags==0], s=1,c='b', label='extended')
plt.scatter(xH160[xflags==1],(xhlf['flux_radius'][xsel]*.06)[xflags==1], s=3,c='r', label='star')
plt.legend()
plt.xlabel('H160 mag')
plt.ylabel('Flux radius (arcsec)')
plt.ylim(-.1,1.2)
plt.xlim(15,35)

plt.subplot(121)
plt.title('GOODS-S')
plt.scatter(gH160,hlf['flux_radius'][gsel]*.06, s=1,c='k')
plt.scatter(gH160[gflags==0],(hlf['flux_radius'][gsel]*.06)[gflags==0], s=1,c='b', label='extended')
plt.scatter(gH160[gflags==1],(hlf['flux_radius'][gsel]*.06)[gflags==1], s=3,c='r', label='star')
plt.legend()
plt.xlabel('H160 mag')
plt.ylabel('Flux radius (arcsec)')
plt.ylim(-.1,1.2)
plt.xlim(15,35)

plt.savefig('/Users/hendel/Desktop/h160.png')

#####################################################
#####################################################
#####################################################
#Almost all objects with flux_radius < .2 arcsec at H~26 are galaxies
plt.figure()
plt.hist(gH160[gflags==0][(hlf['flux_radius'][gsel]*.06)[gflags==0]<.2],bins = np.linspace(20,35,50),color='b', label='extended')
plt.hist(gH160[gflags==1][(hlf['flux_radius'][gsel]*.06)[gflags==1]<.2],bins = np.linspace(20,35,50),color='r', label='star')

#####################################################
#####################################################
#####################################################
#Seems like non-XDF is good (completeness > 80%) to H~28
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.xlabel('H160 mag')
plt.ylabel('counts')
gn,gbins,gpatches = plt.hist(gH160, bins = np.linspace(20,35,50), histtype='step', lw=2, color='k', label = 'GOODS-S')
xn,xbins,xpatches =plt.hist(xH160, bins = np.linspace(20,35,50), histtype='step', lw=2, color='r', label = 'XDFx45', weights=(45*np.ones(len(xH160))))
plt.legend(loc='upper left')
plt.xlim(20,35)
plt.subplot(122)
plt.xlabel('H160 mag')
plt.ylabel('Completeness')
plt.plot(gbins[:-1]+.40816327/2, gn/xn, c='k', lw=3)
plt.plot([0,40],[1,1],c='k', ls='--')
plt.xlim(20,35)

plt.savefig('/Users/hendel/Desktop/h160_completeness.png')

#####################################################
#####################################################
#####################################################
#Unfortunately the HLF does not have F475W or F625W, 
#the Sloan g and r equivalents. Estimate  completeness & 
#background from F775W i.e. Sloan i instead.
#Seems to be complete (>90%) to F775=28.2

nsel = ((hlf['f_f160w']>0)&(hlf['f_f775w']>0))
nflags = hlf['star_flag'][nsel]
nH160 =  -2.5*np.log10(hlf['f_f160w'][nsel])+25.94
nF775 =  -2.5*np.log10(hlf['f_f775w'][nsel])+25.69

nxsel = ((xhlf['f_f160w']>0)&(xhlf['f_f775w']>0))
nxflags = xhlf['star_flag'][nxsel]
nxH160 =  -2.5*np.log10(xhlf['f_f160w'][nxsel])+25.94
nxF775 =  -2.5*np.log10(xhlf['f_f775w'][nxsel])+25.69

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.xlabel('F775 mag')
plt.ylabel('counts')
gn,gbins,gpatches =plt.hist(nF775, bins = np.linspace(20,35,50), histtype='step', lw=2, color='k', label = 'GOODS-S')
xn,xbins,xpatches =plt.hist(nxF775, bins = np.linspace(20,35,50), histtype='step', lw=2, color='r', label = 'XDFx17', weights=(17*np.ones(len(nxF775))))
plt.legend(loc='upper left')
plt.xlim(20,35)
plt.subplot(122)
plt.xlabel('F775 mag')
plt.ylabel('Completeness')
plt.plot(gbins[:-1]+.40816327/2, gn/xn, c='k', lw=3)
plt.plot([0,40],[1,1],c='k', ls='--')
plt.xlim(20,35)

plt.savefig('/Users/hendel/Desktop/f775_completeness.png')


#####################################################
#####################################################
#####################################################
#F336 for CASTOR u?
#Seems to be complete (>80%) to F336=27.2

nsel = ((hlf['f_f160w']>0)&(hlf['f_f336w']>0))
nflags = hlf['star_flag'][nsel]
nH160 =  -2.5*np.log10(hlf['f_f160w'][nsel])+25.94
nF336 =  -2.5*np.log10(hlf['f_f336w'][nsel])+24.67

nxsel = ((xhlf['f_f160w']>0)&(xhlf['f_f336w']>0))
nxflags = xhlf['star_flag'][nxsel]
nxH160 =  -2.5*np.log10(xhlf['f_f160w'][nxsel])+25.94
nxF336 =  -2.5*np.log10(xhlf['f_f336w'][nxsel])+24.67

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.xlabel('F336 mag')
plt.ylabel('counts')
gn,gbins,gpatches =plt.hist(nF336, bins = np.linspace(20,35,50), histtype='step', lw=2, color='k', label = 'GOODS-S')
xn,xbins,xpatches =plt.hist(nxF336, bins = np.linspace(20,35,50), histtype='step', lw=2, color='r', label = 'XDFx29', weights=(29*np.ones(len(nxF336))))
plt.legend(loc='upper left')
plt.xlim(20,35)
plt.subplot(122)
plt.xlabel('F336 mag')
plt.ylabel('Completeness')
plt.plot(gbins[:-1]+.40816327/2, gn/xn, c='k', lw=3)
plt.plot([0,40],[1,1],c='k', ls='--')
plt.xlim(20,35)


plt.savefig('/Users/hendel/Desktop/f336_completeness.png')

#####################################################
#####################################################
#####################################################
#Number of unresolved galaxies per sq deg
#for Sloan i at 0.4 arcsec and H/u at 0.2 arcsec








