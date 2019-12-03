from astropy import table
from scipy.interpolate import interp1d
import ebf
import astropy.table as atpy
import numpy as np
import string
import pickle
import dill
import scipy
from matplotlib.colors import LogNorm
import read_girardi
import scipy.spatial
import astropy.units as auni
import scipy.interpolate
import scipy.optimize
import glob
from galpy.util import save_pickles
import time

import astropy.units as u

def load_iso_interps(remake=False, save=False, maxlabel=2):
    """
        Load or create some useful isochrone interpolators.

        Metal fractions from from 0.0001 to 0.03, logages from 6.6 to 10.2.

        Complete filter sets for WFIRST, LSST, 2MASS and SDSS unless specified (38 (zs) * 21 (bands) * ~172 (ages)= 137260 isochrones)

        SDSS (AB), 2MASS (Vega), WFIRST (Vega), and LSST (AB) from CMD 3.3; Galaxia SDSS isochrones (CMD 2.1) are commented out

        Castor from David V-G, Castor NFC WG - something weird going on, commented out

        if remake==True, reinitialize the interpolators; else reload the pickled version.

        if save==True, save the remade version, otherwise return without overwriting the save.
        
        maxlabel is the latest evolutionary phase used, from CMD; should be <9 
            0 = PMS, pre main sequence
            1 = MS, main sequence
            2 = SGB, subgiant branch, or Hertzsprung gap for more intermediate+massive stars
            3 = RGB, red giant branch, or the quick stage of red giant for intermediate+massive stars
            4 = CHEB, core He-burning for low mass stars, or the very initial stage of CHeB for intermediate+massive stars
            5 = still CHEB, the blueward part of the Cepheid loop of intermediate+massive stars
            6 = still CHEB, the redward part of the Cepheid loop of intermediate+massive stars
            7 = EAGB, the early asymptotic giant branch, or a quick stage of red giant for massive stars
            8 = TPAGB, the thermally pulsing asymptotic giant branch
            9 = post-AGB (in preparation!)

        Usage
            absolute mag = interpdict['band-age-z'](mass)
            accepts vectors
            out-of-bounds gives 99s
            set of bands, ages, and metallicities stored under keywords 'bands','ages','zs'

    """    

    if remake==False: 
        isointerp = dill.load(open("/Users/hendel/projects/streamgaps/streampepper/SDSS2MASSLSSTWFIRST_iso_interp_dict_maxlabel%i.pkl"%maxlabel, "rb" ))
        return isointerp
    if remake==True:

        interpdict={}

        """
        ##############################
        #Castor

        castor_files = np.sort(glob.glob("/Users/hendel/data/isochrones/castor/*"))
        castor_zs = [string.split(file, '_')[1][1:] for file in castor_files]
        castor_bands = ['Cas_uv', 'Cas_uvD', 'Cas_u', 'Cas_uW', 'Cas_g']
        tb = {
        'Cas_uv':'castor_uv',
        'Cas_uvD':'castor_uvD',
        'Cas_u':'castor_u',
        'Cas_uW':'castor_uW',
        'Cas_g':'castor_g'
        }
        
        for i, file in enumerate(castor_files):
            isodata = scipy.genfromtxt(file, names = True, skip_header = 17)
            colnames = isodata.dtype.names
  
            zp = (isodata['Vmag']+2.5*np.log10(isodata['Vnew']))[0]

            castor_ages = np.unique(isodata['Age_yr'])

            for j, age in enumerate(castor_ages):
                for band in castor_bands:
                    sel = ((isodata['Age_yr']==age))#&(isodata['label']<=maxlabel)) no labels for these
                    interpdict[tb[band]+'-%1.2f'%(np.log10(age))+'-%1.4f'%(float(castor_zs[i]))] = interp1d(isodata['Mini_Mo'][sel], (-2.5*np.log10(isodata[band]) + zp)[sel], fill_value=99, bounds_error=False)

        interpdict['castor_zs']  =np.array(castor_zs).astype(float)
        interpdict['castor_ages']=castor_ages
        """

        """
        ##############################
        #Galaxia SDSS

        sdss_files = np.sort(glob.glob("/Users/hendel/modules/Galaxia/GalaxiaData/Isochrones/padova/SDSS/output*"))
        sdss_zs = [string.split(string.split(file, '_')[1],'.dat')[0] for file in sdss_files]
        sdss_bands = ['u', 'g', 'r', 'i', 'z']
        tb = {
        'u':'sdss_u',
        'g':'sdss_g',
        'r':'sdss_r',
        'i':'sdss_i',
        'z':'sdss_z'
        }
        
        for i, file in enumerate(sdss_files):
            isodata = scipy.genfromtxt(file, names = True, skip_header = 9)
            colnames = isodata.dtype.names

            sdss_ages = np.unique(isodata['logageyr'])

            for j, age in enumerate(sdss_ages):
                for band in sdss_bands:
                    sel = ((isodata['logageyr']==age))#&(isodata['label']<=maxlabel)) no labels for these
                    interpdict[tb[band]+'-%1.2f'%(age)+'-%1.4f'%(float(sdss_zs[i]))] = interp1d(isodata['M_ini'][sel], isodata[band][sel], fill_value=99, bounds_error=False)

        interpdict['sdss_zs']  =np.array(sdss_zs).astype(float)
        interpdict['sdss_ages']=sdss_ages
        """

        ##############################
        #SDSS & 2MASS

        sdss2mass_files = np.sort(glob.glob("/Users/hendel/data/isochrones/sdss2mass/*")) #note LSST are ABmags and WFIRST are Vegamags
        sdss2mass_zs = [string.split(string.split(file, '_')[1],'.txt')[0] for file in sdss2mass_files]
        sdss2mass_bands = ['umag', 'gmag', 'rmag', 'imag', 'zmag', 'Jmag', 'Hmag', 'Ksmag']

        tb = {'umag':'sdss_u', 'gmag':'sdss_g', 'rmag':'sdss_r', 'imag':'sdss_i', 'zmag':'sdss_z', 'Jmag':'2mass_J', 'Hmag':'2mass_H', 'Ksmag':'2mass_Ks'}

        for i, file in enumerate(sdss2mass_files):
            isodata = scipy.genfromtxt(file, names = True, skip_header = 11)
            colnames = isodata.dtype.names

            sdss2mass_ages = np.unique(isodata['logAge'])

            for j, age in enumerate(sdss2mass_ages):
                for band in sdss2mass_bands:
                    sel = ((isodata['logAge']==age)&(isodata['label']<=maxlabel))
                    interpdict[tb[band]+'-%1.2f'%(age)+'-%1.4f'%(float(sdss2mass_zs[i]))] = interp1d(isodata['Mini'][sel], isodata[band][sel], fill_value=99, bounds_error=False)
                sdss_g_mag = interpdict['sdss_g'+'-%1.2f'%(age)+'-%1.4f'%(float(sdss2mass_zs[i]))].y
                sdss_r_mag = interpdict['sdss_r'+'-%1.2f'%(age)+'-%1.4f'%(float(sdss2mass_zs[i]))].y
                interpdict['cfht_g'+'-%1.2f'%(age)+'-%1.4f'%(float(sdss2mass_zs[i]))] = interp1d(isodata['Mini'][sel], (sdss_g_mag-0.185*(sdss_g_mag-sdss_r_mag)) , fill_value=99, bounds_error=False)
                interpdict['cfht_r'+'-%1.2f'%(age)+'-%1.4f'%(float(sdss2mass_zs[i]))] = interp1d(isodata['Mini'][sel], (sdss_r_mag-0.024*(sdss_g_mag-sdss_r_mag)) , fill_value=99, bounds_error=False)

        interpdict['sdss2mass_zs']=np.array(sdss2mass_zs).astype(float)
        interpdict['sdss2mass_ages']=sdss2mass_ages
        interpdict['cfht_zs']=np.array(sdss2mass_zs).astype(float)
        interpdict['cfht_ages']=sdss2mass_ages
        interpdict['cfht_bands']=['cfht_g', 'cfht_r']

        ##############################
        #LSST & WFIRST

        lsstwfirst_files = np.sort(glob.glob("/Users/hendel/data/isochrones/lsstwfirst/*")) #note LSST are ABmags and WFIRST are Vegamags
        lsstwfirst_zs = [string.split(string.split(file, '_')[1],'.txt')[0] for file in lsstwfirst_files]
        lsstwfirst_bands = ['umag', 'gmag', 'rmag', 'imag', 'zmag', 'Ymag', 'R062mag', 'Z087mag', 'Y106mag', 'J129mag', 'H158mag', 'F184mag', 'W149mag']

        tb = {'umag':'lsst_u', 'gmag':'lsst_g', 'rmag':'lsst_r', 'imag':'lsst_i', 'zmag':'lsst_z', 'Ymag':'lsst_y', 'R062mag':'wfirst_r', 'Z087mag':'wfirst_z', 'Y106mag':'wfirst_y', 'J129mag':'wfirst_j', 'H158mag':'wfirst_h', 'F184mag':'wfirst_h', 'W149mag':'wfirst_w'}

        for i, file in enumerate(lsstwfirst_files):
            isodata = scipy.genfromtxt(file, names = True, skip_header = 11)
            colnames = isodata.dtype.names

            lsstwfirst_ages = np.unique(isodata['logAge'])

            for j, age in enumerate(lsstwfirst_ages):
                for band in lsstwfirst_bands:
                    sel = ((isodata['logAge']==age)&(isodata['label']<=maxlabel))
                    interpdict[tb[band]+'-%1.2f'%(age)+'-%1.4f'%(float(lsstwfirst_zs[i]))] = interp1d(isodata['Mini'][sel], isodata[band][sel], fill_value=99, bounds_error=False)

        interpdict['lsstwfirst_zs']=np.array(lsstwfirst_zs).astype(float)
        interpdict['lsstwfirst_ages']=lsstwfirst_ages



    if save==True:
        dill.dump(interpdict, open("/Users/hendel/projects/streamgaps/streampepper/SDSS2MASSLSSTWFIRST_iso_interp_dict_maxlabel%i.pkl"%maxlabel,'wb'))

    return interpdict


def cfht_g_iso(age, z, mass, isodict=None):
    sdss_g = isodict['sdss_g'+'-%1.2f'%(age)+'-%1.4f'%(float(z))](mass)
    sdss_r = isodict['sdss_r'+'-%1.2f'%(age)+'-%1.4f'%(float(z))](mass)
    return sdss_g - 0.185*(sdss_g - sdss_r)

def cfht_r_iso(age, z, mass, isodict=None):
    sdss_g = isodict['sdss_g'+'-%1.2f'%(age)+'-%1.4f'%(float(z))](mass)
    sdss_r = isodict['sdss_r'+'-%1.2f'%(age)+'-%1.4f'%(float(z))](mass)
    return sdss_r - 0.024*(sdss_g - sdss_r)


def get_mag(masses, ages, fehs, rs = None, bands=None, interpdict=None, verbose=False):
    #rs in kpc
    mags={}
    zs = 10**fehs*0.019

    if interpdict==None:
        interpdict = load_iso_interps(remake=False, save=False)
    print bands
    for band in bands:
        if ('cfht' in band):
            bands = bands+['sdss_g', 'sdss_r']
    bands = list(set(bands))
    bands.sort(key='cfht_r'.__eq__)
    bands.sort(key='cfht_g'.__eq__)

    for band in bands:
        print band
        mags[band] = np.zeros(len(masses))
        if (('sdss' in band) or ('2mass' in band) or ('cfht' in band)):
            tband = string.replace(band,'cfht','sdss')
            zs_to_use = get_closest((interpdict['sdss2mass_zs']), zs)
            ages_to_use = get_closest((interpdict['sdss2mass_ages']), ages)
            for ii in interpdict['sdss2mass_ages']:
                psel = (ages_to_use==ii)
                if np.sum(psel)>0:
                    for jj in interpdict['sdss2mass_zs']:
                        sel = ((ages_to_use==ii)&(np.round(zs_to_use,4)==jj))
                        if verbose==True: print ii, jj, np.sum(sel)
                        mags[band][sel] = interpdict[tband+'-%1.2f'%(ii)+'-%1.4f'%jj](masses[sel])
                        if band == 'cfht_g': mags[band] = mags['sdss_g'] - 0.185*(mags['sdss_g'] - mags['sdss_r'])
                        if band == 'cfht_r': mags[band] = mags['sdss_r'] - 0.024*(mags['sdss_g'] - mags['sdss_r'])

        elif (('wfirst' in band) or ('lsst' in band)):
            tband = string.replace(band,'lsst10','lsst')
            zs_to_use = get_closest((interpdict['lsstwfirst_zs']), zs)
            ages_to_use = get_closest(interpdict['lsstwfirst_ages'], ages)
            for ii in interpdict['lsstwfirst_ages']:
                psel = (ages_to_use==ii)
                if np.sum(psel)>0:
                    for jj in interpdict['lsstwfirst_zs']:
                        sel = ((ages_to_use==ii)&(np.round(zs_to_use,4)==jj))
                        if verbose==True: print ii, jj, np.sum(sel)
                        mags[band][sel] = interpdict[tband+'-%1.2f'%ii+'-%1.4f'%jj](masses[sel])
        else: print band, ' not set'
    for band in bands:
        if rs is not None:
            mags[band] = mags[band] + (5*np.log10(rs*1e3)-5)
    return mags

def get_closest(array, values):
    #`values` should be sorted
    sortindex = np.argsort(values)
    svalues = values[sortindex]

    #make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, svalues, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(svalues - array[np.maximum(idxs-1, 0)]) < np.fabs(svalues - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1

    #ii will return the indexes to the original order
    ii = values.argsort().argsort()

    return array[idxs[ii]]


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
    if survey == 'WFIRST':
        wndict = {'h':'H158','z':'Z087'}
        #1335 sec exposures, from primary survey
        mags, magerrs = scipy.genfromtxt('/Users/hendel/projects/wfirst/errormodel/'+wndict[filt]+'_photerr.txt', skip_header=1, unpack=True, usecols=(0,1))
        magerr = np.interp(mag, mags, magerrs)
        return magerr
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
        g, g_err, r, r_err = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/CFHT_photoerr.txt', skiprows=1, unpack=True)
        if filt == 'g':
            magerr = np.interp(mag, g, g_err)
        if filt == 'r':
            magerr = np.interp(mag, r, r_err)
        return np.sqrt(magerr**2 + calibration_err**2)
    if survey == 'SDSS':
        # this is DR9 photometry
        g, g_err, r, r_err = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/SDSS_photoerr.txt', skiprows=1, unpack=True)
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

def getMagLimit(filt, survey='LSST', maxerr=0.1):
    "A sophisticated calculation of LSST magntude limit"
    xgrid = np.linspace(15, 28, 1000)
    err = getMagErrVec(xgrid, filt, survey)
    xid = np.argmax(err * (err < maxerr))
    return xgrid[xid]

def load_errormodels():
    errormodels = {}
    errormodels['LSST']   = { 'g':interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), 'g', survey='LSST'), bounds_error=False, fill_value=99.),   'r':interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), 'r', survey='LSST'), bounds_error=False, fill_value=99.)}
    errormodels['LSST10'] = { 'g':interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), 'g', survey='LSST10'), bounds_error=False, fill_value=99.), 'r':interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), 'r', survey='LSST10'), bounds_error=False, fill_value=99.)}
    errormodels['CFHT']   = { 'g':interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), 'g', survey='CFHT'), bounds_error=False, fill_value=99.),   'r':interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), 'r', survey='CFHT'), bounds_error=False, fill_value=99.)}
    errormodels['SDSS']   = { 'g':interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), 'g', survey='SDSS'), bounds_error=False, fill_value=99.),   'r':interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), 'r', survey='SDSS'), bounds_error=False, fill_value=99.)}
    errormodels['CASTOR'] = { 'g':interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), 'g', survey='CASTOR'), bounds_error=False, fill_value=99.), 'u':interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), 'u', survey='CASTOR'), bounds_error=False, fill_value=99.)}
    errormodels['WFIRST'] = { 'z':interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), 'z', survey='WFIRST'), bounds_error=False, fill_value=99.), 'h':interp1d(np.linspace(14,28,100), getMagErrVec(np.linspace(14,28,100), 'h', survey='WFIRST'), bounds_error=False, fill_value=99.)}
    return errormodels

def gen_bg_counts_interp(dms = np.linspace(14,20,6), magerror_mods=[1.], surveys=['SDSS', 'CFHT', 'LSST', 'LSST10', 'CASTOR', 'WFIRST'], bands=['gr', 'gr', 'gr', 'gr', 'ug', 'zh'], minerr=0.003, maxerr=0.1, thresh=2, isodict=None, errormodels=None, bg_name=None, verbose=False):
    from scipy.interpolate import RegularGridInterpolator
    import ebf
    bgdict = {}

    if bg_name == None:
        bg_name = '/Users/hendel/modules/Galaxia/GalaxiaData/Examples/test_gap_mock.ebf'
    F = ebf.read(bg_name)
    center = (np.median(F['glat']), np.median(F['glon']))
    sel = np.sqrt((F['glat']-center[0])**2+(F['glon']-center[1])**2)<(1/np.sqrt(np.pi))

    if isodict ==None:
        isodict = load_iso_interps(remake=False)
    if errormodels==None:
        errormodels = load_errormodels()

    for i, survey in enumerate(surveys):
        tbands = [string.lower(survey+'_'+bands[i][0]),string.lower(survey+'_'+bands[i][1])]
        apmags = get_mag(F['smass'][sel], F['age'][sel], F['feh'][sel], rs =F['rad'][sel], bands=tbands, interpdict = isodict)
        maglim = getMagLimit(bands[i][0], survey=survey)
        maglims = np.linspace(maglim-5,maglim, 10) #turn this up in production
        #data = np.zeros((len(dms), len(magerror_mods), len(maglims)))
        data = np.zeros((len(dms), len(maglims)))
        for ii in np.arange(len(dms)):
            for jj in np.arange(len(magerror_mods)):
                for kk in np.arange(len(maglims)):
                    #data[ii,jj,kk] = bg_count_func(apmags, dms[ii], magerror_mods[jj], maglims[kk], survey=survey, band = bands[i], isodict=isodict, minerr=minerr, maxerr=maxerr, thresh=thresh)
                    data[ii,kk] = bg_count_func(apmags, dms[ii], magerror_mods[jj], maglims[kk], survey=survey, band = bands[i], isodict=isodict, minerr=minerr, maxerr=maxerr, thresh=thresh)
                    #if verbose==True: print survey, 'dm ', dms[ii], 'magerror ', magerror_mods[jj], 'maglim ', maglims[kk], 'n bg ', data[ii,jj,kk] 
                    if verbose==True: print survey, 'dm ', dms[ii], 'maglim ', maglims[kk], 'n bg ', data[ii,kk] 
        #bgdict[survey] = RegularGridInterpolator((np.array(dms), np.array(magerror_mods), np.array(maglims)), data)
        bgdict[survey] = RegularGridInterpolator((np.array(dms), np.array(maglims)), data)

    return bgdict


def gen_gal_counts_interp(dms = np.linspace(14,20,6), magerror_mods=[1.], surveys=['SDSS', 'CFHT', 'LSST', 'LSST10', 'CASTOR', 'WFIRST'], bands=['gr', 'gr', 'gr', 'gr', 'ug', 'zh'], minerr=0.003, maxerr=0.1, thresh=2, isodict=None, errormodels=None, verbose=False):
    from scipy.interpolate import RegularGridInterpolator
    import ebf
    bgdict = {}

    hlf = scipy.genfromtxt('/Users/hendel/data/hlsp_hlf_hst_60mas_goodss_v2.0_catalog.txt',names=True)
    area = ((max(hlf['dec'])-min(hlf['dec']))*u.deg)*((max(hlf['ra'])-min(hlf['ra']))*u.deg)

    if isodict ==None:
        isodict = load_iso_interps(remake=True)
    if errormodels==None:
        errormodels = load_errormodels()

    for i, survey in enumerate(surveys):
        tbands = [string.lower(survey+'_'+bands[i][0]),string.lower(survey+'_'+bands[i][1])]
        if bands[i]=='ug':
            sel = ((hlf['f_f336w']>0)&(hlf['f_f435w']>0)&(hlf['flux_radius']>0)&(hlf['flux_radius']<3.))
            apmags = {tbands[0]:(-2.5*np.log10(hlf['f_f336w'][sel])+24.67),tbands[1]:(-2.5*np.log10(hlf['f_f435w'][sel])+25.68)}
        if bands[i]=='gr':
            sel = ((hlf['f_f435w']>0)&(hlf['f_f606w']>0)&(hlf['flux_radius']>0)&(hlf['flux_radius']<7.))
            apmags = {tbands[0]:(-2.5*np.log10(hlf['f_f435w'][sel])+25.68), tbands[1]:(-2.5*np.log10(hlf['f_f606w'][sel])+26.51)}
        if bands[i]=='zh':
            sel = ((hlf['f_f160w']>0)&(hlf['f_f850lp']>0)&(hlf['flux_radius']>0)&(hlf['flux_radius']<3.))
            apmags = {'wfirst_h':(-2.5*np.log10(hlf['f_f160w'][sel])+25.94), 'wfirst_z':(-2.5*np.log10(hlf['f_f850lp'][sel])+24.87)}

        maglim = getMagLimit(bands[i][0], survey=survey)
        maglims = np.linspace(maglim-5,maglim, 10) #turn this up in production
        #data = np.zeros((len(dms), len(magerror_mods), len(maglims)))
        data = np.zeros((len(dms), len(maglims)))
        for ii in np.arange(len(dms)):
            for jj in np.arange(len(magerror_mods)):
                for kk in np.arange(len(maglims)):
                    #data[ii,jj,kk] = bg_count_func(apmags, dms[ii], magerror_mods[jj], maglims[kk], survey=survey, band = bands[i], isodict=isodict, minerr=minerr, maxerr=maxerr, thresh=thresh)/area.value
                    data[ii,kk] = bg_count_func(apmags, dms[ii], magerror_mods[jj], maglims[kk], survey=survey, band = bands[i], isodict=isodict, minerr=minerr, maxerr=maxerr, thresh=thresh)/area.value
                    #if verbose==True: print survey, 'dm ', dms[ii], 'magerror ', magerror_mods[jj], 'maglim ', maglims[kk], 'n bg ', data[ii,jj,kk] 
                    if verbose==True: print survey, 'dm ', dms[ii], 'maglim ', maglims[kk], 'n bg ', data[ii,kk] 
        #bgdict[survey] = RegularGridInterpolator((np.array(dms), np.array(magerror_mods), np.array(maglims)), data)
        bgdict[survey] = RegularGridInterpolator((np.array(dms), np.array(maglims)), data)
    return bgdict


def bg_count_func(mags, dm, magerror_mod, maglim, area=1., survey='SDSS', band='gr', isodict=None, minerr=0.003, maxerr=0.1, thresh=2,):
    if isodict ==None:
        isodict = load_iso_interps(remake=False)
    tbands = [string.lower(survey+'_'+band[0]),string.lower(survey+'_'+band[1])]
    #if survey =='WFIRST':
    #    g, r = mags['H158'], mags['Z087']
    g, r = mags[tbands[0]], mags[tbands[1]]

    if survey=='LSST10':
        gc_isochrone1 = isodict[string.replace(tbands[0],'lsst10','lsst')+'-10.06-0.0008']
        gc_isochrone2 = isodict[string.replace(tbands[1],'lsst10','lsst')+'-10.06-0.0008']
    elif survey=='WFIRST':
        wndict = {'h':'wfirst_h','z':'wfirst_z'}
        gc_isochrone1 = isodict[wndict[band[0]]+'-10.06-0.0008']
        gc_isochrone2 = isodict[wndict[band[1]]+'-10.06-0.0008']
    else:
        gc_isochrone1 = isodict[tbands[0]+'-10.06-0.0008']
        gc_isochrone2 = isodict[tbands[1]+'-10.06-0.0008']

    gcurve, rcurve = getIsoCurve(gc_isochrone1,gc_isochrone2)
    gcurve, rcurve = [_ + dm for _ in [gcurve, rcurve]]

    mincol, maxcol = -1., 2.
    minmag, maxmag = 17, 28
    colbin = 0.01
    magbin = 0.01

    colbins = np.arange(mincol, maxcol, colbin)
    magbins = np.arange(minmag, maxmag, magbin)

    grgrid, rgrid = np.mgrid[mincol:maxcol:colbin, minmag:maxmag:magbin]
    ggrid = grgrid + rgrid

    arr0 = np.array([(ggrid).flatten(), rgrid.flatten()]).T
    arr = np.array([gcurve, rcurve]).T
    tree = scipy.spatial.cKDTree(arr)
    D, xind = tree.query(arr0)

    gerr = getMagErrVec(ggrid.flatten(), band[0], survey).reshape(ggrid.shape)
    rerr = getMagErrVec(rgrid.flatten(), band[1], survey).reshape(rgrid.shape)

    maglim_g = maglim#getMagLimit('g', survey)
    maglim_r = maglim#getMagLimit('r', survey)

    errfactor=magerror_mod
    gerr = gerr*errfactor
    rerr = rerr*errfactor
    gerr, rerr = [np.maximum(_, minerr) for _ in [gerr, rerr]]

    dg = ggrid - gcurve[xind].reshape(ggrid.shape)
    dr = rgrid - rcurve[xind].reshape(rgrid.shape)

    mask = (np.abs(dg / gerr) < thresh) & (np.abs(dr / rerr) <
                                           thresh) & (rgrid < maglim_r) & (ggrid < maglim_g)

    colid = np.digitize(g - r, colbins) - 1
    magid = np.digitize(r, magbins) - 1
    xind = betw(colid, 0, grgrid.shape[0] - 1) & betw(magid, 0, grgrid.shape[1])
    xmask = np.zeros(len(g), dtype=bool)
    xmask[xind] = mask[colid[xind], magid[xind]]
    nbgstars = xmask.sum()
    bgdens = nbgstars / area

    return bgdens

def getIsoCurve(iso1, iso2, magstep=0.01):
    """
    Returns a (dense) list of points sampling along the isochrone

    Arguments:
    ---------
    iso1, iso2:
        interp1d instances from isodict
    magstep: float(optional)
        The step in magntidues along the isochrone
    Returns:
    -------
    gcurve,rcurve: Tuple of numpy arrays
        The tupe of arrays of magnitudes in g and r going along the isochrone
    """
    mini = iso1.x
    mag1 = iso1.y
    mag2 = iso2.y
    out1, out2 = [], []
    for i in range(len(mini) - 1):
        l_1, l_2, r_1, r_2 = mag1[i], mag2[i], mag1[i + 1], mag2[i + 1]
        maggap = max(abs(r_1 - l_1), abs(r_2 - l_2))
        if maggap < .2:
            npt = maggap / magstep + 2
            mag1grid = np.linspace(l_1, r_1, npt)
            mag2grid = np.linspace(l_2, r_2, npt)
            out1.append(mag1grid)
            out2.append(mag2grid)
        else: 
            npt = 2
            mag1grid = np.linspace(l_1, r_1, npt)
            mag2grid = np.linspace(l_2, r_2, npt)
            out1.append(mag1grid)
            out2.append(mag2grid)

    out1, out2 = [np.concatenate(_) for _ in [out1, out2]]
    return out1, out2

def betw(x, x1, x2): return (x >= x1) & (x <= x2)






"""
DEPRECIATED




def get_mag(masses, ages, fehs, rs = None, bands=None, interpdict=None, verbose=False):
    #rs in kpc
    mags={}
    zs = 10**fehs*0.019

    if interpdict==None:
        interpdict = load_iso_interps(remake=True, save=False, bands=bands)

    for band in bands:
        print band
        mags[band] = np.zeros(len(masses))
        if 'castor' in band:
            zs_to_use = get_closest((interpdict['castor_zs']), zs)
            ages_to_use = get_closest(np.log10(interpdict['castor_ages']), ages)
            for ii in np.log10(interpdict['castor_ages']):
                psel = (ages_to_use==ii)
                if np.sum(psel)>0:
                    for jj in interpdict['castor_zs']:
                        sel = ((ages_to_use==ii)&(np.round(zs_to_use,4)==jj))
                        if verbose==True: print ii, jj, np.sum(sel)
                        mags[band][sel] = interpdict[band+'-%1.2f'%(ii)+'-%1.4f'%jj](masses[sel])
        elif ('sdss' in band) or ('cfht' in band):
            zs_to_use = get_closest((interpdict['sdss_zs']), zs)
            ages_to_use = get_closest((interpdict['sdss_ages']), ages)
            for ii in interpdict['sdss_ages']:
                psel = (ages_to_use==ii)
                if np.sum(psel)>0:
                    for jj in interpdict['sdss_zs']:
                        sel = ((ages_to_use==ii)&(np.round(zs_to_use,4)==jj))
                        if verbose==True: print ii, jj, np.sum(sel)
                        mags[band][sel] = interpdict[string.replace(band,'cfht','sdss')+'-%1.2f'%(ii)+'-%1.4f'%jj](masses[sel])
        elif ('lsst' in band):
            zs_to_use = get_closest((interpdict['lsstwfirst_zs']), zs)
            ages_to_use = get_closest(interpdict['lsstwfirst_ages'], ages)
            for ii in interpdict['lsstwfirst_ages']:
                psel = (ages_to_use==ii)
                if np.sum(psel)>0:
                    for jj in interpdict['lsstwfirst_zs']:
                        sel = ((ages_to_use==ii)&(np.round(zs_to_use,4)==jj))
                        if verbose==True: print ii, jj, np.sum(sel)
                        mags[band][sel] = interpdict[string.replace(band,'lsst10','lsst')+'-%1.2f'%ii+'-%1.4f'%jj](masses[sel])
        else:
            zs_to_use = get_closest((interpdict['lsstwfirst_zs']), zs)
            ages_to_use = get_closest(interpdict['lsstwfirst_ages'], ages)
            wndict = {'wfirst_h':'H158','wfirst_z':'Z087'}
            for ii in interpdict['lsstwfirst_ages']:
                psel = (ages_to_use==ii)
                if np.sum(psel)>0:
                    for jj in interpdict['lsstwfirst_zs']:
                        sel = ((ages_to_use==ii)&(np.round(zs_to_use,4)==jj))
                        if verbose==True: print ii, jj, np.sum(sel)
                        mags[band][sel] = interpdict[wndict[band]+'-%1.2f'%ii+'-%1.4f'%jj](masses[sel])
        if rs is not None:
            mags[band] = mags[band] + (5*np.log10(rs*1e3)-5)
    return mags

"""