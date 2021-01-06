import numpy as np

import tqdm

import scipy.interpolate
from scipy import ndimage

#from numba import jit, prange

import h5py

import emcee

import os, datetime, fnmatch

import matplotlib
import matplotlib.pyplot as plt

from astroquery.mast import Catalogs
import astroquery.mast
import astroquery

import astropy
import astropy.time

import time as timemodu

import tdpy.mcmc
from tdpy.util import summgene
import tdpy.util
import tesstarg.util

from lion import main as lionmain


def read_psfntess(a, b, k, l):
    
    indxcams = range(1, 5)
    indxccds = range(1, 5)
    indxrows = np.array([1, 513, 1025, 1536, 2048])
    indxcols = np.array([45, 557, 1069, 1580, 2092])
    
    indxrowsthis = np.argmin(abs(k - indxrows))
    indxcolsthis = np.argmin(abs(l - indxcols))
    
    pathdata = os.environ['LYGOS_DATA_PATH'] + '/tesspsfn/'
    pathsubs = pathdata + 'tess_prf-master/cam%d_ccd%d/' % (a, b)
    if a == 2 and b == 4 or a > 2:
        path = pathsubs + 'tess2018243163601-prf-%d-%d-row%04d-col%04d.fits' % (a, b, indxrows[indxrowsthis], indxcols[indxcolsthis])
    else:
        path = pathsubs + 'tess2018243163600-prf-%d-%d-row%04d-col%04d.fits' % (a, b, indxrows[indxrowsthis], indxcols[indxcolsthis])
    listhdun = astropy.io.fits.open(path)
    listhdun.info()
    psfn = listhdun[0].data
    listhdun.close()
    
    return psfn


def plot_cntpwrap(gdat, cntp, o, strgsecc, typecntpscal, nameplotcntp, strgdataextn, \
                                                    boolresi=False, listindxpixlcolr=None, indxpcol=None, indxtimeanim=None, \
                                                                                time=None, lcur=None, \
                                                                                vmin=None, vmax=None, \
                                                                                listtime=None, listtimelabl=None, \
                                                                                ):
    
    if time is None:
        time = [None for t in gdat.indxtime]

    if listtimelabl is None:
        listtimelabl = [None for t in gdat.indxtime]

    if indxtimeanim is None:
        indxtimeanim = np.arange(cntp.shape[2])
    
    listpath = []
    for tt, t in enumerate(indxtimeanim):
        pathcntp = retr_pathvisu(gdat, nameplotcntp, strgdataextn, indxpcol=indxpcol, indxtimeplot=t)

        # make title
        strgtitl = gdat.strgtitlcntpplot
        if listtimelabl[t] is not None:
            strgtitl += ', %s' % listtimelabl[t]
        
        path = plot_cntp(gdat, cntp[:, :, t], o, typecntpscal, nameplotcntp, strgdataextn, \
                                                strgtitl=strgtitl, boolresi=boolresi, listindxpixlcolr=listindxpixlcolr, \
                                                                                            timelabl=listtimelabl[t], thistime=time[t], \
                                                                                                vmin=vmin, vmax=vmax, lcur=lcur, time=time)
        
        listpath.append(path)

    return listpath


def plot_cntp(gdat, cntp, o, typecntpscal, nameplotcntp, strgdataextn, indxpcol=None, \
                                            cbar='Greys_r', strgtitl='', boolresi=False, xposoffs=None, yposoffs=None, strgextn='', \
                                                                                           lcur=None, boolanno=True, \
                                                                                           time=None, timelabl=None, thistime=None, \
                                                                                           vmin=None, vmax=None, listindxpixlcolr=None):
    
    if typecntpscal == 'asnh':
        cntp = np.arcsinh(cntp)
    
    if strgextn != '':
        strgextn = '_' + strgextn

    if boolresi:
        cbar = 'PuOr'
    
    if lcur is None:
        figrsize = (6, 6)
        figr, axis = plt.subplots(figsize=figrsize)
        axis =[axis]
    else:
        figrsize = (6, 9)
        figr, axis = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2, 0.7]}, figsize=figrsize)
    
    imag = axis[0].imshow(cntp, origin='lower', interpolation='nearest', cmap=cbar, vmin=vmin, vmax=vmax)
    
    # plot the fitting point sources
    if boolanno:
    
        axis[0].scatter(gdat.catlrefrfilt[0]['xpos'], gdat.catlrefrfilt[0]['ypos'], alpha=1., s=gdat.sizeplotsour, color='r')
        for indxtemp in gdat.indxsourcatlrefr:
            axis[0].text(gdat.catlrefrfilt[0]['xpos'][indxtemp], gdat.catlrefrfilt[0]['ypos'][indxtemp], \
                                                                        '%s' % ('TIC %s' % gdat.catlrefrfilt[0]['tici'][indxtemp]), color='r')
        
        xposfitt = np.copy(gdat.xposfitt)
        yposfitt = np.copy(gdat.yposfitt)
        if xposoffs is not None:
            # add the positional offset, if any
            xposfitt += xposoffs
            yposfitt += yposoffs
        # target
        axis[0].scatter(xposfitt[0], yposfitt[0], alpha=1., color='b', s=2*gdat.sizeplotsour, marker='x')
        # neighbors
        axis[0].scatter(xposfitt[1:], yposfitt[1:], alpha=1., s=gdat.sizeplotsour, color='b', marker='x')
        for k in gdat.indxstar:
            axis[0].text(xposfitt[k] + 0.25, yposfitt[k] + 0.25, '%d' % k, color='b')
        axis[0].set_title(strgtitl) 
    
    # highlight TESS pixels
    for k in range(gdat.numbside+1):
        axis[0].axvline(k - 0.5, ls='--', alpha=0.3, color='y')
    for k in range(gdat.numbside+1):
        axis[0].axhline(k - 0.5, ls='--', alpha=0.3, color='y')
    
    if listindxpixlcolr is not None:
        temp = np.zeros_like(cntp).flatten()
        for indxpixlcolr in listindxpixlcolr:
            rect = patches.Rectangle((indxpixlcolr[1], indxpixlcolr[0]),1,1,linewidth=1,edgecolor='r',facecolor='none')
            axis.add_patch(rect)
    
    cbar = figr.colorbar(imag, fraction=0.046, pad=0.04, ax=axis[0]) 
    if typecntpscal == 'asnh':
        tick = cbar.ax.get_yticks()
        tick = np.sinh(tick)
        labl = ['%d' % tick[k] for k in range(len(tick))]
        cbar.ax.set_yticklabels(labl)
    
    if lcur is not None:
        axis[1].plot(time, lcur, color='black', ls='', marker='o', markersize=1)
        axis[1].set_xlabel('Time [BJD]') 
        axis[1].set_ylabel('Relative flux') 
        axis[1].axvline(thistime)
    
    path = retr_pathvisu(gdat, nameplotcntp, strgdataextn, typecntpscal=typecntpscal, indxpcol=indxpcol)
    print('Writing to %s...' % path)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
    return path


def plot_lcur(gdat, lcurmodl, stdvlcurmodl, k, indxtsecplot, strgsecc, booltpxf, strgoffs, strgdataextn, timeedge=None, listmodeplot=[0], \
                    strgextn='', indxtimelimt=None, indxtzom=None, boolerrr=False):
    
    if k == 0:
        lablcomp = ', Target source' % k
    elif k == gdat.numbcomp - 1:
        lablcomp = ', Background'
    else:
        lablcomp = ', Neighbor Source %d' % k

    timedatatemp = np.copy(gdat.timedata[indxtsecplot])
    timerefrtemp = [[] for q in gdat.indxrefrlcur[indxtsecplot]] 
    for q in gdat.indxrefrlcur[indxtsecplot]:
        timerefrtemp[q] = gdat.refrtime[indxtsecplot][q]
    
    if indxtimelimt is not None:
        timedatatemp = timedatatemp[indxtimelimt]
        stdvlcurmodl = stdvlcurmodl[indxtimelimt]
        lcurmodl = lcurmodl[indxtimelimt]

    lablxaxi = 'Time [BJD]'
    
    for a in listmodeplot:
        
        if a == 1 and k > 0:
            continue
            
        if a == 0:
            figr, axis = plt.subplots(figsize=(12, 4))
            axis = [axis]
            axis[0].set_xlabel(lablxaxi)
                
        if a == 1:
            figr, axis = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2, 0.7]}, figsize=(12, 8))
            axis[1].set_xlabel(lablxaxi)
        
        axis[0].set_ylabel('Relative flux')
        if a == 1:
            axis[0].set_ylabel('Residual')
        
        if boolerrr:
            yerr = stdvlcurmodl
        else:
            yerr = None
        temp, listcaps, temp = axis[0].errorbar(timedatatemp, lcurmodl, yerr=yerr, color='b', ls='', markersize=2, \
                                                                                marker='.', lw=3, alpha=0.3, label='Pandora')
        for caps in listcaps:
            caps.set_markeredgewidth(3)
        
        if timeedge is not None:
            for timeedgetemp in timeedge[1:-1]:
                axis[0].axvline(timeedgetemp, ls='--', color='grey', alpha=0.5)
        if a == 1 and k == 0:
            for q in gdat.indxrefrlcur[indxtsecplot]:
                if boolerrr:
                    yerr = gdat.stdvrefrrflx[indxtsecplot][q]
                else:
                    yerr = None
                print('indxtsecplot')
                print(indxtsecplot)
                print('q')
                print(q)
                print('gdat.refrrflx[indxtsecplot][q]')
                summgene(gdat.refrrflx[indxtsecplot][q])
                temp, listcaps, temp = axis[0].errorbar(timerefrtemp, gdat.refrrflx[indxtsecplot][q], \
                                                    yerr=yerr, color=gdat.colrrefrlcur[indxtsecplot][q], ls='', markersize=2, \
                                                                            marker='.', lw=3, alpha=0.3, label=gdat.lablrefrlcur[indxtsecplot][q])
                for caps in listcaps:
                    caps.set_markeredgewidth(3)
            
            ## residual
            for q in gdat.indxrefrlcur[indxtsecplot]:
                if lcurmodl.size == gdat.refrrflx[indxtsecplot][q][:gdat.numbtimecutt].size:
                    print('q')
                    print(q)
                    print('lcurmodl')
                    summgene(lcurmodl)
                    print('gdat.refrrflx[indxtsecplot][q][:gdat.numbtimecutt]')
                    summgene(gdat.refrrflx[indxtsecplot][q][:gdat.numbtimecutt])
                    ydat = lcurmodl - gdat.refrrflx[indxtsecplot][q][:gdat.numbtimecutt]
                    if boolerrr:
                        yerr = None
                    else:
                        yerr = None
                    axis[1].errorbar(timedatatemp[:gdat.numbtimecutt], ydat, yerr=yerr, label=gdat.lablrefrlcur[indxtsecplot][q], \
                                                        color='k', ls='', marker='.', markersize=2, alpha=0.3)
        axis[0].set_title(gdat.labltarg + lablcomp)
        if gdat.listtimeplotline is not None:
            for timeplotline in gdat.listtimeplotline:
                axis[0].axvline(timeplotline, ls='--')
        
        if gdat.numbrefrlcur[indxtsecplot] > 0:
            axis[0].legend()

        if indxtzom is not None:
            axis[a].set_xlim(gdat.listlimttimetzom[indxtzom])
        
        #plt.tight_layout()
        nameplot = 'rflx_s%03d_mod%d' % (k, a)
        path = retr_pathvisu(gdat, nameplot, strgdataextn, indxtzom=indxtzom)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()


def retr_pathvisu(gdat, \
                    nameplot, \
                    # data
                    strgdataextn, \
                    
                    # rflx 
                    # temporal zoom
                    indxtzom=None, \

                    # cntp
                    typecntpscal=None, \
                    indxpcol=None, \
                    boolanim=False, \
                    indxtimeplot=None, \
                    ):
    
    if typecntpscal is None or typecntpscal == 'self':
        strgscal = ''
    else:
        strgscal = '_' + typecntpscal

    if indxtzom is None:
        strgtzom = ''
    else:
        strgtzom = '_zom%d' % indxtzom
    
    if indxpcol is None:
        strgpcol = ''
    else:
        strgpcol = '_col%d' % indxpcol
    
    if boolanim:
        strgplotextn = 'gif'
    else:
        strgplotextn = gdat.strgplotextn
    
    if indxtimeplot is None:
        strgtime = ''
    else:
        strgtime = '_%08d' % t
    
    pathvisu = gdat.pathimagobjt + '%s%s%s%s%s%s.%s' % \
               (nameplot, strgdataextn, strgscal, strgtzom, strgpcol, strgtime, strgplotextn)
    
    return pathvisu


def retr_cntpmodl(gdat, xpos, ypos, flux, cntpback, coef=None, verbtype=1):
    
    if gdat.psfntype == 'lion':
        gdat.gdatlion.numbtime = flux.shape[0]
        cntpmodl = lionmain.eval_modl(gdat.gdatlion, xpos, ypos, flux[None, :, :], cntpback[None, None, None, :])[0, :, :, :]
    else:
        cntpmodl = np.zeros_like((gdat.numbside, gdat.numbside, gdat.numbtime)) + cntpback
        for k in range(xpos.size):
            deltxpos = gdat.xpos - xpos[k]
            deltypos = gdat.ypos - ypos[k]
            if gdat.psfnshaptype == 'gaus':
                psfnsour = np.exp(-(deltxpos / 0.7)**2 - (deltypos / 0.7)**2)
            if gdat.psfnshaptype.startswith('gfre'):
                psfnsour = np.exp(-0.5 * (deltxpos / coef[0])**2 - 0.5 * (deltypos / coef[1])**2)
            if gdat.psfnshaptype == 'pfre':
                psfnsour = coef[0] * deltxpos + coef[1] * deltypos + coef[2] * deltxpos * deltypos + \
                                coef[3] * deltxpos**2 + coef[4] * deltypos**2 + coef[5] * deltxpos**2 * deltypos + coef[6] * deltypos**2 * deltxpos + \
                                coef[7] * deltxpos**3 + coef[8] * deltypos**3 + coef[9] * np.exp(-deltxpos**2 / coef[10] - deltypos**2 / coef[11])
            
            cntpmodl += flux[k] * psfnsour
            
        #for t in gdat.indxtime:
    
    return cntpmodl


def retr_llik(gdat, para):
    
    coef = para[:gdat.numbcoef]
    cntpback = para[gdat.numbcoef]

    if gdat.psfntype == 'lion':
        coef = None
    else:
        if gdat.psfnshaptype == 'gfreffix':
            flux = gdat.cntsfitt * para[gdat.numbcoef+1]
        elif gdat.psfnshaptype == 'gfrefinf':
            flux = gdat.cntsfitt * para[gdat.numbcoef+1] * para[gdat.numbcoef+2:]
        elif gdat.psfnshaptype == 'gfreffre':
            flux = gdat.cntsfitt * para[gdat.numbcoef+1:]
        else:
            flux = para[gdat.numbcoef+1:]
    cntpmodl = retr_cntpmodl(gdat, gdat.xposfitt, gdat.yposfitt, flux, cntpback, coef)
    
    chi2 = np.sum((gdat.cntpdatatmed - cntpmodl)**2 / gdat.cntpdatatmed)
    llik = -0.5 * chi2
    
    return llik


#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def retr_mlikflux(cntpdata, matrdesi, vari):
    
    varitdim = np.diag(vari.flatten())
    covafittflux = np.linalg.inv(np.matmul(np.matmul(matrdesi.T, np.linalg.inv(varitdim)), matrdesi))
    mlikfittflux = np.matmul(np.matmul(np.matmul(covafittflux, matrdesi.T), np.linalg.inv(varitdim)), cntpdata.flatten())
    
    return mlikfittflux, covafittflux


def retr_strgdataextn(gdat, strgsecc, strgoffs, typecade):
    
    if strgoffs == 'of11':
        strgoffstemp = ''
    else:
        strgoffstemp = '_' + strgoffs
    strgdataextn = '_%s_%s%s_%s' % (gdat.strgcnfg, strgsecc, strgoffstemp, typecade)

    return strgdataextn


def retr_strgoffs(gdat, x, y):
    
    strgoffs = 'of%d%d' % (x, y)

    return strgoffs


def setp_cntp(gdat, strg, typecntpscal):
    
    cntp = getattr(gdat, 'cntp' + strg)
    
    cntptmed = np.nanmedian(cntp, axis=-1)
    setattr(gdat, 'cntp' + strg + 'tmed', cntptmed)
    vmin = np.nanpercentile(cntp, 0)
    vmax = np.nanpercentile(cntp, 100)
    if strg == 'resi':
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    
    if typecntpscal == 'asnh':
        vmin = np.arcsinh(vmin)
        vmax = np.arcsinh(vmax)
    setattr(gdat, 'vmincntp' + strg + typecntpscal, vmin)
    setattr(gdat, 'vmaxcntp' + strg + typecntpscal, vmax)
    

def init( \

         # data
         ## type of data: 'mock', 'obsd'
         typedata='obsd', \
         
         ## mock data
         truesourtype='dwar', \
         ### number of time bins
         numbtime=None, \
         ### Boolean flag
         boolcenttran=False, \
        
         # target
         ## a string to be used to search MAST for the target
         strgmast=None, \
         ## TIC ID of the object of interest
         ticitarg=None, \
         ## RA of the object of interest
         rasctarg=None, \
         ## DEC of the object of interest
         decltarg=None, \

         ## number of pixels on a side to cut out
         numbside=11, \
        
         ## base path
         pathbase=None, \

         ## mask
         ### Boolean flag to put a cut on quality flag
         boolcuttqual=True, \
         ### masking region
         epocmask=None, \
         perimask=None, \
         duramask=None, \
    
         # string indicating the cluster
         strgclus=None, \

         # visualization
         ## Boolean flag to make relative flux plots
         boolplotrflx=False, \
         ## Boolean flag to make image plots
         boolplotcntp=False, \
         ## Boolean flag to plot the quaternions
         boolplotquat=False, \
         ## Boolean flag to make an animation
         boolanim=False, \
         ## Boolean flag to include all time bins in the animation
         boolanimframtotl=False, \
        
         ## Boolean flag to plot the histogram of the number of counts
         boolplothhistcntp=False, \

         # plot extensions
         strgplotextn='png', \

        
         # selected TESS sectors
         listtsecsele=None, \

         strgbase=None, \
         
         # diagnostics
         booldiagmode=True, \

         # model
         ## factor by which to rebin the data along time
         facttimerebn=1., \
         ## maximum number stars in the fit
         maxmnumbstar=None, \
        
         # maximum delta magnitude of neighbor sources to be included in the model
         maxmdmag=4., \
        
         ## PSF evaluation
         psfntype='lion', \
         psfnshaptype='gfrefinf', \

         catlextr=None, \
         lablcatlextr=None, \
    
         # Boolean flag to repeat the fit, putting the target to offset locations
         boolfittoffs=False, \

         ## post-process
         ## baseline-detrending
         boolbdtr=False, \
         bdtrtype='spln', \
         booladdddiscbdtr=False, \
         
         listpixlaper=None, \

         # epoch for correcting the RA and DEC for proper motion
         epocpmot=None, \
         ## RA proper motion, used when RA and DEC are provided
         pmratarg=None, \
         ## DEC proper motion, used when RA and DEC are provided
         pmdetarg=None, \
         
         ## list of limits for temporal zoom
         listlimttimetzom=None, \

         ## the time to indicate on the plots with a vertical line
         listtimeplotline=None, \
         
         # a string that will appear in the plots to label the target, which can be anything the user wants
         labltarg=None, \
         
         # a string that will be used to name output files for this target
         strgtarg=None, \
        
         # image color scale
         listtypecntpscal=['self', 'asnh'], \
         
        ):
   
    # start the timer
    timeinittotl = timemodu.time()
    
    # construct global object
    gdat = tdpy.util.gdatstrt()
    
    # copy unnamed inputs to the global object
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print('lygos initialized at %s...' % gdat.strgtimestmp)
   
    # determine the target
    if gdat.ticitarg is not None and (gdat.rasctarg is not None or gdat.decltarg is not None) or \
                gdat.ticitarg is not None and gdat.strgmast is not None or \
                (gdat.rasctarg is not None or gdat.decltarg is not None) and gdat.strgmast is not None:
        raise Exception('Either a TIC ID (ticitarg) or RA&DEC (rasctarg and decltarg) or a string to be searched on MAST (strgmast) \
                                                                                                         should be provided; not more than one of these.')
    if gdat.ticitarg is None and (gdat.rasctarg is None or gdat.decltarg is None) and gdat.strgmast is None:
        raise Exception('Either a TIC ID (ticitarg) or RA&DEC (rasctarg and decltarg) or a string to be searched on MAST (strgmast) should be provided.')
    if gdat.ticitarg is not None:
        strgtargtype = 'tici'
        print('A TIC ID is provided as input.')
        catalogData = astroquery.mast.Catalogs.query_object('TIC ' + str(gdat.ticitarg), radius='200s', catalog="TIC")
        gdat.rasctarg = catalogData[0]['ra']
        gdat.decltarg = catalogData[0]['dec']
        tmagtarg = catalogData[0]['Tmag']
        print('gdat.rasctarg')
        print(gdat.rasctarg)
        print('gdat.decltarg')
        print(gdat.decltarg)
        print('TIC ID: %d' % gdat.ticitarg)
    else:
        if gdat.rasctarg is not None and gdat.decltarg is not None:
            strgtargtype = 'posi'
            print('RA and DEC are provided as input.')
            catalogData = astroquery.mast.Catalogs.query_region('%g %g' % (gdat.rasctarg, gdat.decltarg), radius='200s', catalog="TIC")
            print('Found %d TIC sources within 200 as.' % len(catalogData))
        else:
            strgtargtype = 'mast'
            print('A MAST keyword is provided as input: %s' % gdat.strgmast)
            catalogData = astroquery.mast.Catalogs.query_region(gdat.strgmast, radius='200s', catalog="TIC")
            print('Found %d TIC sources within 200 as.' % len(catalogData))
            gdat.rasctarg = catalogData[0]['ra']
            gdat.decltarg = catalogData[0]['dec']
            tmagtarg = catalogData[0]['Tmag']
            gdat.ticitarg = int(catalogData[0]['ID'])
    
    print('gdat.ticitarg')
    print(gdat.ticitarg)
    print('gdat.strgmast')
    print(gdat.strgmast)
    print('gdat.rasctarg')
    print(gdat.rasctarg)
    print('gdat.decltarg')
    print(gdat.decltarg)
    
    if gdat.labltarg is None:
        if gdat.strgmast is not None:
            gdat.labltarg = gdat.strgmast
        else:
            gdat.labltarg = 'TIC %d' % gdat.ticitarg
    
    if gdat.strgtarg is None:
        gdat.strgtarg = ''.join(gdat.labltarg.split(' '))
    
    print('Target label: %s' % gdat.labltarg) 
    print('Output folder name: %s' % gdat.strgtarg) 
    print('RA and DEC: %g %g' % (gdat.rasctarg, gdat.decltarg))
    if strgtargtype == 'tici' or strgtargtype == 'mast':
        print('Tmag: %g' % tmagtarg)
   
    print('PSF model: %s' % gdat.psfntype)
    if gdat.psfntype == 'ontf':
        print('PSF model shape:')
        print(gdat.psfnshaptype)
    # paths
    if gdat.pathbase is None:
        gdat.pathbase = os.environ['LYGOS_DATA_PATH'] + '/'
    
    gdat.pathimag = gdat.pathbase + 'imag/'
    gdat.pathdata = gdat.pathbase + 'data/'
    
    if gdat.strgclus is None:
        gdat.strgclus = ''
    else:
        gdat.strgclus += '/'
    
    if gdat.strgbase is None:
        gdat.pathobjt = gdat.pathbase + '%s%s/' % (gdat.strgclus, gdat.strgtarg)
    else:
        gdat.pathobjt = gdat.pathbase + '%s%s/%s/' % (gdat.strgclus, gdat.strgbase, gdat.strgtarg)
    gdat.pathdataobjt = gdat.pathobjt + 'data/'
    gdat.pathimagobjt = gdat.pathobjt + 'imag/'
    gdat.pathclus = gdat.pathbase + '%s' % gdat.strgclus
    gdat.pathdataclus = gdat.pathclus + 'data/'
    gdat.pathimagclus = gdat.pathclus + 'imag/'
    os.system('mkdir -p %s' % gdat.pathimag)
    os.system('mkdir -p %s' % gdat.pathdata)
    os.system('mkdir -p %s' % gdat.pathimagobjt)
    os.system('mkdir -p %s' % gdat.pathdataobjt)
    os.system('mkdir -p %s' % gdat.pathimagclus)
    os.system('mkdir -p %s' % gdat.pathdataclus)
    
    # create a separate folder to place the PSF fit output
    if gdat.psfntype == 'ontf':
        gdat.pathimagobjtpsfn = gdat.pathimagobjt + 'psfn/'
        os.system('mkdir -p %s' % gdat.pathimagobjtpsfn)
   
    # fix the seed
    np.random.seed(0)
    
    # exposure time
    gdat.timeexpo = 1440. # [sec]
    
    print('Number of pixels on a side: %d' % gdat.numbside)

    if gdat.strgtarg is None:
        gdat.strgtarg = '%016d' % gdat.ticitarg

    # construct the string describing the data configuration
    if gdat.typedata == 'obsd':
        strgtypedata = ''
    else:
        strgtypedata = '_' + gdat.typedata
    if gdat.numbside == 11:
        strgnumbside = ''
    else:
        strgnumbside = '_n%03d' % gdat.numbside
    if gdat.facttimerebn != 1:
        strgrebn = '_b%03d' % gdat.facttimerebn
    else:
        strgrebn = ''
    gdat.strgcnfg = '%s%s%s%s' % (gdat.strgtarg, strgtypedata, strgnumbside, strgrebn)
   
    print('Reading the data...')
    # get data
    if gdat.typedata != 'mock':
        
        # check all available TESS (FFI) cutout data 
        objtskyy = astropy.coordinates.SkyCoord(gdat.rasctarg, gdat.decltarg, unit="deg")
        listhdundata = astroquery.mast.Tesscut.get_cutouts(objtskyy, gdat.numbside)
        
        ## parse cutout HDUs
        gdat.listtsec = []
        gdat.listtcam = []
        gdat.listtccd = []
        listobjtwcss = []
        for hdundata in listhdundata:
            gdat.listtsec.append(hdundata[0].header['SECTOR'])
            gdat.listtcam.append(hdundata[0].header['CAMERA'])
            gdat.listtccd.append(hdundata[0].header['CCD'])
            listobjtwcss.append(astropy.wcs.WCS(hdundata[2].header))
        
        print('gdat.listtsec')
        print(gdat.listtsec)
        print('gdat.listtcam')
        print(gdat.listtcam)
        print('gdat.listtccd')
        print(gdat.listtccd)
        
        ## all sectors for which TESS data are available
        gdat.numbtsec = len(gdat.listtsec)
        gdat.indxtsec = np.arange(gdat.numbtsec)

        # get the list of sectors for which TESS SPOC data are available
        print('Retrieving the list of available TESS sectors for which there is SPOC data...')
        # list of TESS sectors for which SPOC data are available
        gdat.listtsecspoc = []
        # get observation tables
        
        if not (gdat.ticitarg is None and gdat.strgmast is None):
            listtablobsv = tesstarg.util.retr_listtablobsv(gdat.strgmast)
            numbtabl = len(listtablobsv)
            indxtabl = np.arange(numbtabl)
            listlistproddata = [[] for k in indxtabl]
            listlistproddatalcur = [[] for k in indxtabl]
            listlistproddatatpxf = [[] for k in indxtabl]
            for k, tablobsv in enumerate(listtablobsv):
                # tablobsv[target_name] is TIC ID
                listlistproddata[k] = astroquery.mast.Observations.get_product_list(tablobsv)
                listlistproddatalcur[k] = astroquery.mast.Observations.filter_products(listlistproddata[k], description='Light curves')
                listlistproddatatpxf[k] = astroquery.mast.Observations.filter_products(listlistproddata[k], description='Target pixel files')
                for a in range(len(listlistproddatatpxf[k])):
                    tsec = int(listlistproddatatpxf[k][a]['obs_id'].split('-')[1][1:])
                    gdat.listtsecspoc.append(tsec) 
        print('gdat.listtsecspoc')
        print(gdat.listtsecspoc)

        gdat.listtsecspoc = np.array(gdat.listtsecspoc)
        gdat.listtsecspoc = np.sort(gdat.listtsecspoc)
        
        # determine whether sectors have 2-minute cadence data
        gdat.booltpxf = tesstarg.util.retr_booltpxf(gdat.listtsec, gdat.listtsecspoc)
        
        gdat.typecade = np.zeros_like(gdat.booltpxf, dtype=object)
        gdat.typecade[:] = '30mn'
        gdat.typecade[np.where(gdat.booltpxf)] = '2min'
        print('gdat.booltpxf')
        print(gdat.booltpxf)

        if len(gdat.listtsecspoc) > 0:
            
            # select sector
            if gdat.listtsecsele is None:
                gdat.listtsecsele = gdat.listtsec
            
            #listproddatatemp = []
            #for k, proddata in enumerate(listproddata):
            #    if k in indxprodsele:
            #        listproddatatemp.append(proddata)
            #listproddata = listproddatatemp
            
            #pathmasttess = gdat.pathdataobjt + 'mastDownload/TESS/'
            listpathdownlcur = []
            listpathdowntpxf = []
            for k in indxtabl:
                if len(listlistproddata[k]) > 0:
                    tsec = int(listlistproddata[k][0]['obs_id'].split('-')[1][1:])
                    if tsec in gdat.listtsecsele:
                        #if listlistproddata[k][a]['description'] == 'Light curves' and tsec in indxtsecsele:
                        manifest = astroquery.mast.Observations.download_products(listlistproddatalcur[k], download_dir=gdat.pathdataobjt)
                        listpathdownlcur.append(manifest['Local Path'][0])
                        manifest = astroquery.mast.Observations.download_products(listlistproddatatpxf[k], download_dir=gdat.pathdataobjt)
                        listpathdowntpxf.append(manifest['Local Path'][0])

            ## make sure the list of paths to sector files are time-sorted
            listpathdownlcur.sort()
            listpathdowntpxf.sort()
            
                    #indxsectfile = np.where(gdat.listtsec == hdundata[0].header['SECTOR'])[0][0]
            #if typedataspoc == 'SAP' or typedataspoc == 'PDC':
            #    for namefold in os.listdir(pathmasttess):
            #        
            #        # eliminate those 2-min data for which 20-sec data already exist
            #        if boolfastonly and namefold.endswith('-s'):
            #            if os.path.exists(pathmasttess + namefold[:-2] + '-a_fast'):
            #                continue    
            #        
            #        if namefold.endswith('-s') or namefold.endswith('-a_fast'):
            #            pathlcurinte = pathmasttess + namefold + '/'
            #            pathlcur = pathlcurinte + fnmatch.filter(os.listdir(pathlcurinte), '*lc.fits')[0]
            #            listpathlcur.append(pathlcur)
            #
            
            if gdat.numbtsec == 0:
                print('No data have been retrieved.' % (gdat.numbtsec))
            else:
                if gdat.numbtsec == 1:
                    strgtemp = ''
                else:
                    strgtemp = 's'
                print('%d sector%s of data retrieved.' % (gdat.numbtsec, strgtemp))
        
    gdat.dictoutp = dict()
    gdat.dictoutp['listtsec'] = gdat.listtsec
    gdat.dictoutp['listtcam'] = gdat.listtcam
    gdat.dictoutp['listtccd'] = gdat.listtccd
    if len(gdat.listtsec) != len(gdat.listtcam):
        raise Exception('')

    print('Found %d sectors of data.' % gdat.numbtsec)

    # get reference catalog
    gdat.lablrefrcatl = ['TIC']
    if gdat.catlextr is not None:
        gdat.lablrefrcatl.extend(gdat.lablcatlextr)

    gdat.numbrefrcatl = len(gdat.lablrefrcatl)
    gdat.indxrefrcatl = np.arange(gdat.numbrefrcatl)
    
    # check for an earlier lygos run
    for o in gdat.indxtsec:

        strgsecc = retr_strgsecc(gdat, o)
        strgoffs = retr_strgoffs(gdat, 1, 1)
        strgdataextn = retr_strgdataextn(gdat, strgsecc, strgoffs, gdat.typecade[o])
        pathsaverflxtarg = gdat.pathdataobjt + 'rflxtarg' + strgdataextn + '.csv'
        gdat.dictoutp['pathsaverflxtargsc%02d' % gdat.listtsec[o]] = pathsaverflxtarg
        if os.path.exists(pathsaverflxtarg):
            print('Run previously completed...')
            gdat.listarry = [[] for o in gdat.indxtsec]
            for o in gdat.indxtsec:
                gdat.listarry[o] = np.loadtxt(pathsaverflxtarg)
            gdat.dictoutp['listarry'] = gdat.listarry
        
    # read TESS PSF
    pathpsfn = gdat.pathdata + 'tesspsfn/listpsfn.h5'
    print('Reading from %s...' % pathpsfn)
    objth5py = h5py.File(pathpsfn, 'r')
    listpsfn = objth5py.get('listpsfn')
    cntppsfntess = np.array(listpsfn)[0, 0, 0, 0, :, :]
    cntppsfntess /= np.sum(cntppsfntess)
    objth5py.close()
        
    #pathpsfn = gdat.pathdata + 'tesspsfn.txt'
    #if not os.path.exists(pathpsfn):
    #    cmnd = 'tsig-psf --resolution 11 --id 4564 --show-contents > %s' % pathpsfn
    #    os.system(cmnd)
    ## read
    #cntppsfnusam = np.loadtxt(pathpsfn, delimiter=',')
    #cntppsfnusam = ndimage.convolve(cntppsfnusam, np.ones((11, 11)) / 121., mode='constant', cval=0.) * 121.
    #cntppsfnusam = cntppsfnusam[None, :, :]
        
    # setup Lion
    if gdat.psfntype == 'lion':
        gdat.gdatlion = tdpy.util.gdatstrt()
        gdat.gdatlion.verbtype = 1
        gdat.gdatlion.boolspre = False
        gdat.gdatlion.sizeimag = [gdat.numbside, gdat.numbside]
        gdat.gdatlion.sizeregi = gdat.numbside
        gdat.gdatlion.numbregixaxi = gdat.numbside
        gdat.gdatlion.numbregiyaxi = gdat.numbside
        gdat.gdatlion.numbsidepsfn = 17
        gdat.gdatlion.diagmode = True
        gdat.gdatlion.numbener = 1
        gdat.gdatlion.indxener = [0]

        gdat.gdatlion.boolbili = False
        gdat.gdatlion.numbsidepsfnusam = 13 * 9
        gdat.gdatlion.factusam = 9
        gdat.gdatlion.marg = 0
        gdat.gdatlion.cntppsfnusam = cntppsfntess
        gdat.gdatlion.cntppsfnusam = ndimage.convolve(gdat.gdatlion.cntppsfnusam, np.ones((9, 9)), mode='constant', cval=0.)
        gdat.gdatlion.cntppsfnusam /= np.sum(gdat.gdatlion.cntppsfnusam)
        gdat.gdatlion.cntppsfnusam *= 81
        gdat.gdatlion.cntppsfnusam = gdat.gdatlion.cntppsfnusam[None, :, :]
        gdat.gdatlion.boolplotsave = False
        lionmain.retr_coefspix(gdat.gdatlion)

    
    gdat.catlrefr = [{}]
    gdat.catlrefrfilt = [{}]
    if gdat.catlextr is not None:
        gdat.numbcatlextr = len(gdat.catlextr)
        gdat.indxcatlextr = np.arange(gdat.numbcatlextr)
        for k in gdat.indxcatlextr:
            gdat.catlrefr.append({})
            gdat.catlrefrfilt.append({})
    
    print('Including the nearby TIC sources to the reference catalog...')
    gdat.catlrefr[0]['tmag'] = catalogData[:]['Tmag']
    
    gdat.catlrefr[0]['rasc'] = catalogData[:]['ra']
    gdat.catlrefr[0]['decl'] = catalogData[:]['dec']
    gdat.catlrefr[0]['tici'] = np.empty(len(catalogData), dtype=int)
    gdat.catlrefr[0]['tici'][:] = catalogData[:]['ID']
    gdat.catlrefr[0]['pmde'] = catalogData[:]['pmDEC']
    gdat.catlrefr[0]['pmra'] = catalogData[:]['pmRA']
    
    # turn tables into numpy arrays
    gdat.liststrgfeat = ['rasc', 'decl', 'tici', 'pmde', 'pmra', 'tmag']
    for strgfeat in gdat.liststrgfeat:
        gdat.catlrefr[0][strgfeat] = np.array(gdat.catlrefr[0][strgfeat])

    print('Number of sources in the reference catalog: %d' % len(gdat.catlrefr[0]['rasc']))
    
    print('maxmdmag')
    print(maxmdmag)
    dmag = gdat.catlrefr[0]['tmag'] - gdat.catlrefr[0]['tmag'][0]
    gdat.indxrefrbrgt = np.where(dmag < maxmdmag)[0]
    gdat.numbrefrbrgt = gdat.indxrefrbrgt.size
    magtcutt = gdat.catlrefr[0]['tmag'][0] + maxmdmag
    print('%d of the reference catalog sources are brighter than the magnitude cutoff of %g.' % (gdat.numbrefrbrgt, magtcutt))
    
    print('Removing nearby sources that are %g mag fainter than the target...' % maxmdmag)
    gdat.catlrefr[0]['rasc'] = gdat.catlrefr[0]['rasc'][gdat.indxrefrbrgt]
    gdat.catlrefr[0]['decl'] = gdat.catlrefr[0]['decl'][gdat.indxrefrbrgt]
    gdat.catlrefr[0]['tmag'] = gdat.catlrefr[0]['tmag'][gdat.indxrefrbrgt]
    gdat.catlrefr[0]['tici'] = gdat.catlrefr[0]['tici'][gdat.indxrefrbrgt]
    
    #print('Removing nearby sources that are too close...')
    ## calculate angular distances
    #distangl = 180. * np.sqrt((gdat.catlrefr[0]['rasc'][None, :] - gdat.catlrefr[0]['rasc'][:, None])**2 + \
    #                   (gdat.catlrefr[0]['decl'][None, :] - gdat.catlrefr[0]['decl'][:, None])**2)
    #print('distangl')
    #print(distangl)
    #summgene(distangl)
    #indxnzer = np.where(distangl != 0)
    #while np.amin(distangl[indxnzer]) < 0.5:
    #    # find the two sources that are too close
    #    indxsort = np.argsort(distangl[indxnzer])
    #    indkill = indxsort[0]
    #    print('indxkill')
    #    print(indxkill)
    #    # determine the fainter one
    #    if gdat.catlrefr[0][strgfeat]['tmag'][indxsort[0]] < gdat.catlrefr[0][strgfeat]['tmag'][indx[1]]:
    #        indxkill = indxsort[1]
    #    
    #    # determine the new indcices (i.e., without the one to be killed)
    #    indxtotl = np.arange(gdat.catlrefr[0]['rasc'].size)
    #    indxneww = np.setdiff1d(indxtotl, np.array(indxkill))
    #    
    #    # remove the faint source
    #    for strgfeat in gdat.liststrgfeat:
    #        gdat.catlrefr[0][strgfeat] = gdat.catlrefr[0][strgfeat][indxneww]
    #    
    #    # recalculate the distances
    #    distangl = np.sqrt((gdat.catlrefr[0]['rasc'][None, :] - gdat.catlrefr[0]['rasc'][:, None])**2 + \
    #                       (gdat.catlrefr[0]['decl'][None, :] - gdat.catlrefr[0]['decl'][:, None])**2)
    
    print('Remaining number of reference sources: %d' % gdat.catlrefr[0]['rasc'].size)
    gdat.catlrefr[0]['cnts'] = 10**(-(gdat.catlrefr[0]['tmag'] - 20.4) / 2.5)
    
    gdat.numbsourrefr = gdat.catlrefr[0]['rasc'].size
    catlrefrskyy = np.empty((gdat.numbsourrefr, 2))
    catlrefrskyy[:, 0] = gdat.catlrefr[0]['rasc']
    catlrefrskyy[:, 1] = gdat.catlrefr[0]['decl']
    
    if gdat.epocpmot is not None:
        print('Correcting the TIC catalog for proper motion...')
        if gdat.rasctarg is not None:
            pmra = gdat.pmratarg
            pmde = gdat.pmdetarg
            print('Using the user-provided values to correct for proper-motion...')
        else:
            pmra = gdat.catlrefr[0]['pmra'][gdat.indxrefrbrgt]
            pmde = gdat.catlrefr[0]['pmde'][gdat.indxrefrbrgt]
            print('Using the TIC catalog to correct for proper-motion...')
        gdat.catlrefr[0]['rasc'] += pmra * (gdat.epocpmot - 2000.) / (1000. * 3600.)
        gdat.catlrefr[0]['decl'] += pmde * (gdat.epocpmot - 2000.) / (1000. * 3600.)

    gdat.numbrefrlcur = np.empty(gdat.numbtsec, dtype=int)
    gdat.indxrefrlcur = [[] for o in gdat.indxtsec]
    gdat.lablrefrlcur = [[] for o in gdat.indxtsec]
    gdat.colrrefrlcur = [[] for o in gdat.indxtsec]
    for o in gdat.indxtsec:
        # determine what reference light curve is available for the sector
        if gdat.booltpxf[o]:
            gdat.lablrefrlcur[o] += ['SPOC']
            gdat.colrrefrlcur[o] = ['r']
    
        # number of reference light curves
        gdat.numbrefrlcur[o] = len(gdat.lablrefrlcur[o])
        gdat.indxrefrlcur[o] = np.arange(gdat.numbrefrlcur[o])
    
    gdat.refrtime = [[[] for k in gdat.indxrefrlcur[o]] for o in gdat.indxtsec]
    gdat.refrrflx = [[[] for k in gdat.indxrefrlcur[o]] for o in gdat.indxtsec]
    gdat.stdvrefrrflx = [[[] for k in gdat.indxrefrlcur[o]] for o in gdat.indxtsec]
    
    # write metadata to file
    gdat.pathsavemetaglob = gdat.pathdataobjt + 'metaglob.csv'
    dictmeta = dict()
    dictmeta['RA'] = gdat.rasctarg
    dictmeta['Dec'] = gdat.decltarg
    print('Writing to %s...' % gdat.pathsavemetaglob)
    objtfile = open(gdat.pathsavemetaglob, 'w')
    for key, value in dictmeta.items():
        objtfile.write('%s,%g\n' % (key, value))
    objtfile.close()

    gdat.boolrefr = [[] for o in gdat.indxtsec]
    cntr = 0
    for o in gdat.indxtsec:
        # get reference light curve
        if gdat.booltpxf[o]:
            arry, indxtimequalgood, indxtimenanngood, isecrefr, tcam, tccd = tesstarg.util.read_tesskplr_file(listpathdownlcur[cntr], strgtype='PDCSAP_FLUX')
            gdat.refrtime[o][0] = arry[:, 0]
            gdat.refrrflx[o][0] = arry[:, 1]
            gdat.stdvrefrrflx[o][0] = arry[:, 2]
            gdat.boolrefr[o] = True
            cntr += 1
        else:
            gdat.boolrefr[o] = False
        
    gdat.timedata = [[] for o in gdat.indxtsec]
    
    gdat.listarry = [[] for o in gdat.indxtsec]
    gdat.numbtime = np.empty(gdat.numbtsec, dtype=int)
    for o in gdat.indxtsec:
        strgsecc = retr_strgsecc(gdat, o)
        print('Sector: %d' % gdat.listtsec[o])
        print('Camera: %d' % gdat.listtcam[o])
        print('CCD: %d' % gdat.listtccd[o])
        if gdat.booltpxf[o]:
            print('TPF data')
        else:
            print('FFI data')
        
        if gdat.boolplotcntp or gdat.boolplotrflx or gdat.boolanim:
            gdat.strgtitlcntpplot = '%s, Sector %d, Cam %d, CCD %d' % (gdat.labltarg, gdat.listtsec[o], gdat.listtcam[o], gdat.listtccd[o])
        
        # get data
        if gdat.typedata == 'mock':
            gdat.listtime = np.linspace(0, gdat.numbtime - 1, gdat.numbtime)
            indxtimedatagood = np.arange(gdat.numbtime)
        else:
            # read the FITS files
            ## time
            gdat.listtime = listhdundata[o][1].data['TIME']
            
            ## count per pixel
            gdat.cntpdata = gdat.timeexpo * (listhdundata[o][1].data['FLUX'] + \
                                             listhdundata[o][1].data['FLUX_BKG']).swapaxes(0, 2).swapaxes(0, 1)
            
            if gdat.booltpxf[o]:
                gdat.numbside = gdat.cntpdata.shape[1]
            ## filter good times
            
            #indxtimedatagood = np.isfinite(np.mean(gdat.cntpdata, axis=(0, 1, 2))) & np.isfinite(gdat.listtime)
            indxtimedatagood = np.isfinite(gdat.listtime)
            if gdat.boolcuttqual:
                indxtimedatagood = indxtimedatagood & (listhdundata[o][1].data['QUALITY'] == 0)
            indxtimedatagood = np.where(indxtimedatagood)[0]
            
            if indxtimedatagood.size == 0:
                print('No good data found.')
                print('')
                return gdat.dictoutp
    
        gdat.listtime = gdat.listtime[indxtimedatagood]
        gdat.cntpdata = gdat.cntpdata[:, :, indxtimedatagood]
        
        gdat.numbtimecutt = gdat.listtime.size
        
        gdat.listtime = gdat.listtime[:gdat.numbtimecutt]
        gdat.cntpdata = gdat.cntpdata[:, :, :gdat.numbtimecutt]
       
        gdat.numbtime[o] = gdat.listtime.size
        gdat.indxtime = np.arange(gdat.numbtime[o])
        
        gdat.numbtimetsec = gdat.numbtime[o]

        gdat.gdatlion.numbtime = gdat.numbtime[o]
        gdat.gdatlion.indxtime = gdat.indxtime
        
        arrytemp = np.linspace(0., float(gdat.numbside - 1), gdat.numbside)
        gdat.xposimag, gdat.yposimag = np.meshgrid(arrytemp, arrytemp)
            
        if gdat.boolplotquat:
            # plot centroid
            figr, axis = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
            for k in range(2):
                if k == 0:
                    strg = 'x'
                    posi = gdat.xposimag
                else:
                    strg = 'y'
                    posi = gdat.yposimag
                cent = np.sum(posi[None, :, :, None] * gdat.cntpdata, axis=(0, 1, 2)) / np.sum(gdat.cntpdata, axis=(0, 1, 2))
                axis[k].plot(gdat.listtime, cent, ls='', marker='.', ms=1)
                axis[k].plot(gdat.listtime, cent, ls='', marker='.', ms=1)
                axis[k].set_ylabel('%s' % strg)
            axis[1].set_xlabel('Time [BJD]')
            path = gdat.pathimagobjt + 'cent_sc%02d.%s' % (gdat.listtsec[o], gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()






        if gdat.listpixlaper is not None:
            gdat.cntpaper = np.zeros(gdta.numbtime[o])
            for pixlaper in gdat.listpixlaper:
                gdat.cntpaper += gdgat.cntpdata[0, pixlaper[0], pixlaper[1], :]
            gdat.cntpaper /= np.median(gdat.cntpaper)
        
        gdat.cntpdatatmed = np.median(gdat.cntpdata, 2)

        # make mock data
        if gdat.typedata == 'mock':
            numbstar = 6
            magtstarprim = 16.
            magttrue = np.zeros((1, gdat.numbtime[o], numbstar))
            magttrue[:] = np.random.random(numbstar)[None, None, :] * 8 + 12.
            if truesourtype == 'supr':
                magttrue[0, :, 0] = 10. / (np.exp(10. * (gdat.listtime / gdat.numbtime[o] - 0.5)) + 1.) + 16.
            fluxtrue = 10**(0.4 * (20.4 - magttrue)) * gdat.timeexpo
            if boolcenttran:
                indxtimetran = []
                for t in gdat.indxtime:
                    if gdat.listtime[t] % 4. < 0.5:
                        indxtimetran.append(t)
                indxtimetran = np.array(indxtimetran)
                magttrue[0, indxtimetran, 0] *= 1e-3
            xpostrue = gdat.numbside * np.random.random(numbstar)
            ypostrue = gdat.numbside * np.random.random(numbstar)
            xpostrue[0] = gdat.numbside / 2. 
            ypostrue[0] = gdat.numbside / 2. 
            cntpbacktrue = np.zeros(gdat.numbtime[o]) + 1800 * 100.
            cntpdatatemp = retr_cntpmodl(gdat, coef, xpostrue, ypostrue, fluxtrue, cntpbacktrue)
            gdat.cntpdata = np.random.poisson(cntpdatatemp).astype(float)
        
        # filter input time series
        gdat.indxtimefitt = np.arange(gdat.cntpdata.shape[2])
        gdat.cntpdata = gdat.cntpdata[:, :, gdat.indxtimefitt]
        gdat.listtime = gdat.listtime[gdat.indxtimefitt]
        
        # rebin the data along the time axis
        gdat.listcade = np.ones_like(gdat.listtime) * gdat.timeexpo
        if gdat.facttimerebn != 1:
            print('Rebinning the data...')
            indxtimerebn = np.arange(0, len(gdat.listtime), gdat.facttimerebn)
            indxtimerebn = np.concatenate([indxtimerebn, np.array([gdat.listtime.size])])
            listtimetemp = np.empty(indxtimerebn.size - 1)
            for t in range(len(indxtimerebn) - 1):
                listtimetemp[t] = np.mean(gdat.listtime[indxtimerebn[t]:indxtimerebn[t+1]])
            gdat.listtime = np.copy(listtimetemp)

        gdat.numbtime[o] = gdat.listtime.size
        gdat.indxtime = np.arange(gdat.numbtime[o])

        if gdat.typedata != 'mock':
            objttimedata = astropy.time.Time(gdat.listtime + 2457000, format='jd', scale='utc')
            gdat.timedata[o] = objttimedata.jd
        else:
            gdat.timedata[o] = gdat.listtime
            gdat.refrtime[o] = gdat.listtime
        
        if gdat.boolplotquat:
            print('Reading quaternions...')
            path = gdat.pathdata + 'quat/'
            listfile = fnmatch.filter(os.listdir(path), 'tess*_sector%02d-quat.fits' % gdat.listtsec[o])
            pathquat = path + listfile[0]
            listhdun = astropy.io.fits.open(pathquat)
            dataquat = listhdun[gdat.listtcam[o]].data
            headquat = listhdun[gdat.listtcam[o]].header
            #for k, key in enumerate(headquat.keys()):
            #    print(key + ' ' + str(headquat[k]))
            figr, axis = plt.subplots(3, 1, figsize=(12, 4), sharex=True)
            quat = np.empty((gdat.numbtime[o], 3))
            for k in range(1, 4):
                strg = 'C%d_Q%d' % (gdat.listtcam[o], k)
                quat[:, k-1] = scipy.interpolate.interp1d(dataquat['TIME']+2457000,  dataquat[strg], fill_value=0, bounds_error=False)(gdat.timedata[o])
                minm = np.percentile(dataquat[strg], 0.05)
                maxm = np.percentile(dataquat[strg], 99.95)
                axis[k-1].plot(dataquat['TIME']+2457000, dataquat[strg], ls='', marker='.', ms=1)
                axis[k-1].set_ylim([minm, maxm])
                axis[k-1].set_ylabel('$Q_{%d}$' % k)
            axis[2].set_xlabel('Time [BJD]')
            path = gdat.pathimagobjt + 'quat_sc%02d.%s' % (gdat.listtsec[o], gdat.strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

        if not np.isfinite(gdat.cntpdata).all():
            print('Warning! Not all data are finite.')
        
        if gdat.catlextr is not None:
            for qq in gdat.indxcatlextr:
                gdat.catlrefr[qq+1]['rasc'] = gdat.catlextr[qq]['rasc']
                gdat.catlrefr[qq+1]['decl'] = gdat.catlextr[qq]['decl']

        ## reference catalogs
        for q in gdat.indxrefrcatl:
            if gdat.typedata == 'mock':
                gdat.catlrefr[q]['xpos'] = xpostrue
                gdat.catlrefr[q]['ypos'] = ypostrue
            else:
                catlrefrposi = listobjtwcss[o].all_world2pix(catlrefrskyy, 0)
                gdat.catlrefr[q]['xpos'] = catlrefrposi[:, 0]
                gdat.catlrefr[q]['ypos'] = catlrefrposi[:, 1]
            
            gdat.catlrefr[q]['flux'] = gdat.catlrefr[q]['xpos'] * 0

            print('Number of reference sources is %d.' % gdat.catlrefr[q]['xpos'].size)
            
            ## filter the reference catalog to be inside borders
            indxsourgood = np.where((gdat.catlrefr[q]['xpos'] > -0.5) & (gdat.catlrefr[q]['xpos'] < gdat.numbside - 0.5) & \
                                            (gdat.catlrefr[q]['ypos'] > -0.5) & (gdat.catlrefr[q]['ypos'] < gdat.numbside - 0.5))[0]
            
            print('Number of reference sources inside the ROI is %d. Discarding the rest...' % indxsourgood.size)
            gdat.catlrefrfilt[q]['rasc'] = gdat.catlrefr[q]['rasc'][indxsourgood]
            gdat.catlrefrfilt[q]['decl'] = gdat.catlrefr[q]['decl'][indxsourgood]
            gdat.catlrefrfilt[q]['xpos'] = gdat.catlrefr[q]['xpos'][indxsourgood]
            gdat.catlrefrfilt[q]['ypos'] = gdat.catlrefr[q]['ypos'][indxsourgood]
            gdat.catlrefrfilt[q]['tmag'] = gdat.catlrefr[q]['tmag'][indxsourgood]
            gdat.catlrefrfilt[q]['tici'] = gdat.catlrefr[q]['tici'][indxsourgood]
            
        gdat.numbsourcatlrefr = gdat.catlrefrfilt[0]['tici'].size
        gdat.indxsourcatlrefr = np.arange(gdat.numbsourcatlrefr)
        
        if gdat.facttimerebn != 1:
            listcadetemp = np.empty(indxtimerebn.size - 1)
            cntpdatatemp = np.empty((gdat.numbside, gdat.numbside, indxtimerebn.size - 1))
            for t in range(len(indxtimerebn) - 1):
                temp = gdat.cntpdata[:, :, indxtimerebn[t]:indxtimerebn[t+1]]
                cntpdatatemp[:, :, t] = np.sum(temp, axis=2)
                listcadetemp[t] = gdat.timeexpo * temp.shape[2]
            gdat.listcade = np.copy(listcadetemp)
            gdat.cntpdata = np.copy(cntpdatatemp)
        
        if np.amin(gdat.cntpdata) < 0:
            print('Minimum of the image is negative.')

        # data weights
        gdat.vari = np.zeros_like(gdat.cntpdata) + np.mean(gdat.cntpdata)

        ## fitting catalog
        if strgtargtype == 'posi':
            gdat.rascfitt = np.empty(1 + len(gdat.catlrefrfilt[0]['rasc']))
            gdat.declfitt = np.empty(1 + len(gdat.catlrefrfilt[0]['rasc']))
            gdat.rascfitt[0] = gdat.rasctarg
            gdat.declfitt[0] = gdat.decltarg
            gdat.rascfitt[1:] = gdat.catlrefrfilt[0]['rasc']
            gdat.declfitt[1:] = gdat.catlrefrfilt[0]['decl']
            if gdat.psfntype == 'ontf':
                gdat.cntsfitt = np.concatenate((np.array([0.]), gdat.catlrefrfilt[0]['cnts']))
                print('Do not know what counts to assume for the target (for the on-the-fly PSF fit). Assuming 0!')
        else:
            gdat.rascfitt = gdat.catlrefrfilt[0]['rasc']
            gdat.declfitt = gdat.catlrefrfilt[0]['decl']
            if gdat.psfntype == 'ontf':
                gdat.cntsfitt = gdat.catlrefrfilt[0]['cnts']
        
        skyyfitttemp = np.empty((gdat.rascfitt.size, 2))
        skyyfitttemp[:, 0] = gdat.rascfitt
        skyyfitttemp[:, 1] = gdat.declfitt
        if gdat.rascfitt.size == 0:
            raise Exception('')
        # transform sky coordinates into dedector coordinates and filter
        posifitttemp = listobjtwcss[o].all_world2pix(skyyfitttemp, 0)
        gdat.xposfitt = posifitttemp[:, 0]
        gdat.yposfitt = posifitttemp[:, 1]
        
        if gdat.xposfitt.size == 0:
            print('No fitting source found...')
            print('')
            return gdat.dictoutp
        
        # filter the fitting sources
        if gdat.maxmnumbstar is not None:
            print('Taking the first %d sources among the neighbors.' % gdat.maxmnumbstar)
            gdat.xposfitt = gdat.xposfitt[:gdat.maxmnumbstar]
            gdat.yposfitt = gdat.yposfitt[:gdat.maxmnumbstar]
        
        gdat.numbstar = gdat.xposfitt.size
        gdat.indxstar = np.arange(gdat.numbstar)
            
        print('Number of point sources in the model: %d' % gdat.numbstar)
    
        gdat.numbpixl = gdat.numbside**2
        
        # plotting settings
        for typecntpscal in gdat.listtypecntpscal:
            for strg in ['data']:
                setp_cntp(gdat, strg, typecntpscal)
    
        # types of image plots
        listnameplotcntp = ['cntpdata', 'cntpdatanbkg', 'cntpmodl', 'cntpresi']
        
        gdat.sizeplotsour = 20
        
        if not np.isfinite(gdat.cntpdata).all():
            print('There is NaN in the data!')
        
        gdat.boolbackoffs = True
        gdat.boolposioffs = False
        
        gdat.cntpback = np.zeros_like(gdat.cntpdata)

        if np.amin(gdat.cntpdata) < 1.:
            print('Minimum of the data is not positive.')

        if not np.isfinite(gdat.cntpback).all():
            raise Exception('')
        
        if gdat.cntpdata.shape[0] != gdat.numbside:
            raise Exception('')

        if gdat.cntpdata.shape[1] != gdat.numbside:
            raise Exception('')

        if gdat.psfntype == 'lion':
            gdat.gdatlion.numbtime = 1
            gdat.gdatlion.indxtime = [0]
        
        # number of components, 1 for background, 3 for quaternions
        gdat.numbcomp = gdat.numbstar + 1# + 3
        gdat.indxcomp = np.arange(gdat.numbcomp)

        gdat.stdvfittflux = np.empty((gdat.numbtime[o], gdat.numbcomp))
        
        #print('Evaluating the PSFs for contamination ratio...')
        #numbstarneig = 10
        #indxstarneig = np.arange(numbstarneig)
        #

        # fit for the PSF
        if gdat.psfntype == 'ontf':
            numbsampwalk = 200000
            numbsampburnwalk = int(0.1 * numbsampwalk)
            numbsampburnwalkseco = int(0.5 * numbsampwalk)
            if gdat.psfnshaptype == 'gaus':
                gdat.numbcoef = 0
            if gdat.psfnshaptype.startswith('gfre'):
                gdat.numbcoef = 2
            if gdat.psfnshaptype == 'pfre':
                gdat.numbcoef = 10
            indxcoef = np.arange(gdat.numbcoef)
            
            # plot the median image
            if gdat.boolplotcntp:
                for typecntpscal in gdat.listtypecntpscal:
                    nameplot = '%scntpdatatmed' % (gdat.pathimagobjtpsfn)
                    strgtitl = gdat.strgtitlcntpplot
                    plot_cntp(gdat, gdat.cntpdatatmed, o, typecntpscal, nameplotcntp, strgdataextn, strgtitl=strgtitl)

            listlablpara = []
            for k in indxcoef:
                listlablpara.append(['$c_{%d}$' % k, ''])
            listlablpara.append(['B', 'e$^{-}$'])
            
            if gdat.psfnshaptype == 'gfreffix':
                listlablpara.append(['$A$', ''])
            elif gdat.psfnshaptype == 'gfrefinf':
                listlablpara.append(['$A$', ''])
                for k in gdat.indxstar:
                    listlablpara.append(['$A_{%i}$' % k, ''])
            elif gdat.psfnshaptype == 'gfreffre':
                for k in gdat.indxstar:
                    listlablpara.append(['$F_{%i}$' % k, 'e$^{-}$'])
                
            gdat.numbparapsfn = len(listlablpara)
            gdat.indxparapsfn = np.arange(gdat.numbparapsfn)
            listscalpara = ['self' for k in gdat.indxparapsfn]
            listmeangauspara = None
            liststdvgauspara = None
            listminmpara = np.zeros(gdat.numbparapsfn)
            listmaxmpara = np.zeros(gdat.numbparapsfn)
            if gdat.psfnshaptype == 'gaus':
                coef = None
            if gdat.psfnshaptype.startswith('gfre'):
                listminmpara[0] = 0.3
                listmaxmpara[0] = 5. 
                listminmpara[1] = 0.3
                listmaxmpara[1] = 5. 
                coef = np.array([1., 1.])
            
            if gdat.psfnshaptype == 'gaus' or gdat.psfntype == 'gfre' or gdat.psfntype == 'pfre':
                # solve for the maximum likelihood fluxes for the median image
                matrdesi = np.ones((gdat.numbpixl, gdat.numbstar + 1))
                for k in np.arange(gdat.numbstar):
                    matrdesi[:, k] = retr_cntpmodl(gdat, coef, gdat.xposfitt[k, None], gdat.yposfitt[k, None], np.array([[1.]]), np.array([0.])).flatten()
                matrdesi[:, gdat.numbstar] = 1.
                gdat.mlikfittfluxmedi, gdat.covafittfluxmedi = retr_mlikflux(gdat.cntpdatatmed, matrdesi, gdat.cntpdatatmed)
            
            # background
            listminmpara[gdat.numbcoef] = 0.
            listmaxmpara[gdat.numbcoef] = np.amax(gdat.cntpdatatmed)#abs(gdat.mlikfittfluxmedi[-1]) * 2.
            
            if gdat.psfnshaptype == 'gfreffix' or gdat.psfntype =='gfrefinf':
                listminmpara[gdat.numbcoef+1] = 0.
                listmaxmpara[gdat.numbcoef+1] = 10000.
                if gdat.psfnshaptype == 'gfrefinf':
                    listminmpara[gdat.numbcoef+2:] = 0.
                    listmaxmpara[gdat.numbcoef+2:] = 10.
            elif gdat.psfnshaptype == 'gfreffre':
                listminmpara[gdat.numbcoef+1:] = 0.
                listmaxmpara[gdat.numbcoef+1:] = np.amax(gdat.cntpdatatmed)#abs(gdat.mlikfittfluxmedi[:-1]) * 2.
            else:
                listminmpara[gdat.numbcoef+1:] = 0.
                listmaxmpara[gdat.numbcoef+1:] = np.amax(gdat.cntpdatatmed)#abs(gdat.mlikfittfluxmedi[:-1]) * 2.
            
            strgextn = 'psfn'
            numbdata = gdat.numbpixl
            strgsaveextn = gdat.pathdataobjt + gdat.psfnshaptype + '_' + gdat.strgcnfg + '.txt'
            parapost = tdpy.mcmc.samp(gdat, gdat.pathimagobjtpsfn, numbsampwalk, numbsampburnwalk, numbsampburnwalkseco, retr_llik, \
                                            listlablpara, listscalpara, listminmpara, listmaxmpara, listmeangauspara, liststdvgauspara, \
                                                numbdata, strgextn=strgextn, strgplotextn=strgplotextn, strgsaveextn=strgsaveextn)
            
            paramedi = np.median(parapost, 0)
            coefmedi = paramedi[:gdat.numbcoef]
            cntpbackmedi = paramedi[gdat.numbcoef]
            if gdat.psfnshaptype == 'gfreffix':
                ampltotlmedi = paramedi[gdat.numbcoef+1]
            elif gdat.psfnshaptype == 'gfrefinf':
                ampltotlmedi = paramedi[gdat.numbcoef+1]
                amplrelamedi = paramedi[gdat.numbcoef+2:]
            elif gdat.psfnshaptype == 'gfreffre':
                fluxmedi = paramedi[gdat.numbcoef+1:]
                
            if gdat.boolplotcntp:
                # plot the posterior median PSF model
                xpossour = np.array([(gdat.numbside - 1.) / 2.])
                ypossour = np.array([(gdat.numbside - 1.) / 2.])
                flux = np.array([[1.]])
                cntpback = np.zeros(1)
                gdat.cntpmodlpsfn = retr_cntpmodl(gdat, coefmedi, xpossour, ypossour, flux, cntpback, verbtype=1)
                nameplot = '%scntpmodlpsfnmedi' % (gdat.pathimagobjtpsfn)
                strgtitl = gdat.strgtitlcntpplot
                for typecntpscal in gdat.listtypecntpscal:
                    plot_cntp(gdat, gdat.cntpmodlpsfn[:, :, 0], o, typecntpscal, nameplotcntp, strgdataextn, strgtitl=strgtitl, boolanno=False)

                # plot the posterior median image model
                if gdat.psfnshaptype == 'gfreffix':
                    flux = gdat.cntsfitt * amplmedi
                elif gdat.psfnshaptype == 'gfrefinf':
                    flux = gdat.cntsfitt * ampltotlmedi * amplrelamedi
                elif gdat.psfnshaptype == 'gfreffre':
                    flux = fluxmedi
                gdat.cntpmodlpsfn = retr_cntpmodl(gdat, coefmedi, gdat.xposfitt, gdat.yposfitt, flux, cntpbackmedi)
                
                nameplot = '%scntpmodlmedi' % (gdat.pathimagobjtpsfn)
                strgtitl = gdat.strgtitlcntpplot
                for typecntpscal in gdat.listtypecntpscal:
                    plot_cntp(gdat, gdat.cntpmodlpsfn[:, :, 0], o, typecntpscal, nameplotcntp, strgdataextn, strgtitl=strgtitl)

                # plot the posterior median residual
                nameplot = '%scntpresimedi' % (gdat.pathimagobjtpsfn)
                strgtitl = gdat.strgtitlcntpplot
                for typecntpscal in gdat.listtypecntpscal:
                    plot_cntp(gdat, gdat.cntpdatatmed - gdat.cntpmodlpsfn[:, :, 0], o, typecntpscal, nameplotcntp, strgdataextn, \
                                                                                                                    strgtitl=strgtitl, boolresi=True)

        if gdat.boolanim:
            if gdat.boolanimframtotl:
                numbplotanim = gdat.numbtime[o].size
            else:
                numbplotanim = 10
            # time indices to be included in the animation
            indxtimeanim = np.linspace(0., gdat.numbtime[o] - 1., numbplotanim).astype(int)
        
            # get time string
            objttime = astropy.time.Time(gdat.timedata[o], format='jd', scale='utc', out_subfmt='date_hm')
            listtimelabl = objttime.iso
        
        gdat.offs = 0.5
        gdat.listoffsxpos = np.linspace(-gdat.offs, gdat.offs, 3)
        gdat.listoffsypos = np.linspace(-gdat.offs, gdat.offs, 3)
        gdat.numboffs = gdat.listoffsxpos.size
        gdat.indxoffs = np.arange(gdat.numboffs)
        for x in gdat.indxoffs:
            for y in gdat.indxoffs:
                
                if not gdat.boolfittoffs and (x != 1 or y != 1):
                    continue

                strgoffs = retr_strgoffs(gdat, x, y)
                strgdataextn = retr_strgdataextn(gdat, strgsecc, strgoffs, gdat.typecade[o])
                pathsaverflx = gdat.pathdataobjt + 'rflx' + strgdataextn + '.csv'
                pathsaverflxtarg = gdat.pathdataobjt + 'rflxtarg' + strgdataextn + '.csv'
                pathsaverflxtargbdtr = gdat.pathdataobjt + 'rflxtargbdtr' + strgdataextn + '.csv'
                pathsavemeta = gdat.pathdataobjt + 'meta' + strgdataextn + '.csv'
        
                # plot a histogram of data counts
                if gdat.boolplothhistcntp:
                    figr, axis = plt.subplots(figsize=(12, 4))
                    bins = np.logspace(np.log10(np.amin(gdat.cntpdata)), np.log10(np.amax(gdat.cntpdata)), 200)
                    axis.hist(gdat.cntpdata.flatten(), bins=bins)
                    axis.set_xscale('log')
                    axis.set_yscale('log')
                    axis.set_ylabel('N')
                    axis.set_xlabel('C [e$^{-}$]')
                    plt.tight_layout()
                    path = gdat.pathimagobjt + 'histcntpdata_%s.%s' % (strgdataextn, gdat.strgplotextn)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
        
                # introduce the positional offset
                xpostemp = np.copy(gdat.xposfitt)
                ypostemp = np.copy(gdat.yposfitt)
                xpostemp[0] = gdat.xposfitt[0] + gdat.listoffsxpos[x]
                ypostemp[0] = gdat.yposfitt[0] + gdat.listoffsypos[y]
                
                if not os.path.exists(pathsaverflx):
                    
                    gdat.covafittflux = np.empty((gdat.numbtime[o], gdat.numbstar + 1, gdat.numbstar + 1))
                    gdat.mlikfittflux = np.empty((gdat.numbtime[o], gdat.numbstar + 1))
                    matrdesi = np.ones((gdat.numbpixl, gdat.numbstar + 1))

                    print('Evaluating the PSFs to create templates for the design matrix...')
                    coef = None
                    fluxtemp = np.ones((1, gdat.numbstar))
                    cntptemp = np.zeros(1)
                    for k in np.arange(gdat.numbstar):
                        matrdesi[:, k] = retr_cntpmodl(gdat, xpostemp[k, None], ypostemp[k, None], fluxtemp[:, k, None], cntptemp, coef).flatten()
                            
                    # solve the linear system
                    print('Solving the linear systems of equations...')
                    timeinit = timemodu.time()
                    for t in gdat.indxtime:
                        gdat.mlikfittflux[t, :], gdat.covafittflux[t, :, :] = retr_mlikflux(gdat.cntpdata[:, :, t], matrdesi, gdat.vari[:, :, t])

                    timefinl = timemodu.time()
                    print('Done in %g seconds.' % (timefinl - timeinit))
                    
                    for k in gdat.indxcomp:
                        gdat.stdvfittflux[:, k] = np.sqrt(gdat.covafittflux[:, k, k])
                    
                    gdat.medifittflux = np.median(gdat.mlikfittflux, 0)
                    print('Median flux of the central source is %g ADU.' % gdat.medifittflux[0])
                    
                    # normalize fluxes to get relative fluxes
                    print('Normalizing by the median flux...')
                    gdat.mlikfittrflx = gdat.mlikfittflux / gdat.medifittflux[None, :]
                    gdat.stdvfittrflx = gdat.stdvfittflux / gdat.medifittflux[None, :]
                    
                    if gdat.booldiagmode:
                        for a in range(gdat.timedata[o].size):
                            if a != gdat.timedata[o].size - 1 and gdat.timedata[o][a] >= gdat.timedata[o][a+1]:
                                raise Exception('')

                    # write the light curve to file
                    print('Writing all light curves to %s...' % pathsavemeta)
                    arry = gdat.medifittflux
                    np.savetxt(pathsavemeta, arry)
                    
                    arry = np.empty((gdat.numbtime[o], 2*gdat.numbcomp+1))
                    arry[:, 0] = gdat.timedata[o]
                    for k in gdat.indxcomp:
                        arry[:, 2*k+1] = gdat.mlikfittrflx[:, k]
                        arry[:, 2*k+2] = gdat.stdvfittrflx[:, k]
                    print('Writing all light curves to %s...' % pathsaverflx)
                    np.savetxt(pathsaverflx, arry)
                    
                    print('Writing the target light curve to %s...' % pathsaverflxtarg)
                    np.savetxt(pathsaverflxtarg, arry[:, :3])
                    gdat.listarry[o] = arry[:, :3]

                    if gdat.booldiagmode:
                        for a in range(gdat.listarry[o][:, 0].size):
                            if a != gdat.listarry[o][:, 0].size - 1 and gdat.listarry[o][a, 0] >= gdat.listarry[o][a+1, 0]:
                                raise Exception('')

                else:
                    print('Skipping the regression...')
                
                if gdat.boolplotrflx:
                    nameplot = 'rflx'
                    path = retr_pathvisu(gdat, nameplot, strgdataextn)
                    if os.path.exists(path):
                        print('Plots already exist at %s...' % path)
                    else:
                    
                        print('Reading from %s...' % pathsavemeta)
                        gdat.medifittflux = np.loadtxt(pathsavemeta)
                        
                        gdat.mlikfittrflx = np.empty((gdat.numbtime[o], gdat.numbcomp))
                        gdat.stdvfittrflx = np.empty((gdat.numbtime[o], gdat.numbcomp))
                        print('Reading from %s...' % pathsaverflx)
                        arry = np.loadtxt(pathsaverflx)
                        gdat.timedata[o] = arry[:, 0]
                        for k in gdat.indxcomp:
                            gdat.mlikfittrflx[:, k] = arry[:, 2*k+1]
                            gdat.stdvfittrflx[:, k] = arry[:, 2*k+2]
                        gdat.mlikfittflux = gdat.medifittflux * gdat.mlikfittrflx
                
                print('Evaluating the model at all time bins...')
                cntpbackneww = gdat.mlikfittflux[:, -1]
                timeinit = timemodu.time()
                coef = None
                gdat.cntpmodl = retr_cntpmodl(gdat, xpostemp, ypostemp, gdat.mlikfittflux[:, :-1], cntpbackneww, coef)
                print('gdat.cntpmodl')
                summgene(gdat.cntpmodl)
                print('gdat.cntpdata')
                summgene(gdat.cntpdata)

                timefinl = timemodu.time()
                print('Done in %g seconds.' % (timefinl - timeinit))
                    
                gdat.cntpdatanbkg = gdat.cntpdata - gdat.mlikfittrflx[None, None, :, -1] * gdat.medifittflux[-1]
                gdat.cntpresi = gdat.cntpdata - gdat.cntpmodl
                chi2 = np.mean(gdat.cntpresi**2 / gdat.cntpdata) + 2 * gdat.numbstar
                
        
                # color scales
                for typecntpscal in gdat.listtypecntpscal:
                    for strg in ['modl', 'resi', 'datanbkg']:
                        setp_cntp(gdat, strg, typecntpscal)
                
                if gdat.boolplotrflx:
                    if not os.path.exists(path):
                        print('Plotting light curves...')
                        if gdat.listlimttimetzom is not None:
                            gdat.indxtimelimt = []
                            for limttimeplot in gdat.listlimttimetzom:
                                gdat.indxtimelimt.append(np.where((gdat.timedata[o] > limttimeplot[0]) & (gdat.timedata[o] < limttimeplot[1]))[0])
                    
                        # plot light curve derived from aperture photometry
                        if gdat.listpixlaper is not None:
                            plot_lcur(gdat, gdat.cntpaper, 0.01 * gdat.cntpaper, k, o, '_' + strgsecc, gdat.booltpxf[o], '_aper', strgdataextn)
                            
                        # plot the light curve of the target stars and background
                        for k in gdat.indxcomp:

                            if gdat.boolbdtr and (x == 1 and y == 1) or not gdat.boolbdtr and (k == 0 and x == 1 and y == 1):

                                if gdat.boolrefr[o] and x == 1 and y == 1:
                                    listmodeplot = [0, 1]
                                else:
                                    listmodeplot = [0]
                                plot_lcur(gdat, gdat.mlikfittrflx[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, \
                                                                                 gdat.booltpxf[o], strgoffs, strgdataextn, listmodeplot=listmodeplot)
                                
                                if gdat.listlimttimetzom is not None:
                                    for p in range(len(gdat.listlimttimetzom)):
                                        plot_lcur(gdat, gdat.mlikfittrflx[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, gdat.booltpxf[o], strgoffs, \
                                                                                       strgdataextn, indxtimelimt=gdat.indxtimelimt[p], indxtzom=p)
                                        if k == 0:
                                            plot_lcur(gdat, gdat.mlikfittflux[:, k], gdat.stdvfittflux[:, k], k, o, '_' + strgsecc, \
                                                                                                    gdat.booltpxf[o], strgoffs, strgdataextn, \
                                                                                              indxtimelimt=gdat.indxtimelimt[p], indxtzom=p)
                                
                        if gdat.boolbdtr:
                            gdat.mlikrflxbdtr = np.empty_like(gdat.mlikfittrflx)
                            gdat.mlikrflxspln = np.empty_like(gdat.mlikfittrflx)
                            for k in gdat.indxcomp:
                                
                                if k > 0 and (x != 1 or y != 1):
                                    continue
                            
                                lcurbdtrregi, indxtimeregi, indxtimeregioutt, listobjtspln, timeedge = \
                                                    tesstarg.util.bdtr_lcur(gdat.timedata[o], gdat.mlikfittrflx[:, k], \
                                                    booladdddiscbdtr=gdat.booladdddiscbdtr, \
                                                    bdtrtype=gdat.bdtrtype, verbtype=1)
                                gdat.mlikrflxbdtr[:, k] = np.concatenate(lcurbdtrregi)
                                gdat.mlikrflxspln[:, k] = 1. + gdat.mlikfittrflx[:, k] - gdat.mlikrflxbdtr[:, k]
                                
                                if k == 0:
                                    arry = np.empty((gdat.numbtime[o], 3))
                                    arry[:, 0] = gdat.timedata[o]
                                    arry[:, 1] = gdat.mlikrflxbdtr[:, 0]
                                    arry[:, 2] = gdat.stdvfittrflx[:, 0]
                                    print('Writing the target light curve to %s...' % pathsaverflxtargbdtr)
                                    np.savetxt(pathsaverflxtargbdtr, arry)
                                
                                plot_lcur(gdat, gdat.mlikrflxspln[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, \
                                                                                             gdat.booltpxf[o], strgoffs, strgdataextn, strgextn='_spln', \
                                                                                                                                    timeedge=timeedge)
                                plot_lcur(gdat, gdat.mlikrflxbdtr[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, \
                                                                                             gdat.booltpxf[o], strgoffs, strgdataextn, strgextn='_bdtr')
                                if gdat.listlimttimetzom is not None:
                                    for p in range(len(gdat.listlimttimetzom)):
                                        plot_lcur(gdat, gdat.mlikrflxbdtr[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, gdat.booltpxf[o], strgoffs, \
                                                                       strgdataextn, indxtimelimt=gdat.indxtimelimt[p], strgextn='_bdtr', indxtzom=p)

                        else:
                            gdat.mlikrflxbdtr = gdat.mlikfittrflx[:, 0]
                        
        # temporal types of image plots
        ## medians
        listtypeplotcntp = ['medi']
        # cadence frames
        if gdat.boolanim:
            listtypeplotcntp += ['anim']
        
        if gdat.boolplotcntp:
            for typeplotcntp in listtypeplotcntp:
                for nameplotcntp in listnameplotcntp:
                    
                    # make animation plot
                    pathanim = retr_pathvisu(gdat, nameplotcntp, strgdataextn, boolanim=True)
                    if typeplotcntp == 'anim' and not os.path.exists(path):
                        continue

                    if typeplotcntp == 'anim':
                        print('Making an animation of frame plots...')
                        strg = ''
                    else:
                        strg = 'tmed'
                    cntptemp = getattr(gdat, nameplotcntp + strg)
                    
                    if nameplotcntp == 'cntpresi':
                        boolresi = True
                    else:
                        boolresi = False
                    
                    for typecntpscal in gdat.listtypecntpscal:
                        # minimum and maximum
                        vmin = getattr(gdat, 'vmin' + nameplotcntp + typecntpscal)
                        vmax = getattr(gdat, 'vmax' + nameplotcntp + typecntpscal)
                    
                        if typeplotcntp == 'medi':
                            strgtitl = gdat.strgtitlcntpplot
                            plot_cntp(gdat, cntptemp, o, typecntpscal, nameplotcntp, strgdataextn, strgtitl=strgtitl, boolresi=boolresi, vmin=vmin, vmax=vmax)
                        if typeplotcntp == 'anim':
                            args = [gdat, cntptemp, o, typecntpscal, nameplotcntp, strgdataextn]
                            kwag = {'indxtimeanim': indxtimeanim, 'nameplotcntp': nameplotcntp, \
                                                'boolresi': boolresi, 'listindxpixlcolr': gdat.listpixlaper, \
                                                'listtimelabl':listtimelabl, \
                                                'vmin':vmin, 'vmax':vmax, \
                                                'lcur':gdat.mlikfittrflx[:, 0], 'time':gdat.timedata[o]}
                            listpath = plot_cntpwrap(*args, **kwag)
                    
                            # make animation
                            cmnd = 'convert -delay 20 '
                            for path in listpath:
                                cmnd += '%s ' % path
                            cmnd += '%s.gif' % pathanim
                            os.system(cmnd)
                            
                            # delete images
                            for path in listpath:
                                os.system('rm %s' % path)
        
    timefinltotl = timemodu.time()
    print('Total execution time: %g seconds.' % (timefinltotl - timeinittotl))
    print('')                
    gdat.dictoutp['listarry'] = gdat.listarry
    
    return gdat.dictoutp


def retr_strgsecc(gdat, o):
    
    strgsecc = '%02d%d%d' % (gdat.listtsec[o], gdat.listtcam[o], gdat.listtccd[o])

    return strgsecc


def init_list(pathfile, strgbase, **kwag):
    
    pathbase = os.environ['LYGOS_DATA_PATH'] + '/%s/' % strgbase
    os.system('mkdir -p %s' % pathbase)

    listticitarg = []
    listlabl = []
    listrasc = []
    listdecl = []
    listintgresu = []
    for line in open(pathfile):
        listline = line.split('  ')
        labltarg = listline[0]
        strgtarg = labltarg
        rasctarg = float(listline[1])
        decltarg = float(listline[2])
        intgresu = main( \
                        strgtarg=strgtarg, \
                        rasctarg=rasctarg, \
                        decltarg=decltarg, \
                        labltarg=labltarg, \
                        strgbase=strgbase, \
                        **kwag \
                       )
        listintgresu.append(intgresu) 
    print('listintgresu')
    print(listintgresu)


