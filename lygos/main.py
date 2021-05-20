import numpy as np

import tqdm

import scipy.interpolate
from scipy import ndimage

#from numba import jit, prange

import h5py

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(10)
figrsizeydob = [8., 4.]
figr, axis = plt.subplots(figsize=figrsizeydob)
axis.plot(x, x)
axis.set_ylabel('Flux')
axis.set_xlabel('Time [BJD]')
plt.savefig('/Users/tdaylan/Desktop/test5.pdf')
plt.close()
            
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
import ephesus.util

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(10)
figrsizeydob = [8., 4.]
figr, axis = plt.subplots(figsize=figrsizeydob)
axis.plot(x, x)
axis.set_ylabel('Flux')
axis.set_xlabel('Time [BJD]')
plt.savefig('/Users/tdaylan/Desktop/test6.pdf')
plt.close()
            
from lion import main as lionmain

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(10)
figrsizeydob = [8., 4.]
figr, axis = plt.subplots(figsize=figrsizeydob)
axis.plot(x, x)
axis.set_ylabel('Flux')
axis.set_xlabel('Time [BJD]')
plt.savefig('/Users/tdaylan/Desktop/test7.pdf')
plt.close()
            

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


def plot_cntpwrap(gdat, cntp, o, typecntpscal, nameplotcntp, strgsave, \
                                                    boolresi=False, listindxpixlcolr=None, indxpcol=None, \
                                                                                time=None, lcur=None, \
                                                                                vmin=None, vmax=None, \
                                                                                listtime=None, listtimelabl=None, \
                                                                                ):
    
    if time is None:
        time = [None for t in gdat.indxtime[o]]

    if listtimelabl is None:
        listtimelabl = [None for t in gdat.indxtime[o]]

    listpath = []
    for tt, t in enumerate(gdat.indxtimeanim):
        pathcntp = retr_pathvisu(gdat, nameplotcntp, strgsave, indxpcol=indxpcol, indxtimeplot=t)
        
        print('tt')
        print(tt)
        print('t')
        print(t)
        print('')
        # make title
        strgtitl = gdat.strgtitlcntpplot
        if listtimelabl[t] is not None:
            strgtitl += ', %s' % listtimelabl[t]
        
        path = plot_cntp(gdat, cntp[:, :, t], o, typecntpscal, nameplotcntp, strgsave, \
                                                strgtitl=strgtitl, boolresi=boolresi, listindxpixlcolr=listindxpixlcolr, \
                                                                                            timelabl=listtimelabl[t], thistime=time[t], indxtimeplot=t, \
                                                                                                vmin=vmin, vmax=vmax, lcur=lcur, time=time)
        
        listpath.append(path)

    return listpath


def plot_cntp(gdat, cntp, o, typecntpscal, nameplotcntp, strgsave, indxpcol=None, \
                                            cbar='Greys_r', strgtitl='', boolresi=False, xposoffs=None, yposoffs=None, strgextn='', \
                                                                                           lcur=None, boolanno=True, indxtimeplot=None, \
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
    
    path = retr_pathvisu(gdat, nameplotcntp, strgsave, typecntpscal=typecntpscal, indxpcol=indxpcol, indxtimeplot=indxtimeplot)
    print('Writing to %s...' % path)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
    return path


def plot_lcur(gdat, lcurmodl, stdvlcurmodl, k, indxtsecplot, strgsecc, booltpxf, strgoffs, strgsave, timeedge=None, listmodeplot=[0], \
                    strgextn='', indxtimelimt=None, indxtzom=None, boolerrr=False):
    
    if k == 0:
        lablcomp = ', Target source'
    elif k == gdat.numbcomp - 1:
        lablcomp = ', Background'
    else:
        lablcomp = ', Neighbor Source %d' % k

    timedatatemp = np.copy(gdat.listtime[indxtsecplot])
    timerefrtemp = [[] for q in gdat.indxrefrlcur[indxtsecplot]] 
    for q in gdat.indxrefrlcur[indxtsecplot]:
        timerefrtemp[q] = gdat.refrtime[indxtsecplot][q]
    
    if indxtimelimt is not None:
        timedatatemp = timedatatemp[indxtimelimt]
        stdvlcurmodl = stdvlcurmodl[indxtimelimt]
        lcurmodl = lcurmodl[indxtimelimt]

    lablxaxi = 'Time [BJD]'
    
    for a in listmodeplot:
        nameplot = 'rflx_s%03d_mod%d' % (k, a)
        path = retr_pathvisu(gdat, nameplot, strgsave, indxtzom=indxtzom)
        
        # skip the plot if it has been made before
        if os.path.exists(path):
            continue

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
                                                                                marker='.', lw=3, alpha=0.3, label='Lygos')
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
                temp, listcaps, temp = axis[0].errorbar(timerefrtemp, gdat.refrrflx[indxtsecplot][q], \
                                                    yerr=yerr, color=gdat.colrrefrlcur[indxtsecplot][q], ls='', markersize=2, \
                                                                            marker='.', lw=3, alpha=0.3, label=gdat.lablrefrlcur[indxtsecplot][q])
                for caps in listcaps:
                    caps.set_markeredgewidth(3)
            
            ## residual
            for q in gdat.indxrefrlcur[indxtsecplot]:
                if lcurmodl.size == gdat.refrrflx[indxtsecplot][q].size:
                    print('q')
                    print(q)
                    print('lcurmodl')
                    summgene(lcurmodl)
                    ydat = lcurmodl - gdat.refrrflx[indxtsecplot][q]
                    if boolerrr:
                        yerr = None
                    else:
                        yerr = None
                    axis[1].errorbar(timedatatemp, ydat, yerr=yerr, label=gdat.lablrefrlcur[indxtsecplot][q], \
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
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()


def retr_pathvisu(gdat, \
                    nameplot, \
                    # data
                    strgsave, \
                    
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
        strgtime = '_%08d' % indxtimeplot
    
    pathvisu = gdat.pathimagtarg + '%s%s%s%s%s%s.%s' % \
               (nameplot, strgsave, strgscal, strgtzom, strgpcol, strgtime, strgplotextn)
    
    return pathvisu


def retr_cntpmodl(gdat, xpos, ypos, flux, cntpback, o, coef=None, verbtype=1):
    
    if gdat.typepsfn == 'lion':
        gdat.gdatlion.numbtime = flux.shape[0]
        cntpmodl = lionmain.eval_modl(gdat.gdatlion, xpos, ypos, flux[None, :, :], cntpback[None, None, None, :])[0, :, :, :]
    else:
        cntpmodl = np.zeros_like((gdat.numbside, gdat.numbside, gdat.numbtime[o])) + cntpback
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
            
    return cntpmodl


def retr_llik(gdat, para):
    
    coef = para[:gdat.numbcoef]
    cntpback = para[gdat.numbcoef]

    if gdat.typepsfn == 'lion':
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
    cntpmodl = retr_cntpmodl(gdat, gdat.xposfitt, gdat.yposfitt, flux, cntpback, o, coef=coef)
    
    chi2 = np.sum((gdat.cntpdatatmed - cntpmodl)**2 / gdat.cntpdatatmed)
    llik = -0.5 * chi2
    
    return llik


#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def retr_mlikregr(cntpdata, matrdesi, vari):
    """Return the maximum likelihood estimate for linear regression."""

    varitdim = np.diag(vari.flatten())
    covafittflux = np.linalg.inv(np.matmul(np.matmul(matrdesi.T, np.linalg.inv(varitdim)), matrdesi))
    mlikfittflux = np.matmul(np.matmul(np.matmul(covafittflux, matrdesi.T), np.linalg.inv(varitdim)), cntpdata.flatten())
    
    return mlikfittflux, covafittflux


def retr_strgsave(gdat, strgsecc, strgoffs, typecade):
    
    if strgoffs == 'of11':
        strgoffstemp = ''
    else:
        strgoffstemp = '_' + strgoffs
    if gdat.maxmnumbstar is None:
        strgmaxmnumbstar = ''
    else:
        strgmaxmnumbstar = '_m%03d' % gdat.maxmnumbstar
    strgsave = '_%s_%s%s_%s%s' % (gdat.strgcnfg, strgsecc, strgoffstemp, typecade, strgmaxmnumbstar)

    return strgsave


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
         
         # selected TESS sectors
         listtsecsele=None, \
    
         # Boolean flag to use Target Pixel Files (TPFs) at the highest cadence whenever possible
         booltpxflygo=True, \

         # list of TESS sectors for which 
         #listtsec2min=None, \

         ## mock data
         truesourtype='dwar', \
         ### number of time bins
         numbtime=None, \
        
         # target
         ## a string to be used to search MAST for the target
         strgmast=None, \
         ## TIC ID of the object of interest
         ticitarg=None, \
         ## RA of the object of interest
         rasctarg=None, \
         ## DEC of the object of interest
         decltarg=None, \
         # TOI ID
         toiitarg=None, \

         # path for the target
         pathtarg=None, \
        
         ## number of pixels on a side to cut out
         numbside=11, \
        
         ## mask
         ### Boolean flag to put a cut on quality flag
         boolcuttqual=True, \
         ## limits of time between which the quality mask will be ignored
         limttimeignoqual=None, \

         ### masking region
         epocmask=None, \
         perimask=None, \
         duramask=None, \
         
         # processing
         ## Boolean flag to turn on CBV detrending
         boolcbvs=True, \

         # string indicating the cluster of targets
         strgclus=None, \

         # visualization
         ## Boolean flag to make relative flux plots
         boolplotrflx=True, \
         ## Boolean flag to make image plots
         boolplotcntp=False, \
         ## Boolean flag to plot the quaternions
         boolplotquat=True, \
         ## Boolean flag to make an animation
         boolanim=False, \
         ## Boolean flag to include all time bins in the animation
         boolanimframtotl=True, \
        
         ## Boolean flag to plot the histogram of the number of counts
         boolplothhistcntp=False, \

         # plot extensions
         strgplotextn='png', \
        
         # diagnostics
         booldiagmode=True, \
        
         # Boolean flag to calculate the contamination ratio
         boolcalcconr = False, \

         # model
         ## factor by which to rebin the data along time
         facttimerebn=1., \
         ## maximum number stars in the fit
         maxmnumbstar=1, \
        
         # maximum delta magnitude of neighbor sources to be included in the model
         maxmdmag=4., \
        
         ## PSF evaluation
         typepsfn='lion', \
         psfnshaptype='gfrefinf', \

         catlextr=None, \
         lablcatlextr=None, \
    
         # Boolean flag to repeat the fit, putting the target to offset locations
         boolfittoffs=False, \

         ## post-process
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
         #listtypecntpscal=['self', 'asnh'], \
         listtypecntpscal=['self'], \
         
         **args, \
        ):
   
    # start the timer
    timeinittotl = timemodu.time()
    
    # construct global object
    gdat = tdpy.util.gdatstrt()
    
    # copy unnamed inputs to the global object
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # copy named arguments to the global object
    for strg, valu in args.items():
        setattr(gdat, strg, valu)

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print('lygos initialized at %s...' % gdat.strgtimestmp)
    # paths
    gdat.pathbase = os.environ['LYGOS_DATA_PATH'] + '/'

    ## ensure target identifiers are not conflicting
    if gdat.ticitarg is None and gdat.strgmast is None and gdat.toiitarg is None and (gdat.rasctarg is None or gdat.decltarg is None):
        raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) should be provided.')
    if gdat.ticitarg is not None and (gdat.strgmast is not None or gdat.toiitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None):
        raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) should be provided.')
    if gdat.strgmast is not None and (gdat.ticitarg is not None or gdat.toiitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None):
        raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) should be provided.')
    if gdat.toiitarg is not None and (gdat.strgmast is not None or gdat.ticitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None):
        raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) should be provided.')
    if gdat.strgmast is not None and (gdat.ticitarg is not None or gdat.toiitarg is not None or gdat.rasctarg is not None or gdat.decltarg is not None):
        raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) should be provided.')

    gdat.pathtoii = gdat.pathbase + 'data/exofop_tess_tois.csv'
    print('Reading from %s...' % gdat.pathtoii)
    objtexof = pd.read_csv(gdat.pathtoii, skiprows=0)

    # conversion factors
    gdat.factrsrj, gdat.factrjre, gdat.factrsre, gdat.factmsmj, gdat.factmjme, gdat.factmsme, gdat.factaurs = ephesus.util.retr_factconv()

    # determine target identifiers
    if gdat.ticitarg is not None:
        strgtargtype = 'tici'
        print('A TIC ID was provided as target identifier.')
        indx = np.where(objtexof['TIC ID'].values == gdat.ticitarg)[0]
        if indx.size > 0:
            gdat.toiitarg = int(str(objtexof['TOI'][indx[0]]).split('.')[0])
            print('Matched the input TIC ID with TOI %d.' % gdat.toiitarg)
        gdat.strgmast = 'TIC %d' % gdat.ticitarg

    elif gdat.toiitarg is not None:
        strgtargtype = 'toii'
        print('A TOI number (%d) was provided as target identifier.' % gdat.toiitarg)
        # determine TIC ID
        gdat.strgtoiibase = str(gdat.toiitarg)
        indx = []
        for k, strg in enumerate(objtexof['TOI']):
            if str(strg).split('.')[0] == gdat.strgtoiibase:
                indx.append(k)
        indx = np.array(indx)
        if indx.size == 0:
            print('Did not find the TOI in the ExoFOP-TESS TOI list.')
            print('objtexof[TOI]')
            summgene(objtexof['TOI'])
            raise Exception('')
        gdat.ticitarg = objtexof['TIC ID'].values[indx[0]]

        gdat.strgmast = 'TIC %d' % gdat.ticitarg
    
    elif gdat.strgmast is not None:
        strgtargtype = 'mast'
        print('A MAST key (%s) was provided as target identifier.' % gdat.strgmast)

    elif gdat.rasctarg is not None and gdat.decltarg is not None:
        strgtargtype = 'posi'
        print('RA and DEC (%g %g) are provided as target identifier.' % (gdat.rasctarg, gdat.decltarg))
        gdat.strgmast = '%g %g' % (gdat.rasctarg, gdat.decltarg)
    
    print('strgtargtype')
    print(strgtargtype)
    print('gdat.strgmast')
    print(gdat.strgmast)
    print('gdat.toiitarg')
    print(gdat.toiitarg)
    # temp -- check that the closest TIC to a given TIC is itself
    catalogData = astroquery.mast.Catalogs.query_region(gdat.strgmast, radius='200s', catalog="TIC")
    print('Found %d TIC sources within 200 as.' % len(catalogData))
    
    gdat.ticitarg = int(catalogData[0]['ID'])
    gdat.rasctarg = catalogData[0]['ra']
    gdat.decltarg = catalogData[0]['dec']
    gdat.tmagtarg = catalogData[0]['Tmag']
    
    print('gdat.ticitarg')
    print(gdat.ticitarg)
    print('gdat.rasctarg')
    print(gdat.rasctarg)
    print('gdat.decltarg')
    print(gdat.decltarg)
    print('gdat.tmagtarg')
    print(gdat.tmagtarg)
    gdat.boolmastavai = True
    
    if gdat.labltarg is None:
        if strgtargtype == 'mast':
            gdat.labltarg = gdat.strgmast
        if strgtargtype == 'toii':
            gdat.labltarg = 'TOI %d' % gdat.toiitarg
        if strgtargtype == 'tici':
            gdat.labltarg = 'TIC %d' % gdat.ticitarg
        if strgtargtype == 'posi':
            gdat.labltarg = 'RA=%.4g, DEC=%.4g' % (gdat.rasctarg, gdat.decltarg)
    
    if gdat.strgtarg is None:
        gdat.strgtarg = ''.join(gdat.labltarg.split(' '))
    
    print('Target label: %s' % gdat.labltarg) 
    print('Output folder name: %s' % gdat.strgtarg) 
    print('RA and DEC: %g %g' % (gdat.rasctarg, gdat.decltarg))
    if strgtargtype == 'tici' or strgtargtype == 'mast':
        print('Tmag: %g' % gdat.tmagtarg)
   
    print('PSF model: %s' % gdat.typepsfn)
    if gdat.typepsfn == 'ontf':
        print('PSF model shape:')
        print(gdat.psfnshaptype)
    
    gdat.pathimag = gdat.pathbase + 'imag/'
    gdat.pathdata = gdat.pathbase + 'data/'
    
    if gdat.strgclus is None:
        gdat.strgclus = ''
    else:
        gdat.strgclus += '/'
    if gdat.pathtarg is None:
        gdat.pathtarg = gdat.pathbase + '%s%s/' % (gdat.strgclus, gdat.strgtarg)
    gdat.pathdatatarg = gdat.pathtarg + 'data/'
    gdat.pathimagtarg = gdat.pathtarg + 'imag/'
    gdat.pathclus = gdat.pathbase + '%s' % gdat.strgclus
    gdat.pathdataclus = gdat.pathclus + 'data/'
    gdat.pathimagclus = gdat.pathclus + 'imag/'
    
    os.system('mkdir -p %s' % gdat.pathimag)
    os.system('mkdir -p %s' % gdat.pathdata)
    os.system('mkdir -p %s' % gdat.pathimagtarg)
    os.system('mkdir -p %s' % gdat.pathdatatarg)
    os.system('mkdir -p %s' % gdat.pathimagclus)
    os.system('mkdir -p %s' % gdat.pathdataclus)
    
    # create a separate folder to place the PSF fit output
    if gdat.typepsfn == 'ontf':
        gdat.pathimagtargpsfn = gdat.pathimagtarg + 'psfn/'
        os.system('mkdir -p %s' % gdat.pathimagtargpsfn)
   
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
        listhdundatatemp = astroquery.mast.Tesscut.get_cutouts(objtskyy, gdat.numbside)
        
        print('gdat.rasctarg')
        print(gdat.rasctarg)
        print('gdat.decltarg')
        print(gdat.decltarg)
        
        ## parse cutout HDUs
        gdat.listtsecffim = []
        gdat.listtcamffim = []
        gdat.listtccdffim = []
        gdat.listhdundataffim = []
        for hdundata in listhdundatatemp:
            tsec = hdundata[0].header['SECTOR']
            gdat.listtsecffim.append(tsec)
            gdat.listtcamffim.append(hdundata[0].header['CAMERA'])
            gdat.listtccdffim.append(hdundata[0].header['CCD'])
            gdat.listhdundataffim.append(hdundata)
        gdat.listtsecffim = np.array(gdat.listtsecffim)
        
        print('temp')
        if gdat.rasctarg == 122.989831564958:
            gdat.listtsecffim = np.array([7, 34])
            gdat.listtcamffim = np.array([1, 1])
            gdat.listtccdffim = np.array([3, 4])
    
        print('gdat.listtsecffim')
        print(gdat.listtsecffim)
        print('gdat.listtcamffim')
        print(gdat.listtcamffim)
        print('gdat.listtccdffim')
        print(gdat.listtccdffim)
        
        gdat.listtsecspoc = []
        gdat.listtcamspoc = []
        gdat.listtccdspoc = []
        if booltpxflygo:
            
            # get the list of sectors for which TESS SPOC TPFs are available
            print('Retrieving the list of available TESS sectors for which there is SPOC TPF data...')
            # get observation tables
            listtablobsv = ephesus.util.retr_listtablobsv(gdat.strgmast)
            listprodspoc = []
            for k, tablobsv in enumerate(listtablobsv):
                
                listprodspoctemp = astroquery.mast.Observations.get_product_list(tablobsv)
                
                if listtablobsv['distance'][k] > 0:
                    continue

                strgdesc = 'Light curves'
                strgdesc = 'Target pixel files'
                strgdesc = ['Target pixel files', 'Light curves']
                listprodspoctemp = astroquery.mast.Observations.filter_products(listprodspoctemp, description=strgdesc)
                for a in range(len(listprodspoctemp)):
                    boolfasttemp = listprodspoctemp[a]['obs_id'].endswith('fast')
                    if not boolfasttemp and listprodspoctemp[a]['description'] == 'Target pixel files':
                        tsec = int(listprodspoctemp[a]['obs_id'].split('-')[1][1:])
                        gdat.listtsecspoc.append(tsec) 
                        
                        print('temp')
                        gdat.listtcamspoc.append(-1) 
                        gdat.listtccdspoc.append(-1) 
                        
                        listprodspoc.append(listprodspoctemp)
            
            gdat.listtsecspoc = np.array(gdat.listtsecspoc)
            gdat.listtcamspoc = np.array(gdat.listtcamspoc)
            gdat.listtccdspoc = np.array(gdat.listtccdspoc)
            
            indx = np.argsort(gdat.listtsecspoc)
            gdat.listtsecspoc = gdat.listtsecspoc[indx]
            gdat.listtcamspoc = gdat.listtcamspoc[indx]
            gdat.listtccdspoc = gdat.listtccdspoc[indx]
            
            print('gdat.listtsecspoc')
            print(gdat.listtsecspoc)
        
            gdat.numbtsecspoc = gdat.listtsecspoc.size

        if len(gdat.listtsecspoc) > 0:
            gdat.indxtsecspoc = np.arange(gdat.numbtsecspoc)
            
            # download data from MAST
            os.system('mkdir -p %s' % gdat.pathdatatarg)
        
            print('Downloading SPOC data products...')
            
            listhdundataspoc = [[] for o in gdat.indxtsecspoc]
            listpathdownspoc = []

            listpathdownspoclcur = []
            listpathdownspoctpxf = []
            for k in range(len(listprodspoc)):
                manifest = astroquery.mast.Observations.download_products(listprodspoc[k], download_dir=gdat.pathdatatarg)
                listpathdownspoclcur.append(manifest['Local Path'][0])
                listpathdownspoctpxf.append(manifest['Local Path'][1])

            ## make sure the list of paths to sector files are time-sorted
            listpathdownspoc.sort()
            listpathdownspoclcur.sort()
            listpathdownspoctpxf.sort()
            
            ## read SPOC TPFs
            for o in gdat.indxtsecspoc:
                listhdundataspoc[o], gdat.listtsecspoc[o], gdat.listtcamspoc[o], \
                                                                gdat.listtccdspoc[o] = ephesus.util.read_tesskplr_file(listpathdownspoctpxf[o])

            print('gdat.listtsecspoc')
            print(gdat.listtsecspoc)
            print('gdat.listtcamspoc')
            print(gdat.listtcamspoc)
            print('gdat.listtccdspoc')
            print(gdat.listtccdspoc)
        
        # merge SPOC TPF and FFI sector lists
        gdat.listtsec = []
        gdat.listtcam = []
        gdat.listtccd = []
        gdat.listtsecconc = np.unique(np.concatenate((gdat.listtsecffim, gdat.listtsecspoc)))
        
        if gdat.listtsecsele is not None:
            gdat.listtsecconc = [tsec for tsec in gdat.listtsecconc if tsec in gdat.listtsecsele]
        
            print('gdat.listtsecsele')
            print(gdat.listtsecsele)
        
        for k in range(len(gdat.listtsecconc)):
            indx = np.where(gdat.listtsecspoc == gdat.listtsecconc[k])[0]
            if indx.size > 0:
                gdat.listtsec.append(gdat.listtsecspoc[indx[0]])
                gdat.listtcam.append(gdat.listtcamspoc[indx[0]])
                gdat.listtccd.append(gdat.listtccdspoc[indx[0]])
            else:
                indx = np.where(gdat.listtsecffim == gdat.listtsecconc[k])[0]
                gdat.listtsec.append(gdat.listtsecffim[indx[0]])
                gdat.listtcam.append(gdat.listtcamffim[indx[0]])
                gdat.listtccd.append(gdat.listtccdffim[indx[0]])
        gdat.listtsec = np.array(gdat.listtsec)
        gdat.listtcam = np.array(gdat.listtcam)
        gdat.listtccd = np.array(gdat.listtccd)
        
        print('gdat.listtsec')
        print(gdat.listtsec)
        gdat.numbtsec = gdat.listtsec.size
        gdat.indxtsec = np.arange(gdat.numbtsec)
        
        # determine whether sectors have TPFs
        gdat.booltpxf = ephesus.util.retr_booltpxf(gdat.listtsec, gdat.listtsecspoc)
        print('gdat.booltpxf')
        print(gdat.booltpxf)

        gdat.listhdundata = [[] for o in gdat.indxtsec]
        for o in gdat.indxtsec:
            if gdat.booltpxf[o]:
                indxtsectemp = np.where(gdat.listtsec[o] == gdat.listtsecspoc)[0][0]
                gdat.listhdundata[o] = listhdundataspoc[indxtsectemp]
            else:
                indxtsectemp = np.where(gdat.listtsec[o] == gdat.listtsecffim)[0][0]
                
                #print('o')
                #print(o)
                #print('gdat.listtsec[o]')
                #print(gdat.listtsec[o])
                #print('gdat.listtsecffim')
                #print(gdat.listtsecffim)
                #print('indxtsectemp')
                #print(indxtsectemp)
                #print('gdat.listhdundataffim')
                #print(gdat.listhdundataffim)
                #print('')
                
                gdat.listhdundata[o] = gdat.listhdundataffim[indxtsectemp]

        gdat.typecade = np.zeros_like(gdat.booltpxf, dtype=object)
        for o, tsec in enumerate(gdat.listtsec):
            if gdat.booltpxf[o]:
                # temp does not work with 20sc
                gdat.typecade[o] = '2min'
            else:
                if tsec > 26:
                    gdat.typecade[o] = '10mn'
                else:
                    gdat.typecade[o] = '30mn'
        print('gdat.typecade')
        print(gdat.typecade)

        if gdat.numbtsec == 0:
            print('No data have been retrieved.')
        else:
            if gdat.numbtsec == 1:
                strgtemp = ''
            else:
                strgtemp = 's'
            print('%d sector%s of data retrieved.' % (gdat.numbtsec, strgtemp))
    
        # get the WCS object
        gdat.listobjtwcss = []
        for o, tsec in enumerate(gdat.listtsec):
            gdat.listobjtwcss.append(astropy.wcs.WCS(gdat.listhdundata[o][2].header))
    
    gdat.dictoutp = dict()
    gdat.dictoutp['listtsec'] = gdat.listtsec
    gdat.dictoutp['listtcam'] = gdat.listtcam
    gdat.dictoutp['listtccd'] = gdat.listtccd
    if len(gdat.listtsec) != len(gdat.listtcam):
        print('gdat.listtsec')
        print(gdat.listtsec)
        print('gdat.listtcam')
        print(gdat.listtcam)
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
        strgsave = retr_strgsave(gdat, strgsecc, strgoffs, gdat.typecade[o])
        pathsaverflxtarg = gdat.pathdatatarg + 'rflxtarg' + strgsave + '.csv'
        gdat.dictoutp['pathsaverflxtargsc%02d' % gdat.listtsec[o]] = pathsaverflxtarg
        if os.path.exists(pathsaverflxtarg):
            print('Run previously completed...')
            gdat.listarry = [[] for o in gdat.indxtsec]
            for o in gdat.indxtsec:
                gdat.listarry[o] = np.loadtxt(pathsaverflxtarg)
        
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
    if gdat.typepsfn == 'lion':
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
    gdat.pathsavemetaglob = gdat.pathdatatarg + 'metaglob.csv'
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
            arry, indxtimequalgood, indxtimenanngood, tsecrefr, tcam, tccd = ephesus.util.read_tesskplr_file(listpathdownspoclcur[cntr], \
                                                                                                                                strgtype='PDCSAP_FLUX')
            gdat.refrtime[o][0] = arry[:, 0]
            gdat.refrrflx[o][0] = arry[:, 1]
            gdat.stdvrefrrflx[o][0] = arry[:, 2]
            gdat.boolrefr[o] = True
            cntr += 1
        else:
            gdat.boolrefr[o] = False
        
    gdat.listtime = [[] for o in gdat.indxtsec]
    gdat.indxtime = [[] for o in gdat.indxtsec]
    
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
            gdat.listtime[o] = np.linspace(0, gdat.numbtime[o] - 1, gdat.numbtime[o])
            indxtimedatagood = np.arange(gdat.numbtime[o])
        else:
            
            # read the FITS files
            ## time
            #print(gdat.listhdundata[o][1].data.names)
            gdat.listtime[o] = gdat.listhdundata[o][1].data['TIME'] + 2457000
            diff = gdat.listtime[o][1:] - gdat.listtime[o][:-1]
            
            ## count per pixel
            gdat.cntpdata = gdat.timeexpo * (gdat.listhdundata[o][1].data['FLUX'] + \
                                             gdat.listhdundata[o][1].data['FLUX_BKG']).swapaxes(0, 2).swapaxes(0, 1)
            
            if gdat.booltpxf[o]:
                gdat.numbside = gdat.cntpdata.shape[1]
            ## filter good times
            
            booldatagood = np.isfinite(gdat.listtime[o])
            if gdat.boolcuttqual:
                print('Masking bad data with quality flags...')
                booldatagood = booldatagood & (gdat.listhdundata[o][1].data['QUALITY'] == 0)
            print('booldatagood')
            summgene(booldatagood)
            if limttimeignoqual is not None:
                print('Ignoring the quality mask between %g and %g...' % (limttimeignoqual[0], limttimeignoqual[1]))
                booldatagood = booldatagood | ((limttimeignoqual[0] < gdat.listtime[o]) & (gdat.listtime[o] < limttimeignoqual[1]))
            print('booldatagood')
            summgene(booldatagood)
            indxtimedatagood = np.where(booldatagood)[0]
            fracgood = 100. * float(indxtimedatagood.size) / gdat.listtime[o].size
            print('Fraction of times: %.4g percent' % fracgood)
            if indxtimedatagood.size == 0:
                print('No good data found.')
                return gdat.dictoutp
    
        gdat.listtime[o] = gdat.listtime[o][indxtimedatagood]
        gdat.cntpdata = gdat.cntpdata[:, :, indxtimedatagood]
        
        gdat.numbtime[o] = gdat.listtime[o].size
        print('gdat.numbtime[o]')
        print(gdat.numbtime[o])
        gdat.indxtime[o] = np.arange(gdat.numbtime[o])
        
        gdat.gdatlion.numbtime = gdat.numbtime[o]
        gdat.gdatlion.indxtime = gdat.indxtime[o]
        
        arrytemp = np.linspace(0., float(gdat.numbside - 1), gdat.numbside)
        gdat.xposimag, gdat.yposimag = np.meshgrid(arrytemp, arrytemp)
        
        gdat.pathcbvs = gdat.pathdata + 'cbvs/'
        print('gdat.boolcbvs')
        print(gdat.boolcbvs)
        print('gdat.pathcbvs')
        print(gdat.pathcbvs)
        #if len(os.listdir(gdat.pathcbvs)) == 0 and 
        if gdat.boolcbvs:
            print('o')
            print(o)
            print('gdat.listtsec')
            print(gdat.listtsec)
            path = gdat.pathcbvs + fnmatch.filter(os.listdir(gdat.pathcbvs), 'tess*-s%04d-%d-%d-*-s_cbv.fits' % (gdat.listtsec[o], \
                                                                                                            gdat.listtcam[o], gdat.listtccd[o]))[0]
            print('path')
            print(path)
            listhdun = astropy.io.fits.open(path)
            listhdun.info()
            timecbvs = listhdun[1].data['TIME']
            gdat.numbcbvs = 5
            gdat.indxcbvs = np.arange(gdat.numbcbvs)
            timecbvstemp = listhdun[1].data['TIME'] + 2457000
            cbvsraww = np.empty((timecbvstemp.size, gdat.numbcbvs))
            gdat.cbvs = np.empty((gdat.numbtime[o], gdat.numbcbvs))
            for i in gdat.indxcbvs:
                cbvsraww[:, i] = listhdun[1].data['VECTOR_%i' % (i + 1)]
                gdat.cbvs[:, i] = scipy.interpolate.interp1d(timecbvstemp, cbvsraww[:, i])(gdat.listtime[o])
            gdat.cbvstmpt = np.ones((gdat.numbtime[o], gdat.numbcbvs + 1))
            gdat.cbvstmpt[:, :-1] = gdat.cbvs
            if gdat.boolplotquat:
                # plot centroid
                for a in range(2):
                    if a == 0:
                        numbplot = 2
                    else:
                        numbplot = gdat.numbcbvs

                    for k in range(numbplot):
                        figr, axis = plt.subplots(figsize=(12, 4))
                        
                        if a == 0:
                            if k == 0:
                                strgyaxi = 'x'
                                posi = gdat.xposimag
                            else:
                                strgyaxi = 'y'
                                posi = gdat.yposimag
                            strgplot = 'cent'
                            temp = np.sum(posi[None, :, :, None] * gdat.cntpdata, axis=(0, 1, 2)) / np.sum(gdat.cntpdata, axis=(0, 1, 2))
                        else:
                            temp = gdat.cbvs[:, k]
                            strgyaxi = 'CBV$_{%d}$' % k
                            posi = gdat.xposimag
                            strgplot = 'cbvs'
                        axis.plot(gdat.listtime[o], temp, ls='', marker='.', ms=1)
                        axis.set_ylabel('%s' % strgyaxi)
                        axis.set_xlabel('Time [BJD]')
                        path = gdat.pathimagtarg + '%s_%s_%02d.%s' % (strgplot, strgsecc, k, gdat.strgplotextn)
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
                magttrue[0, :, 0] = 10. / (np.exp(10. * (gdat.listtime[o] / gdat.numbtime[o] - 0.5)) + 1.) + 16.
            fluxtrue = 10**(0.4 * (20.4 - magttrue)) * gdat.timeexpo
            indxtimetran = []
            for t in gdat.indxtime[o]:
                if gdat.listtime[o][t] % 4. < 0.5:
                    indxtimetran.append(t)
            indxtimetran = np.array(indxtimetran)
            magttrue[0, indxtimetran, 0] *= 1e-3
            xpostrue = gdat.numbside * np.random.random(numbstar)
            ypostrue = gdat.numbside * np.random.random(numbstar)
            xpostrue[0] = gdat.numbside / 2. 
            ypostrue[0] = gdat.numbside / 2. 
            cntpbacktrue = np.zeros(gdat.numbtime[o]) + 1800 * 100.
            cntpdatatemp = retr_cntpmodl(gdat, coef, xpostrue, ypostrue, fluxtrue, cntpbacktrue, o)
            gdat.cntpdata = np.random.poisson(cntpdatatemp).astype(float)
        
            gdat.refrtime[o] = gdat.listtime[o]
        
        if len(os.listdir(gdat.pathcbvs)) == 0 and gdat.boolplotquat:
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
                quat[:, k-1] = scipy.interpolate.interp1d(dataquat['TIME']+2457000,  dataquat[strg], fill_value=0, bounds_error=False)(gdat.listtime[o])
                minm = np.percentile(dataquat[strg], 0.05)
                maxm = np.percentile(dataquat[strg], 99.95)
                axis[k-1].plot(dataquat['TIME']+2457000, dataquat[strg], ls='', marker='.', ms=1)
                axis[k-1].set_ylim([minm, maxm])
                axis[k-1].set_ylabel('$Q_{%d}$' % k)
            axis[2].set_xlabel('Time [BJD]')
            path = gdat.pathimagtarg + 'quat_sc%02d.%s' % (gdat.listtsec[o], gdat.strgplotextn)
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
                catlrefrposi = gdat.listobjtwcss[o].all_world2pix(catlrefrskyy, 0)
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
            if gdat.typepsfn == 'ontf':
                gdat.cntsfitt = np.concatenate((np.array([0.]), gdat.catlrefrfilt[0]['cnts']))
                print('Do not know what counts to assume for the target (for the on-the-fly PSF fit). Assuming 0!')
        else:
            gdat.rascfitt = gdat.catlrefrfilt[0]['rasc']
            gdat.declfitt = gdat.catlrefrfilt[0]['decl']
            if gdat.typepsfn == 'ontf':
                gdat.cntsfitt = gdat.catlrefrfilt[0]['cnts']
        
        skyyfitttemp = np.empty((gdat.rascfitt.size, 2))
        skyyfitttemp[:, 0] = gdat.rascfitt
        skyyfitttemp[:, 1] = gdat.declfitt
        if gdat.rascfitt.size == 0:
            raise Exception('')
        # transform sky coordinates into dedector coordinates and filter
        posifitttemp = gdat.listobjtwcss[o].all_world2pix(skyyfitttemp, 0)
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

        if gdat.typepsfn == 'lion':
            gdat.gdatlion.numbtime = 1
            gdat.gdatlion.indxtime = [0]
        
        # number of components, 1 for background, 3 for quaternions
        gdat.numbcomp = gdat.numbstar + 1# + 3
        gdat.indxcomp = np.arange(gdat.numbcomp)

        gdat.stdvfittflux = np.empty((gdat.numbtime[o], gdat.numbcomp))
        
        if gdat.boolcalcconr and not (gdat.ticitarg is not None):
            raise Exception

        # fit for the PSF
        if gdat.typepsfn == 'ontf':
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
                    nameplot = '%scntpdatatmed' % (gdat.pathimagtargpsfn)
                    strgtitl = gdat.strgtitlcntpplot
                    plot_cntp(gdat, gdat.cntpdatatmed, o, typecntpscal, nameplotcntp, strgsave, strgtitl=strgtitl)

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
            
            if gdat.psfnshaptype == 'gaus' or gdat.typepsfn == 'gfre' or gdat.typepsfn == 'pfre':
                # solve for the maximum likelihood fluxes for the median image
                matrdesi = np.ones((gdat.numbpixl, gdat.numbstar + 1))
                for k in np.arange(gdat.numbstar):
                    matrdesi[:, k] = retr_cntpmodl(gdat, coef, gdat.xposfitt[k, None], gdat.yposfitt[k, None], np.array([[1.]]), np.array([0.]), o).flatten()
                matrdesi[:, gdat.numbstar] = 1.
                gdat.mlikfittfluxmedi, gdat.covafittfluxmedi = retr_mlikregr(gdat.cntpdatatmed, matrdesi, gdat.cntpdatatmed)
            
            # background
            listminmpara[gdat.numbcoef] = 0.
            listmaxmpara[gdat.numbcoef] = np.amax(gdat.cntpdatatmed)#abs(gdat.mlikfittfluxmedi[-1]) * 2.
            
            if gdat.psfnshaptype == 'gfreffix' or gdat.typepsfn =='gfrefinf':
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
            strgsaveextnmcmc = gdat.pathdatatarg + gdat.psfnshaptype + '_' + gdat.strgcnfg + '.txt'
            parapost = tdpy.mcmc.samp(gdat, gdat.pathimagtargpsfn, numbsampwalk, numbsampburnwalk, numbsampburnwalkseco, retr_llik, \
                                            listlablpara, listscalpara, listminmpara, listmaxmpara, listmeangauspara, liststdvgauspara, \
                                                numbdata, strgextn=strgextn, strgplotextn=strgplotextn, strgsaveextn=strgsaveextnmcmc)
            
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
                gdat.cntpmodlpsfn = retr_cntpmodl(gdat, coefmedi, xpossour, ypossour, flux, cntpback, o, verbtype=1)
                nameplot = '%scntpmodlpsfnmedi' % (gdat.pathimagtargpsfn)
                strgtitl = gdat.strgtitlcntpplot
                for typecntpscal in gdat.listtypecntpscal:
                    plot_cntp(gdat, gdat.cntpmodlpsfn[:, :, 0], o, typecntpscal, nameplotcntp, strgsave, strgtitl=strgtitl, boolanno=False)

                # plot the posterior median image model
                if gdat.psfnshaptype == 'gfreffix':
                    flux = gdat.cntsfitt * amplmedi
                elif gdat.psfnshaptype == 'gfrefinf':
                    flux = gdat.cntsfitt * ampltotlmedi * amplrelamedi
                elif gdat.psfnshaptype == 'gfreffre':
                    flux = fluxmedi
                gdat.cntpmodlpsfn = retr_cntpmodl(gdat, coefmedi, gdat.xposfitt, gdat.yposfitt, flux, cntpbackmedi, o)
                
                nameplot = '%scntpmodlmedi' % (gdat.pathimagtargpsfn)
                strgtitl = gdat.strgtitlcntpplot
                for typecntpscal in gdat.listtypecntpscal:
                    plot_cntp(gdat, gdat.cntpmodlpsfn[:, :, 0], o, typecntpscal, nameplotcntp, strgsave, strgtitl=strgtitl)

                # plot the posterior median residual
                nameplot = '%scntpresimedi' % (gdat.pathimagtargpsfn)
                strgtitl = gdat.strgtitlcntpplot
                for typecntpscal in gdat.listtypecntpscal:
                    plot_cntp(gdat, gdat.cntpdatatmed - gdat.cntpmodlpsfn[:, :, 0], o, typecntpscal, nameplotcntp, strgsave, \
                                                                                                                    strgtitl=strgtitl, boolresi=True)

        if gdat.boolanim:
            if gdat.boolanimframtotl:
                numbplotanim = gdat.numbtime[o]
            else:
                numbplotanim = 10
            # time indices to be included in the animation
            gdat.indxtimeanim = np.linspace(0., gdat.numbtime[o] - 1., numbplotanim).astype(int)
            print('gdat.numbtime')
            print(gdat.numbtime)
            print('numbplotanim')
            print(numbplotanim)
            print('gdat.indxtimeanim')
            summgene(gdat.indxtimeanim)
            print('gdat.listtime[o]')
            summgene(gdat.listtime[o])
            print('')
            # get time string
            objttime = astropy.time.Time(gdat.listtime[o], format='jd', scale='utc', out_subfmt='date_hm')
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
                strgsave = retr_strgsave(gdat, strgsecc, strgoffs, gdat.typecade[o])
                pathsaverflx = gdat.pathdatatarg + 'rflx' + strgsave + '.csv'
                pathsaverflxtarg = gdat.pathdatatarg + 'rflxtarg' + strgsave + '.csv'
                pathsaverflxtargbdtr = gdat.pathdatatarg + 'rflxtargbdtr' + strgsave + '.csv'
                pathsavemeta = gdat.pathdatatarg + 'meta' + strgsave + '.csv'
        
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
                    path = gdat.pathimagtarg + 'histcntpdata_%s.%s' % (strgsave, gdat.strgplotextn)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
        
                # introduce the positional offset
                xpostemp = np.copy(gdat.xposfitt)
                ypostemp = np.copy(gdat.yposfitt)
                xpostemp[0] = gdat.xposfitt[0] + gdat.listoffsxpos[x]
                ypostemp[0] = gdat.yposfitt[0] + gdat.listoffsypos[y]
                
                if not os.path.exists(pathsaverflx):
                    
                    timeinit = timemodu.time()
                    gdat.covafittflux = np.empty((gdat.numbtime[o], gdat.numbstar + 1, gdat.numbstar + 1))
                    gdat.mlikfittflux = np.empty((gdat.numbtime[o], gdat.numbstar + 1))
                    matrdesi = np.ones((gdat.numbpixl, gdat.numbstar + 1))
                    
                    print('Solving for the best-fit raw light curves of the sources...')
                    coef = None
                    fluxtemp = np.ones((1, gdat.numbstar))
                    cntptemp = np.zeros(1)
                    for k in np.arange(gdat.numbstar):
                        matrdesi[:, k] = retr_cntpmodl(gdat, xpostemp[k, None], ypostemp[k, None], fluxtemp[:, k, None], cntptemp, o, coef=coef).flatten()
                            
                    # solve the linear system
                    for t in gdat.indxtime[o]:
                        
                        if t == 0:
                            print('gdat.cntpdata[:, :, t]')
                            summgene(gdat.cntpdata[:, :, t])
                            print('matrdesi')
                            summgene(matrdesi)
                            print('gdat.vari[:, :, t]')
                            summgene(gdat.vari[:, :, t])

                        gdat.mlikfittflux[t, :], gdat.covafittflux[t, :, :] = retr_mlikregr(gdat.cntpdata[:, :, t], matrdesi, gdat.vari[:, :, t])

                    if not np.isfinite(gdat.covafittflux).all():
                        indxbaddpixl = (~np.isfinite(gdat.covafittflux)).any(0)
                        print('indxbaddpixl')
                        print(indxbaddpixl)
                        indxbaddtime = (~np.isfinite(gdat.covafittflux)).any(1).any(1)
                        print('indxbaddtime')
                        summgene(indxbaddtime)
                        print('gdat.vari')
                        summgene(gdat.vari)
                        print('gdat.cntpdata')
                        summgene(gdat.cntpdata)
                        print('matrdesi')
                        summgene(matrdesi)
                        #raise Exception('')

                    for k in gdat.indxcomp:
                        gdat.stdvfittflux[:, k] = np.sqrt(gdat.covafittflux[:, k, k])
                        
                        if not np.isfinite(gdat.stdvfittflux[:, k]).all():
                            print('temp: error went NaN because of negative covariance. Reseting error to 1e-2')
                            gdat.stdvfittflux[:, k] = 1e-2
                            print('k')
                            print(k)
                            print('gdat.covafittflux[:, k, k]')
                            summgene(gdat.covafittflux[:, k, k])
                            print('np.isfinite(gdat.stdvfittflux[:, k])')
                            summgene(np.isfinite(gdat.stdvfittflux[:, k]))
                            #raise Exception('')
                    
                    gdat.varifittflux = gdat.stdvfittflux**2
        
                    if gdat.boolcbvs:
                        # light curve before CBV-detrending
                        gdat.mlikfittfluxraww = np.copy(gdat.mlikfittflux[:, 0])

                        # subtract CBVs
                        print('Solving for the detrended target light curve using the CBVs and the raw light curve...')
                        print('gdat.mlikfittflux[:, 0]')
                        summgene(gdat.mlikfittflux[:, 0])
                        print('gdat.varifittflux[:, 0]')
                        summgene(gdat.varifittflux[:, 0])
                        #for k in range(gdat.numbcbvs+1):
                        #    print('gdat.cbvstmpt[:, k]')
                        #    summgene(gdat.cbvstmpt[:, k])
                        gdat.mlikamplcbvs, gdat.covaamplcbvs = retr_mlikregr(gdat.mlikfittflux[:, 0], gdat.cbvstmpt, gdat.varifittflux[:, 0])
                        print('gdat.mlikamplcbvs')
                        print(gdat.mlikamplcbvs)
                        rflxcbvs = gdat.mlikamplcbvs[None, :] * gdat.cbvstmpt
                        for k in range(gdat.numbcbvs+1):
                            print('rflxcbvs[:, k]')
                            summgene(rflxcbvs[:, k])
                        rflxcbvstotl = np.sum(rflxcbvs[:, :-1], 1)
                        print('rflxcbvstotl')
                        summgene(rflxcbvstotl)
                        gdat.mlikfittflux[:, 0] -= rflxcbvstotl
                        print('gdat.mlikfittflux[:, 0]')
                        summgene(gdat.mlikfittflux[:, 0])
                        print('')
                        
                    timefinl = timemodu.time()
                    print('Done in %g seconds.' % (timefinl - timeinit))
                    
                    gdat.medifittflux = np.median(gdat.mlikfittflux, 0)
                    print('Median flux of the central source is %g ADU.' % gdat.medifittflux[0])
                    
                    # normalize fluxes to get relative fluxes
                    print('Normalizing by the median flux...')
                    gdat.mlikfittrflx = gdat.mlikfittflux / gdat.medifittflux[None, :]
                    gdat.stdvfittrflx = gdat.stdvfittflux / gdat.medifittflux[None, :]
                    
                    if gdat.booldiagmode:
                        for a in range(gdat.listtime[o].size):
                            if a != gdat.listtime[o].size - 1 and gdat.listtime[o][a] >= gdat.listtime[o][a+1]:
                                raise Exception('')

                    # write the light curve to file
                    print('Writing all light curves to %s...' % pathsavemeta)
                    arry = gdat.medifittflux
                    np.savetxt(pathsavemeta, arry)
                    
                    arry = np.empty((gdat.numbtime[o], 2*gdat.numbcomp+1))
                    arry[:, 0] = gdat.listtime[o]
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
                
                    nameplot = 'rflx'
                    path = retr_pathvisu(gdat, nameplot, strgsave)
                    if not os.path.exists(path):
                    
                        print('Reading from %s...' % pathsavemeta)
                        gdat.medifittflux = np.loadtxt(pathsavemeta)
                        
                        gdat.mlikfittrflx = np.empty((gdat.numbtime[o], gdat.numbcomp))
                        gdat.stdvfittrflx = np.empty((gdat.numbtime[o], gdat.numbcomp))
                        print('Reading from %s...' % pathsaverflx)
                        arry = np.loadtxt(pathsaverflx)
                        gdat.listtime[o] = arry[:, 0]
                        for k in gdat.indxcomp:
                            try:
                                gdat.mlikfittrflx[:, k] = arry[:, 2*k+1]
                            except:
                                print('arry')
                                summgene(arry)
                                print('gdat.mlikfittrflx')
                                summgene(gdat.mlikfittrflx)
                                raise Exception('')
                            gdat.stdvfittrflx[:, k] = arry[:, 2*k+2]
                        gdat.mlikfittflux = gdat.medifittflux * gdat.mlikfittrflx
                        gdat.listarry[o] = arry[:, :3]
                    else:
                        print('Plots already exist at %s...' % path)
                
                print('Evaluating the model at all time bins...')
                cntpbackneww = gdat.mlikfittflux[:, -1]
                timeinit = timemodu.time()
                coef = None
                gdat.cntpmodl = retr_cntpmodl(gdat, xpostemp, ypostemp, gdat.mlikfittflux[:, :-1], cntpbackneww, o, coef=coef)
                print('gdat.cntpmodl')
                summgene(gdat.cntpmodl)
                print('gdat.cntpdata')
                summgene(gdat.cntpdata)

                timefinl = timemodu.time()
                print('Done in %g seconds.' % (timefinl - timeinit))
                    
                gdat.cntpdatanbkg = gdat.cntpdata - gdat.mlikfittrflx[None, None, :, -1] * gdat.medifittflux[-1]
                gdat.cntpresi = gdat.cntpdata - gdat.cntpmodl
                chi2 = np.mean(gdat.cntpresi**2 / gdat.cntpdata) + 2 * gdat.numbstar
                
                if gdat.boolcalcconr:
                    print('Evaluating the PSFs for contamination ratio...')
                    corr = np.sum(matrdesi * matrdesi[:, 0, None], 0)
                    #corr /= np.amax()
                    print('corr')
                    print(corr)

        
                # color scales
                for typecntpscal in gdat.listtypecntpscal:
                    for strg in ['modl', 'resi', 'datanbkg']:
                        setp_cntp(gdat, strg, typecntpscal)
                
                if not os.path.exists(pathsaverflx) and gdat.boolplotrflx:
                    print('Plotting light curves...')
                    if gdat.listlimttimetzom is not None:
                        gdat.indxtimelimt = []
                        for limttimeplot in gdat.listlimttimetzom:
                            gdat.indxtimelimt.append(np.where((gdat.listtime[o] > limttimeplot[0]) & (gdat.listtime[o] < limttimeplot[1]))[0])
                    
                    # plot light curve derived from aperture photometry
                    if gdat.listpixlaper is not None:
                        plot_lcur(gdat, gdat.cntpaper, 0.01 * gdat.cntpaper, k, o, '_' + strgsecc, gdat.booltpxf[o], '_aper', strgsave)
                        
                    if gdat.boolcbvs:
                        plot_lcur(gdat, gdat.mlikfittfluxraww, 0 * gdat.mlikfittfluxraww, 0, o, '_' + strgsecc, gdat.booltpxf[o], strgoffs, strgsave)
                        
                    # plot the light curve of the target stars and background
                    for k in gdat.indxcomp:

                        if x == 1 and y == 1 or k == 0:

                            if gdat.boolrefr[o] and x == 1 and y == 1:
                                listmodeplot = [0, 1]
                            else:
                                listmodeplot = [0]
                            plot_lcur(gdat, gdat.mlikfittrflx[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, \
                                                                             gdat.booltpxf[o], strgoffs, strgsave, listmodeplot=listmodeplot)
                            
                            if gdat.listlimttimetzom is not None:
                                for p in range(len(gdat.listlimttimetzom)):
                                    plot_lcur(gdat, gdat.mlikfittrflx[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, gdat.booltpxf[o], strgoffs, \
                                                                                   strgsave, indxtimelimt=gdat.indxtimelimt[p], indxtzom=p)
                                    if k == 0:
                                        plot_lcur(gdat, gdat.mlikfittflux[:, k], gdat.stdvfittflux[:, k], k, o, '_' + strgsecc, \
                                                                                                gdat.booltpxf[o], strgoffs, strgsave, \
                                                                                          indxtimelimt=gdat.indxtimelimt[p], indxtzom=p)
                            
                    else:
                        gdat.mlikrflxbdtr = gdat.mlikfittrflx[:, 0]
                        
        # temporal types of image plots
        ## medians
        listtypeplotcntp = []
        if gdat.boolplotcntp:
            listtypeplotcntp += ['medi']
        # cadence frames
        if gdat.boolanim:
            listtypeplotcntp += ['anim']
        
        
        for typeplotcntp in listtypeplotcntp:
            for nameplotcntp in listnameplotcntp:
                
                # make animation plot
                pathanim = retr_pathvisu(gdat, nameplotcntp, strgsave, boolanim=True)
                if typeplotcntp == 'anim' and os.path.exists(pathanim):
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
                        plot_cntp(gdat, cntptemp, o, typecntpscal, nameplotcntp, strgsave, strgtitl=strgtitl, boolresi=boolresi, vmin=vmin, vmax=vmax)
                    if typeplotcntp == 'anim':
                        args = [gdat, cntptemp, o, typecntpscal, nameplotcntp, strgsave]
                        kwag = { \
                                            'boolresi': boolresi, 'listindxpixlcolr': gdat.listpixlaper, \
                                            'listtimelabl':listtimelabl, \
                                            'vmin':vmin, 'vmax':vmax, \
                                            'lcur':gdat.mlikfittrflx[:, 0], 'time':gdat.listtime[o]}
                        listpath = plot_cntpwrap(*args, **kwag)
                
                        # make animation
                        cmnd = 'convert -delay 20 '
                        for path in listpath:
                            cmnd += '%s ' % path
                        cmnd += '%s' % pathanim
                        os.system(cmnd)
                        
                        # delete images
                        for path in listpath:
                            os.system('rm %s' % path)
        
    timefinltotl = timemodu.time()
    print('Total execution time: %g seconds.' % (timefinltotl - timeinittotl))
    print('')                
    gdat.dictoutp['strgtarg'] = gdat.strgtarg
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
        intgresu = init( \
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


