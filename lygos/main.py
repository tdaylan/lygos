import os, datetime, fnmatch, time as timemodu

import numpy as np

import scipy.interpolate
from scipy import ndimage

#from numba import jit, prange

import h5py

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from astroquery.mast import Catalogs
import astroquery.mast
import astroquery

import astropy

from tdpy.util import summgene
import tdpy
import ephesus
import miletos


def retr_fluxfromtmag(tmag):
    '''
    Calculate the flux per second from the TESS magnitude
    '''
    
    flux = 10**(0.4 * (20.4 - tmag))
    
    return flux


def retr_tmagfromflux(flux):
    '''
    Calculate the TESS magnitude from flux per second
    '''
    
    tmag = 20.4 - 2.5 * np.log10(flux)
    
    return tmag


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


def anim_cntp(gdat, cntp, o, typecntpscal, nameplotcntp, strgsave, \
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


def plot_cntp(gdat, \
              cntp, \
              o, \
              typecntpscal, \
              nameplotcntp, \
              strgsave, \
              indxpcol=None, \
              cbar='Greys_r', \
              strgtitl='', \
              boolresi=False, \
              xposoffs=None, \
              yposoffs=None, \
              strgextn='', \
              lcur=None, \
              boolanno=True, \
              indxtimeplot=None, \
              time=None, \
              timelabl=None, \
              thistime=None, \
              vmin=None, \
              vmax=None, \
              listindxpixlcolr=None, \
             ):
    
    if typecntpscal == 'asnh':
        cntp = np.arcsinh(cntp)
    
    if strgextn != '':
        strgextn = '_' + strgextn
    
    path = retr_pathvisu(gdat, nameplotcntp, strgsave, typecntpscal=typecntpscal, indxpcol=indxpcol, indxtimeplot=indxtimeplot)

    if os.path.exists(path):
        print('Plot exists at %s. Skipping the plotting...' % path)
    else:
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
            
            for q in gdat.refr.indxcatl:
                # reference sources
                axis[0].scatter(gdat.refr.catl[q][o]['xpos'], gdat.refr.catl[q][o]['ypos'], alpha=1., s=gdat.sizeplotsour, color='r', marker='o')
                # label the reference sources
                for indxtemp in gdat.refr.indxsour[q][o]:
                    axis[0].text(gdat.refr.catl[q][o]['xpos'][indxtemp], gdat.refr.catl[q][o]['ypos'][indxtemp], gdat.refr.catl[q][o]['labl'][indxtemp], color='r')
                
                # fitting sources
                xposfitt = np.copy(gdat.fitt.catl['xpos'])
                yposfitt = np.copy(gdat.fitt.catl['ypos'])
                if xposoffs is not None:
                    # add the positional offset, if any
                    xposfitt += xposoffs
                    yposfitt += yposoffs
                ## target
                axis[0].scatter(xposfitt[0], yposfitt[0], alpha=1., color='b', s=2*gdat.sizeplotsour, marker='o')
                ## neighbors
                axis[0].scatter(xposfitt[1:], yposfitt[1:], alpha=1., s=gdat.sizeplotsour, color='b', marker='o')
                for k in gdat.indxsour:
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
            axis[1].plot(time - gdat.timeoffs, lcur, color='black', ls='', marker='o', markersize=1)
            axis[1].set_xlabel(gdat.labltime) 
            axis[1].set_ylabel('Relative flux') 
            axis[1].axvline(thistime - gdat.timeoffs)
        
        print('Writing to %s...' % path)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    
    return path


def plot_lcurcomp(gdat, lcurmodl, stdvlcurmodl, k, indxtsecplot, strgsecc, strgsave, strgplot, timeedge=None, \
                    strgextn='', indxtimelimt=None, indxtzom=None, boolerrr=False):
    
    if k == 0:
        lablcomp = ', Target source'
    elif k == gdat.fitt.numbcomp - 1:
        lablcomp = ', Background'
    else:
        lablcomp = ', Neighbor Source %d' % k
    
    timedatatemp = np.copy(gdat.listtime[indxtsecplot])
    timerefrtemp = [[] for q in gdat.refr.indxtser[indxtsecplot]] 
    for q in gdat.refr.indxtser[indxtsecplot]:
        timerefrtemp[q] = gdat.refrtime[indxtsecplot][q]
    
    if indxtimelimt is not None:
        timedatatemp = timedatatemp[indxtimelimt]
        stdvlcurmodl = stdvlcurmodl[indxtimelimt]
        lcurmodl = lcurmodl[indxtimelimt]

    nameplot = 'rflx_%s' % (strgplot)
    path = retr_pathvisu(gdat, nameplot, strgsave + '_s%03d' % k, indxtzom=indxtzom)
    
    # skip the plot if it has been made before
    if os.path.exists(path):
        print('Plot exists at %s. Skipping the plotting...' % path)
        return

    figr, axis = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2, 0.7]}, figsize=(12, 8))
    axis[1].set_xlabel(gdat.labltime)
    
    axis[0].set_ylabel('Relative flux')
    axis[1].set_ylabel('Residual')
    
    if boolerrr:
        yerr = stdvlcurmodl
    else:
        yerr = None
    temp, listcaps, temp = axis[0].errorbar(timedatatemp - gdat.timeoffs, lcurmodl, yerr=yerr, color='grey', ls='', ms=1, \
                                                                                marker='.', lw=3, alpha=0.5, label='Lygos', rasterized=True)
    for caps in listcaps:
        caps.set_markeredgewidth(3)
    
    if timeedge is not None:
        for timeedgetemp in timeedge[1:-1]:
            axis[0].axvline(timeedgetemp - gdat.timeoffs, ls='--', color='grey', alpha=0.5)
    
    for q in gdat.refr.indxtser[indxtsecplot]:
        if boolerrr:
            yerr = gdat.stdvrefrrflx[indxtsecplot][q]
        else:
            yerr = None
        temp, listcaps, temp = axis[0].errorbar(timerefrtemp - gdat.timeoffs, gdat.refrrflx[indxtsecplot][q], \
                                            yerr=yerr, color=gdat.refr.colrlcur[indxtsecplot][q], ls='', markersize=2, \
                                                                    marker='.', lw=3, alpha=0.3, label=gdat.refr.labltser[indxtsecplot][q])
        for caps in listcaps:
            caps.set_markeredgewidth(3)
    
    ## residual
    for q in gdat.refr.indxtser[indxtsecplot]:
        if lcurmodl.size == gdat.refrrflx[indxtsecplot][q].size:
            ydat = lcurmodl - gdat.refrrflx[indxtsecplot][q]
            if boolerrr:
                yerr = None
            else:
                yerr = None
            axis[1].errorbar(timedatatemp - gdat.timeoffs, ydat, yerr=yerr, label=gdat.refr.labltser[indxtsecplot][q], \
                                                color='k', ls='', marker='.', markersize=2, alpha=0.3)
    axis[0].set_title(gdat.labltarg + lablcomp)
    if gdat.listtimeplotline is not None:
        for timeplotline in gdat.listtimeplotline:
            axis[0].axvline(timeplotline - gdat.timeoffs, ls='--')
    
    if gdat.refr.numbtser[indxtsecplot] > 0:
        axis[0].legend()

    if indxtzom is not None:
        axis[a].set_xlim(gdat.listlimttimetzom[indxtzom] - gdat.timeoffs)
    
    #plt.tight_layout()
    print('Writing to %s...' % path)
    plt.savefig(path, dpi=200)
    plt.close()


    
def plot_lcur(gdat, lcurmodl, stdvlcurmodl, k, indxtsecplot, strgsecc, strgsave, strgplot, timeedge=None, \
                    strgextn='', indxtimelimt=None, indxtzom=None, boolerrr=False):
    
    if k == 0:
        lablcomp = ', Target source'
    elif k == gdat.fitt.numbcomp - 1:
        lablcomp = ', Background'
    else:
        lablcomp = ', Neighbor Source %d' % k
    timedatatemp = np.copy(gdat.listtime[indxtsecplot])
    
    if indxtimelimt is not None:
        timedatatemp = timedatatemp[indxtimelimt]
        stdvlcurmodl = stdvlcurmodl[indxtimelimt]
        lcurmodl = lcurmodl[indxtimelimt]

    nameplot = 'rflx_%s' % (strgplot)
    path = retr_pathvisu(gdat, nameplot, strgsave + '_s%03d' % k, indxtzom=indxtzom)
    
    # skip the plot if it has been made before
    if os.path.exists(path):
        print('Plot exists at %s. Skipping the plotting...' % path)
        return

    figr, axis = plt.subplots(figsize=(8, 2.5))
    axis.set_xlabel(gdat.labltime)
    axis.set_ylabel('Relative flux')
    
    if boolerrr:
        yerr = stdvlcurmodl
    else:
        yerr = None
    temp, listcaps, temp = axis.errorbar(timedatatemp - gdat.timeoffs, lcurmodl, yerr=yerr, color='grey', ls='', ms=1, marker='.', lw=3, rasterized=True)
    for caps in listcaps:
        caps.set_markeredgewidth(3)
    
    if timeedge is not None:
        for timeedgetemp in timeedge[1:-1]:
            axis.axvline(timeedgetemp - gdat.timeoffs, ls='--', color='grey', alpha=0.5)
    axis.set_title(gdat.labltarg + lablcomp)
    if gdat.listtimeplotline is not None:
        for timeplotline in gdat.listtimeplotline:
            axis.axvline(timeplotline - gdat.timeoffs, ls='--')
    
    if indxtzom is not None:
        axis.set_xlim(gdat.listlimttimetzom[indxtzom] - gdat.timeoffs)
    
    plt.subplots_adjust(bottom=0.2)
    print('Writing to %s...' % path)
    plt.savefig(path, dpi=200)
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
                  typevisu='plot', \
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
    
    if typevisu == 'anim':
        typefileplot = 'gif'
    else:
        typefileplot = gdat.typefileplot
    
    if indxtimeplot is None:
        strgtime = ''
    else:
        strgtime = '_%08d' % indxtimeplot
    
    pathvisu = gdat.pathimagtarg + '%s%s%s%s%s%s.%s' % (nameplot, strgsave, strgscal, strgtzom, strgpcol, strgtime, typefileplot)
    
    return pathvisu


def retr_cntpmodl(gdat, xpos, ypos, cnts, cntpbackscal, parapsfn):
    '''
    Calculate the model image
    '''
    
    if gdat.typepsfnsubp == 'eval':
         
        if cnts.ndim == 2:
            cntpmodl = np.zeros((gdat.numbside, gdat.numbside, cnts.shape[0])) + cntpbackscal
        else:
            cntpmodl = np.zeros((gdat.numbside, gdat.numbside)) + cntpbackscal
            
        for k in range(xpos.size):
            deltxpos = gdat.xposimag - xpos[k]
            deltypos = gdat.yposimag - ypos[k]
            if gdat.typepsfnshap == 'gaus':
                psfnsour = np.exp(-(deltxpos / parapsfn[0])**2 - (deltypos / parapsfn[0])**2)
            if gdat.typepsfnshap.startswith('gauselli'):
                psfnsour = np.exp(-0.5 * (deltxpos / coef[0])**2 - 0.5 * (deltypos / coef[1])**2)
            if gdat.typepsfnshap == 'pfre':
                psfnsour = coef[0] * deltxpos + coef[1] * deltypos + coef[2] * deltxpos * deltypos + \
                                coef[3] * deltxpos**2 + coef[4] * deltypos**2 + coef[5] * deltxpos**2 * deltypos + coef[6] * deltypos**2 * deltxpos + \
                                coef[7] * deltxpos**3 + coef[8] * deltypos**3 + coef[9] * np.exp(-deltxpos**2 / coef[10] - deltypos**2 / coef[11])
            
            if cnts.ndim == 2:
                cntpmodl += cnts[None, None, :, k] * psfnsour[:, :, None]
            else:
                cntpmodl += cnts[k] * psfnsour
    
    if gdat.typepsfnsubp == 'regrcubipixl':
         
        # construct the design matrix
        ## subpixel shifts
        dx = xpos - xpos.astype(int) - 0.5
        dy = ypos - ypos.astype(int) - 0.5
        ## design matrix
        matrdesi = np.column_stack((np.ones(x.size), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy))
        
        tmpt = np.dot(matrdesi, gdat.coefspix).reshape((-1, gdat.numbside, gdat.numbside))
        cntpmodl = np.sum(tmpt[:, :, :, None] * cnts[:, None, None, :], 0)
        
        if gdat.diagmode:
            if not np.isfinite(cntpmodl).all():
                print('WARNING: cntpmodl was infinite!!!!')
                raise Exception('')

    return cntpmodl


def retr_llik(para, gdat):
    
    # parse the parameter vector
    if gdat.typefittpsfnposi == 'fixd':
        xpos = gdat.fitt.catl['xpos']
        ypos = gdat.fitt.catl['ypos']
    else:
        xpos = para[:gdat.fitt.numbsour]
        ypos = para[gdat.fitt.numbsour:2*gdat.fitt.numbsour]
    
    #cnts = para[2*gdat.fitt.numbsour:3*gdat.fitt.numbsour]
    cnts = para[:gdat.fitt.numbsour]
    
    cntpbackscal = para[gdat.fitt.numbsour]
    
    parapsfn = para[gdat.fitt.numbsour+1:gdat.fitt.numbsour+1+gdat.numbparapsfn]
    
    cntpmodl = retr_cntpmodl(gdat, xpos, ypos, cnts, cntpbackscal, parapsfn)
    
    chi2 = np.sum((gdat.cntpdatatmed - cntpmodl)**2 / gdat.cntpdatatmed)
    
    llik = -0.5 * chi2
    
    return llik


#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def retr_mlikregr(cntpdata, matrdesi, vari):
    '''
    Return the maximum likelihood estimate for linear regression.
    '''

    varitdim = np.diag(vari.flatten())
    covafittcnts = np.linalg.inv(np.matmul(np.matmul(matrdesi.T, np.linalg.inv(varitdim)), matrdesi))
    mlikfittcnts = np.matmul(np.matmul(np.matmul(covafittcnts, matrdesi.T), np.linalg.inv(varitdim)), cntpdata.flatten())
    
    return mlikfittcnts, covafittcnts


def retr_strgsave(gdat, strgsecc, x, y, o):
    
    strgnumbside = '_n%03d' % gdat.numbside
    strgmaxmdmag = '_d%3.1f' % gdat.maxmdmag
    strgoffs = '_of%d%d' % (x, y)
    strgsave = '_%s_%s_%s%s%s%s' % (gdat.strgcnfg, strgsecc, gdat.typecade[o], strgnumbside, strgmaxmdmag, strgoffs)

    return strgsave


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
    

def retr_strgsecc(gdat, o):
    
    strgsecc = '%02d%d%d' % (gdat.listtsec[o], gdat.listtcam[o], gdat.listtccd[o])

    return strgsecc


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
    
         # TESS magniude of the target
         tmagtarg=None, \

         # path for the target
         pathtarg=None, \
        
         ## number of pixels on a side to cut out
         numbside=11, \
        
         ## mask
         ### Boolean flag to put a cut on quality flag
         boolmaskqual=True, \
         ## limits of time between which the quality mask will be ignored
         limttimeignoqual=None, \

         ### masking region
         epocmask=None, \
         perimask=None, \
         duramask=None, \
         
         # processing
         ## Boolean flag to turn on CBV detrending
         booldetrcbvs=False, \

         # string indicating the cluster of targets
         strgclus=None, \
         
         # reference time-series
         refrarrytser=None, \
         # list of labels for reference time-series
         refrlistlabltser=None, \

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
         boolanimframtotl=True, \
        
         ## Boolean flag to plot the histogram of the number of counts
         boolplothhistcntp=False, \

         # plot extensions
         typefileplot='pdf', \
        
         # diagnostics
         booldiagmode=True, \
        
         # Boolean flag to calculate the contamination ratio
         boolcalcconr = False, \

         # model
        
         # maximum delta magnitude of neighbor sources to be included in the model
         maxmdmag=4., \
        
         # time offset for time-series plots
         timeoffs=2457000., \

         ## PSF
         ### type of inference
         #### 'fixd': fixed
         #### 'osam': based on only the dithered image data collected during commissioning
         #### 'locl': based on only the image data to be analyzed
         #### 'both': based on both
         typepsfninfe='fixd', \
         
         # type of template PSF model shape
         #### 'gaus': univariate Gaussian
         #### 'osam': based on the dithered image data collected during commissioning
         typepsfnshap='gaus', \

         ### type of sub-pixel interpolation of the point source emission model
         #### 'eval': evaluation of a functional form on the data grid (not to be used when point source positions are floating)
         #### 'regrcubi': cubic regression
         #### 'regrline': linear regression
         #### 'regrcubipixl': separate cubic regression in each pixel
         typepsfnsubp='eval', \
         
         catlextr=None, \
         lablcatlextr=None, \
            
         # Boolean flag to run miletos at the end on the light curve
         boolmile=False, \
        
         # input dictionary to miletos
         dictmileinpt=None, \

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
        
         # verbosity level
         typeverb=1, \
        
         # Boolean flag to turn on diagnostic mode
         diagmode=True, \
        
         # Boolean flag to turn on regression C library
         boolclib=False, \
         
         #**kargs, \
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
    #for strg, valu in args.items():
    #    setattr(gdat, strg, valu)

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print('lygos initialized at %s...' % gdat.strgtimestmp)
    # paths
    gdat.pathbase = os.environ['LYGOS_DATA_PATH'] + '/'
    gdat.pathimaglygo = gdat.pathbase + 'imag/'
    gdat.pathdatalygo = gdat.pathbase + 'data/'
    
    np.set_printoptions(linewidth=200, \
                        precision=5, \
                       )

    # check input
    ## ensure target identifiers are not conflicting
    if gdat.typedata == 'mock' or gdat.typedata == 'obsd':
        if not (gdat.ticitarg is not None and gdat.strgmast is None and gdat.toiitarg is None and gdat.rasctarg is None and gdat.decltarg is None or \
                gdat.ticitarg is None and gdat.strgmast is not None and gdat.toiitarg is None and gdat.rasctarg is None and gdat.decltarg is None or \
                gdat.ticitarg is None and gdat.strgmast is None and gdat.toiitarg is not None and gdat.rasctarg is None and gdat.decltarg is None or \
                gdat.ticitarg is None and gdat.strgmast is None and gdat.toiitarg is None and gdat.rasctarg is not None and gdat.decltarg is not None):
            print('gdat.ticitarg')
            print(gdat.ticitarg)
            print('gdat.strgmast')
            print(gdat.strgmast)
            print('gdat.toiitarg')
            print(gdat.toiitarg)
            print('gdat.rasctarg')
            print(gdat.rasctarg)
            print('gdat.decltarg')
            print(gdat.decltarg)
            raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) should be provided.')
    
    if gdat.typedata == 'toyy' and gdat.tmagtarg is None or gdat.typedata != 'toyy' and gdat.tmagtarg is not None:
        print('gdat.typedata')
        print(gdat.typedata)
        print('gdat.tmagtarg')
        print(gdat.tmagtarg)
        raise Exception('tmagtarg needs to be set when toy data are generated.')

    gdat.pathtoii = gdat.pathbase + 'data/exofop_tess_tois.csv'
    print('Reading from %s...' % gdat.pathtoii)
    objtexof = pd.read_csv(gdat.pathtoii, skiprows=0)

    # conversion factors
    gdat.dictfact = ephesus.retr_factconv()
    
    print('gdat.numbside')
    print(gdat.numbside)

    # determine target identifiers
    if gdat.ticitarg is not None:
        gdat.strgtargtype = 'tici'
        print('A TIC ID was provided as target identifier.')
        indx = np.where(objtexof['TIC ID'].values == gdat.ticitarg)[0]
        if indx.size > 0:
            gdat.toiitarg = int(str(objtexof['TOI'][indx[0]]).split('.')[0])
            print('Matched the input TIC ID with TOI %d.' % gdat.toiitarg)
        gdat.strgmast = 'TIC %d' % gdat.ticitarg
    elif gdat.toiitarg is not None:
        gdat.strgtargtype = 'toii'
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
        gdat.strgtargtype = 'mast'
        print('A MAST key (%s) was provided as target identifier.' % gdat.strgmast)
    elif gdat.rasctarg is not None and gdat.decltarg is not None:
        gdat.strgtargtype = 'posi'
        print('RA and DEC (%g %g) are provided as target identifier.' % (gdat.rasctarg, gdat.decltarg))
        gdat.strgmast = '%g %g' % (gdat.rasctarg, gdat.decltarg)
    
    if gdat.typedata == 'mock' or gdat.typedata == 'obsd':
        print('gdat.strgtargtype')
        print(gdat.strgtargtype)
        print('gdat.strgmast')
        print(gdat.strgmast)
        print('gdat.toiitarg')
        print(gdat.toiitarg)
    
    if gdat.typedata != 'toyy':
        # temp -- check that the closest TIC to a given TIC is itself
        
        maxmradi = 1.1 * 20. * 1.4 * (gdat.numbside + 2) / 2.
        print('Querying the TIC within %d as to get the RA, DEC, Tmag, and TIC ID of the closest source to the MAST keywrod %s...' % (maxmradi, gdat.strgmast))
        catalogData = astroquery.mast.Catalogs.query_region(gdat.strgmast, radius='%ds' % maxmradi, catalog="TIC")
        print('Found %d TIC sources within 200 as.' % len(catalogData))
    
        if gdat.strgtargtype != 'posi':
            gdat.ticitarg = int(catalogData[0]['ID'])
            gdat.rasctarg = catalogData[0]['ra']
            gdat.decltarg = catalogData[0]['dec']
            gdat.tmagtarg = catalogData[0]['Tmag']
        
        objticrs = astropy.coordinates.SkyCoord(ra=gdat.rasctarg*astropy.units.degree, \
                                               dec=gdat.decltarg*astropy.units.degree, frame='icrs')
        gdat.lgalttarg = objticrs.galactic.l
        gdat.bgalttarg = objticrs.galactic.b
        gdat.laecttarg = objticrs.barycentricmeanecliptic.lon.degree
        gdat.beecttarg = objticrs.barycentricmeanecliptic.lat.degree
        
        print('gdat.ticitarg')
        print(gdat.ticitarg)
        print('gdat.rasctarg')
        print(gdat.rasctarg)
        print('gdat.decltarg')
        print(gdat.decltarg)
        print('gdat.tmagtarg')
        print(gdat.tmagtarg)
    
    if gdat.labltarg is None:
        if gdat.typedata == 'mock' or gdat.typedata == 'obsd':
            if gdat.strgtargtype == 'mast':
                gdat.labltarg = gdat.strgmast
            if gdat.strgtargtype == 'toii':
                gdat.labltarg = 'TOI %d' % gdat.toiitarg
            if gdat.strgtargtype == 'tici':
                gdat.labltarg = 'TIC %d' % gdat.ticitarg
            if gdat.strgtargtype == 'posi':
                gdat.labltarg = 'RA=%.4g, DEC=%.4g' % (gdat.rasctarg, gdat.decltarg)
        else:
            raise Exception('A label must be provided for toy targets.')
    if gdat.strgtarg is None:
        gdat.strgtarg = ''.join(gdat.labltarg.split(' '))
    
    print('Target label: %s' % gdat.labltarg) 
    print('Output folder name: %s' % gdat.strgtarg) 
    if gdat.typedata != 'toyy':
        print('RA and DEC: %g %g' % (gdat.rasctarg, gdat.decltarg))
    if gdat.typedata == 'toyy' or gdat.strgtargtype == 'tici' or gdat.strgtargtype == 'mast':
        print('Tmag: %g' % gdat.tmagtarg)
   
    print('PSF inference type: %s' % gdat.typepsfninfe)
    print('PSF model shape type: %s' % gdat.typepsfnshap)
    print('PSF subpixel interpolation type: %s' % gdat.typepsfnsubp)
    
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
    
    os.system('mkdir -p %s' % gdat.pathimaglygo)
    os.system('mkdir -p %s' % gdat.pathdatalygo)
    os.system('mkdir -p %s' % gdat.pathimagtarg)
    os.system('mkdir -p %s' % gdat.pathdatatarg)
    os.system('mkdir -p %s' % gdat.pathimagclus)
    os.system('mkdir -p %s' % gdat.pathdataclus)
    
    # create a separate folder to place the PSF fit output
    if gdat.typepsfninfe != 'fixd':
        gdat.pathimagtargpsfn = gdat.pathimagtarg + 'psfn/'
        os.system('mkdir -p %s' % gdat.pathimagtargpsfn)
   
    # header that will be added to the output CSV files
    gdat.strghead = 'time [BJD], relative flux, relative flux error'
    
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
    gdat.strgcnfg = '%s%s' % (strgtypedata, gdat.strgtarg)
    
    gdat.dictoutp = dict()
    if gdat.typedata == 'toyy':
        gdat.numbtsec = 1
        gdat.indxtsec = np.arange(gdat.numbtsec)
        gdat.listtsec = [0]
        gdat.listtcam = [0]
        gdat.listtccd = [0]
    
    if gdat.typedata != 'toyy':
        
        # check all available TESS (FFI) cutout data 
        objtskyy = astropy.coordinates.SkyCoord(gdat.rasctarg, gdat.decltarg, unit="deg")
        print('Calling TESSCut to get the data...')
        listhdundatatemp = astroquery.mast.Tesscut.get_cutouts(objtskyy, gdat.numbside)
        
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
            
            # get the list of sectors for which TPF data are available
            print('Retrieving the list of available TESS sectors for which there are TPF data...')
            # get observation tables
            listtablobsv = ephesus.retr_listtablobsv(gdat.strgmast)
            listprodspoc = []
            for k, tablobsv in enumerate(listtablobsv):
                
                listprodspoctemp = astroquery.mast.Observations.get_product_list(tablobsv)
                
                if listtablobsv['distance'][k] > 0:
                    continue

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
                                                                gdat.listtccdspoc[o] = ephesus.read_tesskplr_file(listpathdownspoctpxf[o])

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
        
        gdat.numbtsec = gdat.listtsec.size
        gdat.indxtsec = np.arange(gdat.numbtsec)
        
    gdat.typecade = np.zeros(gdat.numbtsec, dtype=object)
    if gdat.typedata != 'toyy':
        # determine whether sectors have TPFs
        gdat.booltpxf = ephesus.retr_booltpxf(gdat.listtsec, gdat.listtsecspoc)
        print('gdat.booltpxf')
        print(gdat.booltpxf)

        gdat.listhdundata = [[] for o in gdat.indxtsec]
        for o in gdat.indxtsec:
            if gdat.booltpxf[o]:
                indxtsectemp = np.where(gdat.listtsec[o] == gdat.listtsecspoc)[0][0]
                gdat.listhdundata[o] = listhdundataspoc[indxtsectemp]
            else:
                indxtsectemp = np.where(gdat.listtsec[o] == gdat.listtsecffim)[0][0]
                gdat.listhdundata[o] = gdat.listhdundataffim[indxtsectemp]

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

        # check for an earlier lygos run
        for o in gdat.indxtsec:

            strgsecc = retr_strgsecc(gdat, o)
            strgsave = retr_strgsave(gdat, strgsecc, 1, 1, o)
            pathsaverflxtarg = gdat.pathdatatarg + 'rflxtarg' + strgsave + '.csv'
            gdat.dictoutp['pathsaverflxtargsc%02d' % gdat.listtsec[o]] = pathsaverflxtarg
            if os.path.exists(pathsaverflxtarg):
                print('Run previously completed...')
                gdat.listarry = [[] for o in gdat.indxtsec]
                for o in gdat.indxtsec:
                    gdat.listarry[o] = np.loadtxt(pathsaverflxtarg, delimiter=',', skiprows=1)
        
    if gdat.typedata == 'toyy':
        for o, tsec in enumerate(gdat.listtsec):
            gdat.typecade[o] = '10mn'
    
    gdat.fitt = tdpy.util.gdatstrt()
    gdat.refr = tdpy.util.gdatstrt()
    
    # get define reference catalogs
    gdat.refr.lablcatl = ['TIC']
    if gdat.catlextr is not None:
        gdat.refr.lablcatl.extend(gdat.lablcatlextr)

    gdat.refr.numbcatl = len(gdat.refr.lablcatl)
    gdat.refr.indxcatl = np.arange(gdat.refr.numbcatl)
    gdat.refr.numbsour = np.empty((gdat.refr.numbcatl, gdat.numbtsec), dtype=int)
    gdat.refr.indxsour = [[[] for o in gdat.indxtsec] for q in np.arange(gdat.refr.numbcatl)]
        
    # read TESS PSF
    pathpsfn = gdat.pathdatalygo + 'tesspsfn/listpsfn.h5'
    print('Reading from %s...' % pathpsfn)
    objth5py = h5py.File(pathpsfn, 'r')
    listpsfn = objth5py.get('listpsfn')
    cntppsfntess = np.array(listpsfn)[0, 0, 0, 0, :, :]
    cntppsfntess /= np.sum(cntppsfntess)
    objth5py.close()
        
    #pathpsfn = gdat.pathdatalygo + 'tesspsfn.txt'
    #if not os.path.exists(pathpsfn):
    #    cmnd = 'tsig-psf --resolution 11 --id 4564 --show-contents > %s' % pathpsfn
    #    os.system(cmnd)
    ## read
    #cntppsfnusam = np.loadtxt(pathpsfn, delimiter=',')
    #cntppsfnusam = ndimage.convolve(cntppsfnusam, np.ones((11, 11)) / 121., mode='constant', cval=0.) * 121.
    #cntppsfnusam = cntppsfnusam[None, :, :]
        
    # Gaussian width
    if gdat.typepsfnshap == 'gaus':
        gdat.para = np.empty(1)
        gdat.para[0] = 1.1 # [pixel]

    # setup regression
    if gdat.typepsfnsubp == 'regrcubipixl':
        gdat.numbregixaxi = gdat.numbside
        gdat.numbregiyaxi = gdat.numbside
        gdat.numbsidepsfn = 17

        gdat.numbsidepsfnusam = 13 * 9
        gdat.factusam = 9
        gdat.cntppsfnusam = cntppsfntess
        gdat.cntppsfnusam = ndimage.convolve(gdat.cntppsfnusam, np.ones((9, 9)), mode='constant', cval=0.)
        gdat.cntppsfnusam /= np.sum(gdat.cntppsfnusam)
        gdat.cntppsfnusam *= 81
        print('gdat.cntppsfnusam')
        summgene(gdat.cntppsfnusam)
        gdat.boolplotsave = False
   
        # make design matrix for each original pixel
        print('gdat.factusam')
        print(gdat.factusam)
        print('gdat.numbsidepsfnusam')
        print(gdat.numbsidepsfnusam)
        gdat.numbsidepsfn = int(gdat.numbsidepsfnusam / gdat.factusam)
        print('gdat.numbsidepsfn')
        print(gdat.numbsidepsfn)
        assert(gdat.numbsidepsfn % 2 == 1)
        
        # construct the design matrix for the pixel

        maxm = 1. / gdat.factusam
        arry = np.linspace(-maxm, maxm, gdat.factusam)
        x, y = np.meshgrid(arry, arry)
        x = x.flatten()
        y = y.flatten()
        A = np.column_stack([np.ones(gdat.factusam**2), x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y])
        
        # number of regressors
        gdat.numbpararegr = A.shape[1]

        # regression coefficients for each pixel
        gdat.coefspix = np.zeros((gdat.numbpararegr, gdat.numbsidepsfn, gdat.numbsidepsfn))
        
        # loop over original psf pixels and regress the coefficients
        for a in np.arange(gdat.numbsidepsfn):
            for j in np.arange(gdat.numbsidepsfn):
                # solve p = matrdesi x coefspix for coefspix
                p = gdat.cntppsfnusam[a*gdat.factusam:(a+1)*gdat.factusam, j*gdat.factusam:(j+1)*gdat.factusam].flatten()
                gdat.coefspix[:, a, j] = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, p)) 
        
        if gdat.diagmode:
            if np.amax(gdat.coefspix) == 0.:
                raise Exception('')

        gdat.coefspix = gdat.coefspix[:, 1:12, 1:12]
        
        gdat.coefspix = gdat.coefspix.reshape(gdat.numbpararegr, gdat.numbside**2)
        
        if gdat.boolplotsave:
            figr, axis = plt.subplots()
            axis.imshow(gdat.cntppsfnusam, interpolation='nearest')
            plt.tight_layout()
            plt.savefig(gdat.pathdatartag + '%s_psfnusamp.%s' % (gdat.rtag, gdat.strgplotfile))
            plt.close()

    gdat.fitt.catl = {}
    gdat.refr.catl = [[dict() for o in gdat.indxtsec] for q in gdat.refr.indxcatl]
    
    gdat.liststrgfeat = ['tmag', 'labl']
    if gdat.typedata != 'toyy':
        gdat.liststrgfeat += ['tici', 'rasc', 'decl', 'pmde', 'pmra']
    
    if gdat.strgtargtype != 'posi':
        print('gdat.maxmdmag')
        print(gdat.maxmdmag)
    
    for q in gdat.refr.indxcatl:
        print('Constructing reference catalog %d...' % q)
        for o in gdat.indxtsec:
            if gdat.typedata == 'toyy':
                gdat.refr.catl[q][o]['xpos'] = np.array([gdat.xpostarg] + gdat.xposneig.tolist())
                gdat.refr.catl[q][o]['ypos'] = np.concatenate([np.array([gdat.ypostarg])] + [gdat.yposneig])
                gdat.refr.catl[q][o]['tmag'] = np.concatenate([np.array([gdat.tmagtarg])] + [gdat.tmagneig])
            else:
                gdat.refr.catl[q][o]['tmag'] = catalogData[:]['Tmag']
                if gdat.typedata != 'toyy':
                    gdat.refr.catl[q][o]['rasc'] = catalogData[:]['ra']
                    gdat.refr.catl[q][o]['decl'] = catalogData[:]['dec']
                    gdat.refr.catl[q][o]['tici'] = np.empty(len(catalogData), dtype=int)
                    gdat.refr.catl[q][o]['tici'][:] = catalogData[:]['ID']
                gdat.refr.catl[q][o]['pmde'] = catalogData[:]['pmDEC']
                gdat.refr.catl[q][o]['pmra'] = catalogData[:]['pmRA']
            gdat.refr.catl[q][o]['labl'] = np.empty(gdat.refr.catl[q][o]['tmag'].size, dtype=object)
            
            for strgfeat in gdat.liststrgfeat:
                gdat.refr.catl[q][o][strgfeat] = np.array(gdat.refr.catl[q][o][strgfeat])
            
            print('Number of sources in the reference catalog %d: %d' % (q, len(gdat.refr.catl[q][o]['tmag'])))
            
            if gdat.strgtargtype != 'posi':
                dmag = gdat.refr.catl[q][o]['tmag'] - gdat.refr.catl[q][o]['tmag'][0]
                indxsourbrgt = np.where(dmag < gdat.maxmdmag)[0]
                gdat.numbrefrbrgt = indxsourbrgt.size
                magtcutt = gdat.refr.catl[q][o]['tmag'][0] + gdat.maxmdmag
                print('%d of the reference catalog sources are brighter than the magnitude cutoff of %g.' % (gdat.numbrefrbrgt, magtcutt))
                
                if gdat.numbrefrbrgt < gdat.refr.catl[q][o]['tmag'].size:
                    print('Removing nearby sources that are %g mag fainter than the target...' % gdat.maxmdmag)
                    for strgfeat in gdat.liststrgfeat:
                        gdat.refr.catl[q][o][strgfeat] = gdat.refr.catl[q][o][strgfeat][indxsourbrgt]
                    print('Remaining number of reference sources: %d' % gdat.refr.catl[q][o]['tmag'].size)
            
            gdat.refr.catl[q][o]['cnts'] = retr_fluxfromtmag(gdat.refr.catl[q][o]['tmag']) * gdat.timeexpo
            
            gdat.refr.numbsour[q, o] = gdat.refr.catl[q][o]['cnts'].size

            if gdat.typedata != 'toyy':
                gdat.refr.cequ = np.empty((gdat.refr.numbsour[q, o], 2))
                gdat.refr.cequ[:, 0] = gdat.refr.catl[q][o]['rasc']
                gdat.refr.cequ[:, 1] = gdat.refr.catl[q][o]['decl']
    
            if gdat.epocpmot is not None:
                print('Correcting the TIC catalog for proper motion...')
                if gdat.rasctarg is not None:
                    pmra = gdat.pmratarg
                    pmde = gdat.pmdetarg
                    print('Using the user-provided values to correct for proper-motion...')
                else:
                    pmra = gdat.refr.catl[q][o]['pmra'][indxsourbrgt]
                    pmde = gdat.refr.catl[q][o]['pmde'][indxsourbrgt]
                    print('Using the TIC catalog to correct for proper-motion...')
                gdat.refr.catl[q][o]['rasc'] += pmra * (gdat.epocpmot - 2000.) / (1000. * 3600.)
                gdat.refr.catl[q][o]['decl'] += pmde * (gdat.epocpmot - 2000.) / (1000. * 3600.)

    
    #print('Removing nearby sources that are too close...')
    ## calculate angular distances
    #distangl = 180. * np.sqrt((gdat.refr.catl[q][o]['rasc'][None, :] - gdat.refr.catl[q][o]['rasc'][:, None])**2 + \
    #                   (gdat.refr.catl[q][o]['decl'][None, :] - gdat.refr.catl[q][o]['decl'][:, None])**2)
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
    #    if gdat.refr.catl[q][o][strgfeat]['tmag'][indxsort[0]] < gdat.refr.catl[q][o][strgfeat]['tmag'][indx[1]]:
    #        indxkill = indxsort[1]
    #    
    #    # determine the new indcices (i.e., without the one to be killed)
    #    indxtotl = np.arange(gdat.refr.catl[q][o]['rasc'].size)
    #    indxneww = np.setdiff1d(indxtotl, np.array(indxkill))
    #    
    #    # remove the faint source
    #    for strgfeat in gdat.liststrgfeat:
    #        gdat.refr.catl[q][o][strgfeat] = gdat.refr.catl[q][o][strgfeat][indxneww]
    #    
    #    # recalculate the distances
    #    distangl = np.sqrt((gdat.refr.catl[q][o]['rasc'][None, :] - gdat.refr.catl[q][o]['rasc'][:, None])**2 + \
    #                       (gdat.refr.catl[q][o]['decl'][None, :] - gdat.refr.catl[q][o]['decl'][:, None])**2)
    
    gdat.refr.numbtser = np.empty(gdat.numbtsec, dtype=int)
    gdat.refr.indxtser = [[] for o in gdat.indxtsec]
    gdat.refr.labltser = [[] for o in gdat.indxtsec]
    gdat.refr.colrlcur = [[] for o in gdat.indxtsec]
    
    if gdat.refrlistlabltser is not None:
        gdat.refr.labltser = gdat.refrlistlabltser
    else:
        # determine what reference light curve is available for the sector
        for o in gdat.indxtsec:
            gdat.refr.labltser[o] = []
            if gdat.typedata != 'toyy' and gdat.booltpxf[o]:
                gdat.refr.labltser[o] += ['SPOC']
    
            # number of reference light curves
            gdat.refr.numbtser[o] = len(gdat.refr.labltser[o])
            gdat.refr.indxtser[o] = np.arange(gdat.refr.numbtser[o])
    
            gdat.refr.colrlcur[o] = ['r', 'orange'][:gdat.refr.numbtser[o]]
        
    gdat.refrtime = [[[] for k in gdat.refr.indxtser[o]] for o in gdat.indxtsec]
    gdat.refrrflx = [[[] for k in gdat.refr.indxtser[o]] for o in gdat.indxtsec]
    gdat.stdvrefrrflx = [[[] for k in gdat.refr.indxtser[o]] for o in gdat.indxtsec]
    
    gdat.liststrgfeat += ['xpos', 'ypos']
    # write metadata to file
    gdat.pathsavemetaglob = gdat.pathdatatarg + 'metatarg.csv'
    dictmeta = dict()
    if gdat.typedata == 'toyy':
        dictmeta['xpos'] = gdat.xpostarg
        dictmeta['ypos'] = gdat.ypostarg
    if gdat.typedata != 'toyy':
        dictmeta['RA'] = gdat.rasctarg
        dictmeta['Dec'] = gdat.decltarg
    print('Writing to %s...' % gdat.pathsavemetaglob)
    objtfile = open(gdat.pathsavemetaglob, 'w')
    for key, value in dictmeta.items():
        objtfile.write('%s,%g\n' % (key, value))
    objtfile.close()

    # Boolean flag to indicate whether there is a reference time-series
    gdat.boolrefrtser = [[] for o in gdat.indxtsec]
    if gdat.refrarrytser is None:
        if gdat.typedata == 'toyy':
            for o in gdat.indxtsec:
                gdat.boolrefrtser[o] = False
        if gdat.typedata != 'toyy':
            cntr = 0
            for o in gdat.indxtsec:
                # get reference light curve
                if gdat.booltpxf[o]:
                    arry, indxtimequalgood, indxtimenanngood, tsecrefr, tcam, tccd = ephesus.read_tesskplr_file(listpathdownspoclcur[cntr], strgtype='PDCSAP_FLUX')
                    gdat.refrtime[o][0] = arry[:, 0]
                    gdat.refrrflx[o][0] = arry[:, 1]
                    gdat.stdvrefrrflx[o][0] = arry[:, 2]
                    gdat.boolrefrtser[o] = True
                    cntr += 1
                else:
                    gdat.boolrefrtser[o] = False
    else:
        for o in gdat.indxtsec:
            gdat.boolrefrtser[o] = True

    gdat.listtime = [[] for o in gdat.indxtsec]
    gdat.indxtime = [[] for o in gdat.indxtsec]
    
    gdat.listarry = [[] for o in gdat.indxtsec]
    gdat.numbtime = np.empty(gdat.numbtsec, dtype=int)
    
    arrytemp = np.linspace(0., float(gdat.numbside - 1), gdat.numbside)
    gdat.xposimag, gdat.yposimag = np.meshgrid(arrytemp, arrytemp)
    
    gdat.fitt.arryrflx = [[] for o in gdat.indxtsec]
    for o in gdat.indxtsec:
        strgsecc = retr_strgsecc(gdat, o)
        print('Sector: %d' % gdat.listtsec[o])
        print('Camera: %d' % gdat.listtcam[o])
        print('CCD: %d' % gdat.listtccd[o])
        if gdat.typedata != 'toyy':
            if gdat.booltpxf[o]:
                print('TPF data')
            else:
                print('FFI data')
        
        if gdat.boolplotcntp or gdat.boolplotrflx or gdat.boolanim:
            gdat.strgtitlcntpplot = '%s, Sector %d, Cam %d, CCD %d' % (gdat.labltarg, gdat.listtsec[o], gdat.listtcam[o], gdat.listtccd[o])
        
        if gdat.typedata != 'toyy':
            
            # get data
            ## read the FITS files
            #print(gdat.listhdundata[o][1].data.names)
            ## time
            gdat.listtime[o] = gdat.listhdundata[o][1].data['TIME'] + 2457000
            print('Number of raw data points: %d' % gdat.listtime[o].size)
            diff = gdat.listtime[o][1:] - gdat.listtime[o][:-1]
            
            ## count per pixel
            gdat.cntpdata = gdat.timeexpo * (gdat.listhdundata[o][1].data['FLUX'] + \
                                             gdat.listhdundata[o][1].data['FLUX_BKG']).swapaxes(0, 2).swapaxes(0, 1)
            
            if gdat.booltpxf[o]:
                gdat.numbside = gdat.cntpdata.shape[1]
            
            booldatagood = np.isfinite(gdat.listtime[o])
            if gdat.boolmaskqual:
                print('Masking bad data with quality flags...')
                booldatagood = booldatagood & (gdat.listhdundata[o][1].data['QUALITY'] == 0)
            if limttimeignoqual is not None:
                print('Ignoring the quality mask between %g and %g...' % (limttimeignoqual[0], limttimeignoqual[1]))
                booldatagood = booldatagood & ((limttimeignoqual[0] < gdat.listtime[o]) & (gdat.listtime[o] < limttimeignoqual[1]))
            print('Ignoring data with 0 counts...')
            booldatagood = booldatagood & (np.amax(np.amax(gdat.cntpdata, 0), 0) > 0.)
            indxtimedatagood = np.where(booldatagood)[0]
            fracgood = 100. * float(indxtimedatagood.size) / gdat.listtime[o].size
            print('Fraction of unmasked times: %.4g percent' % fracgood)
            if indxtimedatagood.size == 0:
                print('No good data found.')
                return gdat.dictoutp
    
            # keep good times and discard others
            gdat.listtime[o] = gdat.listtime[o][indxtimedatagood]
            gdat.cntpdata = gdat.cntpdata[:, :, indxtimedatagood]
        
        if gdat.typedata == 'toyy':
            if gdat.typecade[o] == '2min':
                difftime = 2. / 60. / 24. # [days]
            elif gdat.typecade[o] == '10mn':
                difftime = 10. / 60. / 24. # [days]
            else:
                raise Exception('')
            gdat.listtime[o] = np.arange(0., 30., difftime)
            gdat.numbtime[o] = gdat.listtime[o].size
            gdat.refr.catl[0][o]['cnts'] = retr_fluxfromtmag(gdat.refr.catl[0][o]['tmag']) * gdat.timeexpo
            gdat.refr.catl[0][o]['cnts'] = np.zeros((gdat.numbtime[o], gdat.refr.catl[0][o]['cnts'].size)) + gdat.refr.catl[0][o]['cnts']
        
        if gdat.typedata == 'toyy' or gdat.typedata == 'mock':
            # make mock count data
            gdat.refr.parapsfn = np.empty(1)
            gdat.refr.parapsfn[0] = 0.7 # [pixel]
            gdat.refr.fluxbackscal = 100. # [counts/s]
            gdat.refr.cntpbackscal = gdat.refr.fluxbackscal * gdat.timeexpo
            cntpdatatemp = retr_cntpmodl(gdat, gdat.refr.catl[0][o]['xpos'], gdat.refr.catl[0][o]['ypos'], gdat.refr.catl[0][o]['cnts'], gdat.refr.cntpbackscal, gdat.refr.parapsfn)
            
            gdat.cntpdata = np.random.poisson(cntpdatatemp).astype(float)
        
            gdat.refrtime[o] = gdat.listtime[o]
        
        if gdat.timeoffs != 0.:
            gdat.labltime = 'Time [BJD - %d]' % gdat.timeoffs
        else:
            gdat.labltime = 'Time [BJD]'

        gdat.numbtime[o] = gdat.listtime[o].size
        gdat.indxtime[o] = np.arange(gdat.numbtime[o])
        
        gdat.pathcbvs = gdat.pathdatalygo + 'cbvs/'
        if gdat.booldetrcbvs:
            path = gdat.pathcbvs + \
                     fnmatch.filter(os.listdir(gdat.pathcbvs), 'tess*-s%04d-%d-%d-*-s_cbv.fits' % (gdat.listtsec[o], gdat.listtcam[o], gdat.listtccd[o]))[0]
            print('Reading from %s...' % path)
            listhdun = astropy.io.fits.open(path)
            #listhdun.info()
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
                        axis.plot(gdat.listtime[o] - gdat.timeoffs, temp, ls='', marker='.', ms=1)
                        axis.set_ylabel('%s' % strgyaxi)
                        axis.set_xlabel(gdat.labltime)
                        path = gdat.pathimagtarg + '%s_%s_%02d.%s' % (strgplot, strgsecc, k, gdat.typefileplot)
                        print('Writing to %s...' % path)
                        plt.savefig(path)
                        plt.close()

        if gdat.listpixlaper is not None:
            gdat.cntpaper = np.zeros(gdta.numbtime[o])
            for pixlaper in gdat.listpixlaper:
                gdat.cntpaper += gdgat.cntpdata[0, pixlaper[0], pixlaper[1], :]
            gdat.cntpaper /= np.median(gdat.cntpaper)
        
        gdat.cntpdatatmed = np.median(gdat.cntpdata, 2)

        if len(os.listdir(gdat.pathcbvs)) == 0 and gdat.boolplotquat:
            print('Reading quaternions...')
            path = gdat.pathdatalygo + 'quat/'
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
                quat[:, k-1] = scipy.interpolate.interp1d(dataquat['TIME'] + 2457000,  dataquat[strg], fill_value=0, bounds_error=False)(gdat.listtime[o])
                minm = np.percentile(dataquat[strg], 0.05)
                maxm = np.percentile(dataquat[strg], 99.95)
                axis[k-1].plot(dataquat['TIME'] + 2457000 - gdat.timeoffs, dataquat[strg], ls='', marker='.', ms=1)
                axis[k-1].set_ylim([minm, maxm])
                axis[k-1].set_ylabel('$Q_{%d}$' % k)
            axis[2].set_xlabel(gdat.labltime)
            path = gdat.pathimagtarg + 'quat_sc%02d.%s' % (gdat.listtsec[o], gdat.typefileplot)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
        if not np.isfinite(gdat.cntpdata).all():
            print('Warning! Not all data are finite.')
        
        if gdat.catlextr is not None:
            for qq in gdat.refr.indxcatl:
                gdat.refr.catl[qq+1]['rasc'] = gdat.catlextr[qq]['rasc']
                gdat.refr.catl[qq+1]['decl'] = gdat.catlextr[qq]['decl']

        ## reference catalogs
        for q in gdat.refr.indxcatl:
            
            if gdat.typedata != 'toyy':
                gdat.refr.cpix = gdat.listobjtwcss[o].all_world2pix(gdat.refr.cequ, 0)
                gdat.refr.catl[q][o]['xpos'] = gdat.refr.cpix[:, 0]
                gdat.refr.catl[q][o]['ypos'] = gdat.refr.cpix[:, 1]
            
            ## indices of the reference catalog sources within the cutout
            indxsourwthn = np.where((gdat.refr.catl[q][o]['xpos'] > -0.5) & (gdat.refr.catl[q][o]['xpos'] < gdat.numbside - 0.5) & \
                                            (gdat.refr.catl[q][o]['ypos'] > -0.5) & (gdat.refr.catl[q][o]['ypos'] < gdat.numbside - 0.5))[0]
            
            print('Number of reference sources inside the cutout is %d. Discarding the rest...' % indxsourwthn.size)
            ## filter the reference catalog
            for strgfeat in gdat.liststrgfeat:
                gdat.refr.catl[q][o][strgfeat] = gdat.refr.catl[q][o][strgfeat][indxsourwthn]
            
            gdat.refr.catl[q][o]['cnts'] = retr_fluxfromtmag(gdat.refr.catl[q][o]['tmag']) * gdat.timeexpo
            
            print('gdat.refr.catl[q][o][xpos]')
            print(gdat.refr.catl[q][o]['xpos'])
            print('gdat.refr.catl[q][o][ypos]')
            print(gdat.refr.catl[q][o]['ypos'])
            print('gdat.refr.catl[q][o][tmag]')
            print(gdat.refr.catl[q][o]['tmag'])
            
        for q in gdat.refr.indxcatl:
            for k in gdat.refr.indxsour[q][o]:
                if gdat.typedata != 'toyy':
                    gdat.refr.catl[q][o]['labl'][k] = '%s' % 'TIC %s' % gdat.refr.catl[q][o]['tici'][k]
                else:
                    gdat.refr.catl[q][o]['labl'][k] = 'Mock Target %d' % k
            
            gdat.refr.numbsour[q, o] = gdat.refr.catl[q][o]['xpos'].size
            gdat.refr.indxsour[q][o] = np.arange(gdat.refr.numbsour[q, o])
            print('Number of reference sources is %d.' % gdat.refr.numbsour[q, o])
        
        if np.amin(gdat.cntpdata) < 0:
            print('Minimum of the image is negative.')

        # data variance
        gdat.vari = np.copy(gdat.cntpdata)

        ## fitting catalog
        if gdat.typedata == 'toyy':
            for name in ['xpos', 'ypos', 'cnts']:
                gdat.fitt.catl[name] = gdat.refr.catl[0][o][name]
        if gdat.typedata != 'toyy':
            if gdat.strgtargtype == 'posi':
                gdat.fitt.catl['rasc'] = np.empty(1 + len(gdat.refr.catl[0][o]['rasc']))
                gdat.fitt.catl['decl'] = np.empty(1 + len(gdat.refr.catl[0][o]['rasc']))
                gdat.fitt.catl['cntsesti'] = np.empty(1 + len(gdat.refr.catl[0][o]['rasc']))
                gdat.fitt.catl['rasc'][0] = gdat.rasctarg
                gdat.fitt.catl['decl'][0] = gdat.decltarg
                gdat.fitt.catl['cntsesti'][0] = 1e1
                gdat.fitt.catl['rasc'][1:] = gdat.refr.catl[0][o]['rasc']
                gdat.fitt.catl['decl'][1:] = gdat.refr.catl[0][o]['decl']
                gdat.fitt.catl['cntsesti'][1:] = gdat.refr.catl[0][o]['cnts']
            else:
                gdat.fitt.catl['rasc'] = gdat.refr.catl[0][o]['rasc']
                gdat.fitt.catl['decl'] = gdat.refr.catl[0][o]['decl']
                gdat.fitt.catl['cntsesti'] = gdat.refr.catl[0][o]['cnts']
            
            skyyfitttemp = np.empty((gdat.fitt.catl['rasc'].size, 2))
            skyyfitttemp[:, 0] = gdat.fitt.catl['rasc']
            skyyfitttemp[:, 1] = gdat.fitt.catl['decl']
            if gdat.fitt.catl['rasc'].size == 0:
                raise Exception('')
            # transform sky coordinates into dedector coordinates and filter
            posifitttemp = gdat.listobjtwcss[o].all_world2pix(skyyfitttemp, 0)
            gdat.fitt.catl['xpos'] = posifitttemp[:, 0]
            gdat.fitt.catl['ypos'] = posifitttemp[:, 1]
        
        print('Merging fitting source pairs that are too close')
        while True:
            
            #print('Iteration %d...' % n)
            
            # for each source, find the distance to all other sources
            dist = np.sqrt((gdat.fitt.catl['xpos'][:, None] - gdat.fitt.catl['xpos'][None, :])**2 + (gdat.fitt.catl['ypos'][:, None] - gdat.fitt.catl['ypos'][None, :])**2)
            dist[range(dist.shape[0]), range(dist.shape[0])] = 1e10
            # find the index of the closest neighbor
            n, m = np.unravel_index(np.argmin(dist), dist.shape)
            #n, m = np.argmin(dist)
            
            if dist[n, m] < 0.5:
                
                cnts = gdat.fitt.catl['cntsesti']
                gdat.fitt.catl['xpos'][n] = (cnts[n] * gdat.fitt.catl['xpos'][n] + cnts[m] * gdat.fitt.catl['xpos'][m]) / (cnts[n] + cnts[m])
                gdat.fitt.catl['ypos'][n] = (cnts[n] * gdat.fitt.catl['ypos'][n] + cnts[m] * gdat.fitt.catl['ypos'][m]) / (cnts[n] + cnts[m])
                gdat.fitt.catl['cntsesti'][n] = gdat.fitt.catl['cntsesti'][n] + gdat.fitt.catl['cntsesti'][m]
                
                # delete the close source
                gdat.fitt.catl['xpos'] = np.delete(gdat.fitt.catl['xpos'], m)
                gdat.fitt.catl['ypos'] = np.delete(gdat.fitt.catl['ypos'], m)
                gdat.fitt.catl['cntsesti'] = np.delete(gdat.fitt.catl['cntsesti'], m)
                
            else:
                break
        
        if gdat.fitt.catl['xpos'].size == 0:
            print('No fitting source found...')
            print('')
            return gdat.dictoutp
        
        gdat.fitt.numbsour = gdat.fitt.catl['xpos'].size
        gdat.indxsour = np.arange(gdat.fitt.numbsour)
            
        print('Number of point sources in the model: %d' % gdat.fitt.numbsour)
    
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

        # number of components, 1 for background, 3 for quaternions
        gdat.fitt.numbcomp = gdat.fitt.numbsour + 1# + 3
        gdat.indxcomp = np.arange(gdat.fitt.numbcomp)

        gdat.fitt.arryrflx[o] = np.empty((gdat.numbtime[o], gdat.fitt.numbcomp, 3, 3, 3))
        
        gdat.stdvfittcnts = np.empty((gdat.numbtime[o], gdat.fitt.numbcomp))
        
        if gdat.boolcalcconr and not (gdat.ticitarg is not None):
            raise Exception

        strgsave = retr_strgsave(gdat, strgsecc, 1, 1, o)
        
        if gdat.typepsfninfe != 'fixd':
            # fit for the PSF
            numbsampwalk = 200
            numbsampburnwalk = int(0.1 * numbsampwalk)
            numbsampburnwalkseco = int(0.5 * numbsampwalk)
            
            # plot the median image
            if gdat.boolplotcntp:
                for typecntpscal in gdat.listtypecntpscal:
                    nameplotcntp = 'cntpdatatmed'
                    strgtitl = gdat.strgtitlcntpplot
                    plot_cntp(gdat, gdat.cntpdatatmed, o, typecntpscal, nameplotcntp, strgsave, strgtitl=strgtitl)

            gdat.typefittpsfnposi = 'fixd'
            gdat.typefittpsfncnts = 'vari'
            
            listlablpara = []
            listminmpara = []
            listmaxmpara = []
            
            if gdat.typefittpsfnposi == 'vari':
                for k in gdat.indxsour:
                    listlablpara.append(['$x_{%d}$' % k, ''])
                    listlablpara.append(['$y_{%d}$' % k, ''])
            
            if gdat.typefittpsfncnts == 'vari':
                for k in gdat.indxsour:
                    listminmpara.append(0.)
                    listmaxmpara.append(np.amax(gdat.cntpdatatmed))
                    listlablpara.append(['$F_{%d}$' % k, 'e$^-$/s'])
            
            listminmpara.append(0.)
            listmaxmpara.append(np.amax(gdat.cntpdatatmed))
            listlablpara.append(['$B$', 'e$^-$/s/px'])
            
            if gdat.typepsfnshap == 'gaus':
                listminmpara.append(0.)
                listmaxmpara.append(2.)
                listlablpara.append(['$\sigma$', 'px'])
            
            gdat.numbparapsfn = len(listlablpara)
            gdat.indxparapsfn = np.arange(gdat.numbparapsfn)
            listscalpara = ['self' for k in gdat.indxparapsfn]
            listmeangauspara = None
            liststdvgauspara = None
            
            listminmpara = np.array(listminmpara)
            listmaxmpara = np.array(listmaxmpara)
            
            strgextn = 'psfn'
            numbdata = gdat.numbpixl
            strgsaveextnmcmc = gdat.pathdatatarg + gdat.typepsfnshap + '_' + gdat.strgcnfg + '.txt'
            listparapost, _ = tdpy.samp(gdat, gdat.pathimagtargpsfn, numbsampwalk, numbsampburnwalk, numbsampburnwalkseco, retr_llik, \
                                            listlablpara, listscalpara, listminmpara, listmaxmpara, listmeangauspara, liststdvgauspara, \
                                                numbdata, strgextn=strgextn, typefileplot=typefileplot, strgsaveextn=strgsaveextnmcmc)
            
            paramedi = np.median(listparapost, 0)
            coefmedi = paramedi[:gdat.numbcoef]
            cntpbackmedi = paramedi[gdat.numbcoef]
            if gdat.typepsfnshap == 'gfreffix':
                ampltotlmedi = paramedi[gdat.numbcoef+1]
            elif gdat.typepsfnshap == 'gfrefinf':
                ampltotlmedi = paramedi[gdat.numbcoef+1]
                amplrelamedi = paramedi[gdat.numbcoef+2:]
            elif gdat.typepsfnshap == 'gfreffre':
                cntsmedi = paramedi[gdat.numbcoef+1:]
                
            if gdat.boolplotcntp:
                # plot the posterior median PSF model
                xpossour = np.array([(gdat.numbside - 1.) / 2.])
                ypossour = np.array([(gdat.numbside - 1.) / 2.])
                cnts = np.array([[1.]])
                cntpback = np.zeros(1)
                gdat.cntpmodlpsfn = retr_cntpmodl(gdat, xpossour, ypossour, cnts, cntpback, gdat.para)
                nameplot = '%scntpmodlpsfnmedi' % (gdat.pathimagtargpsfn)
                strgtitl = gdat.strgtitlcntpplot
                for typecntpscal in gdat.listtypecntpscal:
                    plot_cntp(gdat, gdat.cntpmodlpsfn[:, :, 0], o, typecntpscal, nameplotcntp, strgsave, strgtitl=strgtitl, boolanno=False)

                # plot the posterior median image model
                if gdat.typepsfnshap == 'gfreffix':
                    cnts = gdat.cntsfitt * amplmedi
                elif gdat.typepsfnshap == 'gfrefinf':
                    cnts = gdat.cntsfitt * ampltotlmedi * amplrelamedi
                elif gdat.typepsfnshap == 'gfreffre':
                    cnts = cntsmedi
                gdat.cntpmodlpsfn = retr_cntpmodl(gdat, gdat.fitt.catl['xpos'], gdat.fitt.catl['ypos'], cnts, cntpbackmedi, gdat.para)
                
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
            # get time string
            if gdat.typedata != 'toyy':
                objttime = astropy.time.Time(gdat.listtime[o], format='jd', scale='utc', out_subfmt='date_hm')
                listtimelabl = objttime.iso
            else:
                listtimelabl = gdat.listtime[o].astype(str)
        
        gdat.offs = 0.2
        gdat.listoffsxpos = np.linspace(-gdat.offs, gdat.offs, 3)
        gdat.listoffsypos = np.linspace(-gdat.offs, gdat.offs, 3)
        gdat.numboffs = gdat.listoffsxpos.size
        gdat.indxoffs = np.arange(gdat.numboffs)
        for x in gdat.indxoffs:
            for y in gdat.indxoffs:
                
                if not gdat.boolfittoffs and (x != 1 or y != 1):
                    continue

                strgsave = retr_strgsave(gdat, strgsecc, x, y, o)
                pathsaverflxinit = gdat.pathdatatarg + 'rflxinit' + strgsave + '.csv'
                pathsaverflx = gdat.pathdatatarg + 'rflx' + strgsave + '.csv'
                pathsaverflxtarg = gdat.pathdatatarg + 'rflxtarg' + strgsave + '.csv'
                pathsavemeta = gdat.pathdatatarg + 'metaregr' + strgsave + '.csv'
        
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
                    path = gdat.pathimagtarg + 'histcntpdata_%s.%s' % (strgsave, gdat.typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
        
                # introduce the positional offset
                xpostemp = np.copy(gdat.fitt.catl['xpos'])
                ypostemp = np.copy(gdat.fitt.catl['ypos'])
                xpostemp[0] = gdat.fitt.catl['xpos'][0] + gdat.listoffsxpos[x]
                ypostemp[0] = gdat.fitt.catl['ypos'][0] + gdat.listoffsypos[y]
                
                if not os.path.exists(pathsaverflx):
                    
                    timeinit = timemodu.time()
                    gdat.covafittcnts = np.empty((gdat.numbtime[o], gdat.fitt.numbsour + 1, gdat.fitt.numbsour + 1))
                    gdat.mlikfittcntsinit = np.empty((gdat.numbtime[o], gdat.fitt.numbsour + 1))
                    matrdesi = np.ones((gdat.numbpixl, gdat.fitt.numbsour + 1))
                    
                    print('Solving for the best-fit raw light curves of the sources...')
                    cntstemp = np.ones((1, gdat.fitt.numbsour))
                    cntptemp = np.zeros(1)
                    for k in np.arange(gdat.fitt.numbsour):
                        matrdesi[:, k] = retr_cntpmodl(gdat, xpostemp[k, None], ypostemp[k, None], cntstemp[:, k, None], cntptemp, gdat.para).flatten()
                            
                    # solve the linear system
                    for t in gdat.indxtime[o]:
                        gdat.mlikfittcntsinit[t, :], gdat.covafittcnts[t, :, :] = retr_mlikregr(gdat.cntpdata[:, :, t], matrdesi, gdat.vari[:, :, t])
                    if not np.isfinite(gdat.covafittcnts).all():
                        indxbaddpixl = (~np.isfinite(gdat.covafittcnts)).any(0)
                        indxbaddtime = (~np.isfinite(gdat.covafittcnts)).any(1).any(1)
                        print('indxbaddtime')
                        summgene(indxbaddtime)
                        print('gdat.vari')
                        summgene(gdat.vari)
                        print('gdat.cntpdata')
                        summgene(gdat.cntpdata)
                        print('matrdesi')
                        summgene(matrdesi)
                        raise Exception('')

                    for k in gdat.indxcomp:
                        gdat.stdvfittcnts[:, k] = np.sqrt(gdat.covafittcnts[:, k, k])
                        
                        if not np.isfinite(gdat.stdvfittcnts[:, k]).all():
                            print('temp: error went NaN because of negative covariance. Reseting error to 1e-2')
                            gdat.stdvfittcnts[:, k] = 1e-2
                            print('k')
                            print(k)
                            print('gdat.covafittcnts[:, k, k]')
                            summgene(gdat.covafittcnts[:, k, k])
                            print('np.isfinite(gdat.stdvfittcnts[:, k])')
                            summgene(np.isfinite(gdat.stdvfittcnts[:, k]))
                            #raise Exception('')
                    
                    gdat.varifittcnts = gdat.stdvfittcnts**2
        
                    if gdat.booldetrcbvs:
                        gdat.mlikfittcnts = np.copy(gdat.mlikfittcntsinit)
                        # subtract CBVs
                        print('Solving for the detrended target light curve using the CBVs and the raw light curve...')
                        gdat.mlikamplcbvs, gdat.covaamplcbvs = retr_mlikregr(gdat.mlikfittcnts[:, 0], gdat.cbvstmpt, gdat.varifittcnts[:, 0])
                        rflxcbvs = gdat.mlikamplcbvs[None, :] * gdat.cbvstmpt
                        rflxcbvstotl = np.sum(rflxcbvs[:, :-1], 1)
                        gdat.mlikfittcnts[:, 0] -= rflxcbvstotl
                    else:
                        gdat.mlikfittcnts = gdat.mlikfittcntsinit
                
                    timefinl = timemodu.time()
                    print('Done in %g seconds.' % (timefinl - timeinit))
                    
                    gdat.medifittcnts = np.median(gdat.mlikfittcnts, 0)
                    print('Median flux of the central source is %g ADU.' % gdat.medifittcnts[0])
                    
                    # normalize fluxes to get relative fluxes
                    print('Normalizing by the median flux...')
                    gdat.fitt.arryrflx[o][:, :, 0, x, y] = gdat.listtime[o][:, None]
                    gdat.fitt.arryrflx[o][:, :, 1, x, y] = gdat.mlikfittcnts / gdat.medifittcnts[None, :]
                    gdat.fitt.arryrflx[o][:, :, 2, x, y] = gdat.stdvfittcnts / gdat.medifittcnts[None, :]
                    
                    if gdat.booldiagmode:
                        for a in range(gdat.listtime[o].size):
                            if a != gdat.listtime[o].size - 1 and gdat.listtime[o][a] >= gdat.listtime[o][a+1]:
                                raise Exception('')

                    # write the light curve to file
                    print('Writing meta data to %s...' % pathsavemeta)
                    arry = gdat.medifittcnts
                    np.savetxt(pathsavemeta, arry, delimiter=',', header='Temporal median counts for each component')
                    
                    arry = np.empty((gdat.numbtime[o], 2*gdat.fitt.numbcomp+1))
                    arry[:, 0] = gdat.listtime[o]
                    for k in gdat.indxcomp:
                        arry[:, 2*k+1] = gdat.fitt.arryrflx[o][:, k, 1, x, y]
                        arry[:, 2*k+2] = gdat.fitt.arryrflx[o][:, k, 2, x, y]
                    print('Writing all light curves to %s...' % pathsaverflx)
                    np.savetxt(pathsaverflx, arry, delimiter=',', header=gdat.strghead)
                    
                    print('Writing the target light curve to %s...' % pathsaverflxtarg)
                    np.savetxt(pathsaverflxtarg, arry[:, :3], delimiter=',', header=gdat.strghead)
                    gdat.listarry[o] = arry[:, :3]

                    if gdat.booldiagmode:
                        for a in range(gdat.listarry[o][:, 0].size):
                            if a != gdat.listarry[o][:, 0].size - 1 and gdat.listarry[o][a, 0] >= gdat.listarry[o][a+1, 0]:
                                raise Exception('')

                else:
                    print('Skipping the regression...')
                
                    print('Reading from %s...' % pathsavemeta)
                    gdat.medifittcnts = np.loadtxt(pathsavemeta, delimiter=',', skiprows=1)
                    
                    print('Reading from %s...' % pathsaverflx)
                    arry = np.loadtxt(pathsaverflx, delimiter=',', skiprows=1)
                    gdat.listtime[o] = arry[:, 0]
                    for k in gdat.indxcomp:
                        gdat.fitt.arryrflx[o][:, k, 0, x, y] = arry[:, 0]
                        try:
                            gdat.fitt.arryrflx[o][:, k, 1, x, y] = arry[:, 2*k+1]
                        except:
                            print('arry')
                            summgene(arry)
                            print('gdat.fitt.arryrflx[o][:, k, 1, x, y]')
                            summgene(gdat.fitt.arryrflx[o][:, k, 1, x, y])
                            raise Exception('')
                        gdat.fitt.arryrflx[o][:, k, 2, x, y] = arry[:, 2*k+2]
                    gdat.mlikfittcnts = gdat.medifittcnts[None, :] * gdat.fitt.arryrflx[o][:, :, 1, x, y]
                    gdat.listarry[o] = arry[:, :3]
                
                print('Evaluating the model at all time bins...')
                cntpbackneww = gdat.mlikfittcnts[:, -1]
                timeinit = timemodu.time()
                gdat.cntpmodl = retr_cntpmodl(gdat, xpostemp, ypostemp, gdat.mlikfittcnts[:, :-1], cntpbackneww, gdat.para)
                
                timefinl = timemodu.time()
                print('Done in %g seconds.' % (timefinl - timeinit))
                    
                gdat.cntpdatanbkg = gdat.cntpdata - gdat.mlikfittcnts[None, None, :, -1]
                gdat.cntpresi = gdat.cntpdata - gdat.cntpmodl
                
                chi2 = np.mean(gdat.cntpresi**2 / gdat.cntpdata) + 2 * gdat.fitt.numbsour
                
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
                
                if gdat.boolplotrflx:
                    print('Plotting light curves...')
                    if gdat.listlimttimetzom is not None:
                        gdat.indxtimelimt = []
                        for limttimeplot in gdat.listlimttimetzom:
                            gdat.indxtimelimt.append(np.where((gdat.listtime[o] > limttimeplot[0]) & (gdat.listtime[o] < limttimeplot[1]))[0])
                    
                    # plot light curve derived from aperture photometry
                    if gdat.listpixlaper is not None:
                        plot_lcur(gdat, gdat.cntpaper, 0.01 * gdat.cntpaper, k, o, '_' + strgsecc, '_aper', strgsave, 'aper')
                        
                    if gdat.booldetrcbvs:
                        plot_lcur(gdat, gdat.fitt.arryrflx[o][:, 0, 1, x, y], gdat.fitt.arryrflx[o][:, 0, 2, x, y], 0, o, '_' + strgsecc, strgsave, 'detrcbvs')
                        
                    # plot the light curve of the sources and background
                    for k in gdat.indxcomp:
                        
                        if x == 1 and y == 1 or k == 0:

                            plot_lcur(gdat, gdat.fitt.arryrflx[o][:, k, 1, x, y], gdat.fitt.arryrflx[o][:, k, 2, x, y], k, o, '_' + strgsecc, strgsave, 'stan')
                            if gdat.boolrefrtser[o] and x == 1 and y == 1:
                                plot_lcurcomp(gdat, gdat.fitt.arryrflx[o][:, k, 1, x, y], gdat.fitt.arryrflx[o][:, k, 2, x, y], k, o, '_' + strgsecc, strgsave, 'stan')
                            
                            if gdat.listlimttimetzom is not None:
                                for p in range(len(gdat.listlimttimetzom)):
                                    plot_lcur(gdat, gdat.fitt.arryrflx[o][:, k, 1, x, y], gdat.fitt.arryrflx[o][:, k, 2, x, y], k, o, '_' + strgsecc, \
                                                                                          strgsave, 'zoom', indxtimelimt=gdat.indxtimelimt[p], indxtzom=p)
                            
        # temporal types of image plots
        ## medians
        listtypeplotcntp = []
        if gdat.boolplotcntp:
            listtypeplotcntp += ['medi']
        # cadence frames
        if gdat.boolanim:
            listtypeplotcntp += ['anim']
        
        strgsave = retr_strgsave(gdat, strgsecc, 1, 1, o)
                
        for typeplotcntp in listtypeplotcntp:
            for nameplotcntp in listnameplotcntp:
                
                # make animation plot
                pathanim = retr_pathvisu(gdat, nameplotcntp, strgsave, typevisu='anim')
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
                        plot_cntp(gdat, cntptemp, o, typecntpscal, nameplotcntp, strgsave, strgtitl=strgtitl, boolresi=boolresi)
                    if typeplotcntp == 'anim':
                        args = [gdat, cntptemp, o, typecntpscal, nameplotcntp, strgsave]
                        kwag = { \
                                'boolresi': boolresi, 'listindxpixlcolr': gdat.listpixlaper, \
                                'listtimelabl':listtimelabl, \
                                'vmin':vmin, 'vmax':vmax, \
                                'lcur':gdat.fitt.arryrflx[o][:, 0, 0, x, y], 'time':gdat.listtime[o]}
                        listpath = anim_cntp(*args, **kwag)
                
                        # make animation
                        cmnd = 'convert -delay 20 '
                        for path in listpath:
                            cmnd += '%s ' % path
                        cmnd += '%s' % pathanim
                        os.system(cmnd)
                        
                        # delete images
                        for path in listpath:
                            os.system('rm %s' % path)
     
    if gdat.boolmile:
        # call miletos to analyze and/or model the light curve
        listarrytser = dict()
        
        if dictmileinpt is None:
            dictmileinpt = dict()
        
        dictmileinpt['listtsec'] = gdat.listtsec
        dictmileinpt['timeoffs'] = gdat.timeoffs

        for x in gdat.indxoffs:
            for y in gdat.indxoffs:
                if x != 1 or y != 1:
                    continue
                dictmileinpt['strgcnfg'] = 'n%03d_d%3.1f_of%d%d' % (gdat.numbside, gdat.maxmdmag, x, y)
                dictmileinpt['labltarg'] = gdat.labltarg
                dictmileinpt['pathdatatarg'] = gdat.pathdatatarg
                dictmileinpt['pathimagtarg'] = gdat.pathimagtarg
                if not gdat.boolfittoffs and (x != 1 or y != 1):
                    continue
                listarrytser['raww'] = [[[[] for o in gdat.indxtsec]], []]
                for o in gdat.indxtsec:
                    listarrytser['raww'][0][0][o] = gdat.fitt.arryrflx[o][:, 0, :, x, y]
                dictmileinpt['listarrytser'] = listarrytser
                print('Calling miletos.init() to analyze and model the data for the target...')
                dictmileoutp = miletos.init(**dictmileinpt)
                        
    timefinltotl = timemodu.time()
    print('Total execution time: %g seconds.' % (timefinltotl - timeinittotl))
    print('')                
    gdat.dictoutp['strgtarg'] = gdat.strgtarg
    gdat.dictoutp['listarry'] = gdat.listarry
    
    return gdat.dictoutp


