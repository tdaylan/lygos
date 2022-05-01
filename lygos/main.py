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
    
    flux = 10**(0.4 * (20.4 - tmag)) # [e-/s]
    
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
        
        # make title
        strgtitl = gdat.strgtitlcntpplot
        if listtimelabl[t] is not None:
            strgtitl += ', %s' % listtimelabl[t]
        
        path = plot_cntp(gdat, cntp[:, :, t], o, typecntpscal, gdat.pathimagtarg, nameplotcntp, strgsave, 'fitt', \
                                                strgtitl=strgtitl, boolresi=boolresi, listindxpixlcolr=listindxpixlcolr, \
                                                                                            timelabl=listtimelabl[t], thistime=time[t], indxtimeplot=t, \
                                                                                                vmin=vmin, vmax=vmax, lcur=lcur, time=time)
        
        listpath.append(path)

    return listpath


def plot_histcntp(gdat, cntp, pathimag, strgvarb, strgsave):
    
    figr, axis = plt.subplots(figsize=(8, 3.5))
    bins = np.logspace(np.arcsinh(np.amin(cntp)), np.arcsinh(np.amax(cntp)), 200)
    axis.hist(cntp.flatten(), bins=bins)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_ylabel('N')
    axis.set_xlabel('C [e$^{-}$/s]')
    plt.tight_layout()
    path = pathimag + 'hist%s%s.%s' % (strgvarb, strgsave, gdat.typefileplot)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()


def plot_cntp(gdat, \
              cntp, \
              o, \
              typecntpscal, \
              pathbase, \
              nameplotcntp, \
              strgsave, \
              typecatlplot, \
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
    
    path = retr_pathvisu(gdat, pathbase, nameplotcntp, strgsave, typecntpscal=typecntpscal, indxpcol=indxpcol, indxtimeplot=indxtimeplot, typecatlplot=typecatlplot)

    if os.path.exists(path):
        print('Plot exists at %s. Skipping the plotting...' % path)
    else:
        if boolresi:
            cbar = 'PuOr'
        
            if vmin is None:
                vmax = max(np.amax(cntp), abs(np.amin(cntp)))
                vmin = -vmax

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
            if typecatlplot == 'refr' or typecatlplot == 'refrfitt':
            
                for q in gdat.refr.indxcatl:
                    # reference sources
                    axis[0].scatter(gdat.refr.catl[q][o]['xpos'][gdat.indxsourwthn[q][o]], gdat.refr.catl[q][o]['ypos'][gdat.indxsourwthn[q][o]], \
                                                                                                        alpha=1., s=gdat.sizeplotsour, color='r', marker='o')
                    # label the reference sources
                    for indxtemp in gdat.indxsourwthn[q][o]:
                        axis[0].text(gdat.refr.catl[q][o]['xpos'][indxtemp] + 0.5, gdat.refr.catl[q][o]['ypos'][indxtemp] + 0.5, gdat.refr.catl[q][o]['labl'][indxtemp], color='r')
                    
            if typecatlplot == 'fitt' or typecatlplot == 'refrfitt':
            
                # fitting sources
                xposfitt = np.copy(gdat.fitt.catl['xpos'])
                yposfitt = np.copy(gdat.fitt.catl['ypos'])
                if xposoffs is not None:
                    # add the positional offset, if any
                    xposfitt[0] += xposoffs
                    yposfitt[0] += yposoffs
                ## target
                axis[0].scatter(xposfitt[0], yposfitt[0], alpha=1., color='b', s=gdat.sizeplotsour, marker='o')
                ## neighbors
                axis[0].scatter(xposfitt[1:], yposfitt[1:], alpha=1., s=gdat.sizeplotsour, color='b', marker='o')
                for k in gdat.indxsour:
                    axis[0].text(xposfitt[k] + 0.5, yposfitt[k] + 0.5, '%d' % k, color='b')
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
            axis[1].set_ylabel(gdat.lablyaxilcur)
            axis[1].axvline(thistime - gdat.timeoffs)
        
        print('Writing to %s...' % path)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    
    return path


def plot_lcurcomp(gdat, lcurmodl, stdvlcurmodl, k, indxtsecplot, strgsecc, strgsave, strgplot, timeedge=None, \
                    strgextn='', indxtimelimt=None, indxtzom=None, boolerrr=False):
    
    if k == 0:
        lablcomp = ', Target'
    elif k == gdat.fitt.numbcomp - 1:
        lablcomp = ', Background'
    else:
        lablcomp = ', Neighbor %d' % k
    
    timedatatemp = np.copy(gdat.listtime[indxtsecplot])
    timerefrtemp = [[] for q in gdat.refr.indxtser[indxtsecplot]] 
    for q in gdat.refr.indxtser[indxtsecplot]:
        timerefrtemp[q] = gdat.refrtime[indxtsecplot][q]
    
    if indxtimelimt is not None:
        timedatatemp = timedatatemp[indxtimelimt]
        stdvlcurmodl = stdvlcurmodl[indxtimelimt]
        lcurmodl = lcurmodl[indxtimelimt]

    nameplot = 'rflx_%s' % (strgplot)
    path = retr_pathvisu(gdat, gdat.pathimagtarg, nameplot, strgsave + '_s%03d' % k, indxtzom=indxtzom)
    
    # skip the plot if it has been made before
    if os.path.exists(path):
        print('Plot exists at %s. Skipping the plotting...' % path)
        return

    figr, axis = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2, 0.7]}, figsize=(12, 8))
    axis[1].set_xlabel(gdat.labltime)
    
    axis[0].set_ylabel(gdat.lablyaxilcur)
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


    
def plot_lcur(gdat, lcurmodl, stdvlcurmodl, k, indxtsecplot, strgsecc, strgsave, strgplot, cdpp=None, \
                    timeedge=None, \
                    strgextn='', indxtimelimt=None, indxtzom=None, boolerrr=False):
    
    if k == 0:
        lablcomp = ', Target source'
    elif k == gdat.fitt.numbcomp - 1:
        lablcomp = ', Background'
    else:
        lablcomp = ', Neighbor Source %d' % k
    if not cdpp is None:
        lablcomp += ', 1-hr CDPP = %.1f ppm' % cdpp

    timedatatemp = np.copy(gdat.listtime[indxtsecplot])
    
    if indxtimelimt is not None:
        timedatatemp = timedatatemp[indxtimelimt]
        stdvlcurmodl = stdvlcurmodl[indxtimelimt]
        lcurmodl = lcurmodl[indxtimelimt]

    nameplot = 'rflx_%s' % (strgplot)
    path = retr_pathvisu(gdat, gdat.pathimagtarg, nameplot, strgsave + '_s%03d' % k, indxtzom=indxtzom)
    
    # skip the plot if it has been made before
    if os.path.exists(path):
        print('Plot exists at %s. Skipping the plotting...' % path)
        return

    figr, axis = plt.subplots(figsize=(8, 2.5))
    axis.set_xlabel(gdat.labltime)
    
    axis.set_ylabel(gdat.lablyaxilcur)
    
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
                  pathbase, \
                  nameplot, \
                  # data
                  strgsave, \
                  
                  # rflx 
                  # temporal zoom
                  indxtzom=None, \
                    
                  typecatlplot=None, \

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
    
    if typecatlplot is None:
        strgtypecatlplot = ''
    else:
        strgtypecatlplot = '_catl' + typecatlplot

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
    
    pathvisu = pathbase + '%s%s%s%s%s%s%s.%s' % (nameplot, strgsave, strgscal, strgtzom, strgpcol, strgtime, strgtypecatlplot, typefileplot)
    
    return pathvisu


def retr_cntpmodl(gdat, strgmodl, xpos, ypos, cnts, cntpbackscal, parapsfn):
    '''
    Calculate the model image
    '''
    
    gmod = getattr(gdat, strgmodl)
    
    if gdat.typepsfnsubp == 'eval':
         
        if cnts.ndim == 2:
            cntpmodl = np.zeros((gdat.numbside, gdat.numbside, cnts.shape[0])) + cntpbackscal[None, None, :]
        else:
            cntpmodl = np.zeros((gdat.numbside, gdat.numbside)) + cntpbackscal
            
        for k in range(xpos.size):
            deltxpos = gdat.xposimag - xpos[k]
            deltypos = gdat.yposimag - ypos[k]
            if gmod.typepsfnshap == 'gauscirc':
                psfnsour = np.exp(-(deltxpos / parapsfn[0])**2 - (deltypos / parapsfn[0])**2)
            if gmod.typepsfnshap.startswith('gauselli'):
                gausbvar = np.exp(-0.5 * (deltxpos / parapsfn[0])**2 - 0.5 * (deltypos / parapsfn[1])**2)
                psfnsour = gausbvar + parapsfn[2] * deltxpos * gausbvar + parapsfn[3] * deltypos * gausbvar
                psfnsour[psfnsour < 0] = 0.
            if gmod.typepsfnshap == 'empi':
                z = parapsfn.reshape((gdat.numbsidepsfnfitt, gdat.numbsidepsfnfitt))
                psfnsour = scipy.interpolate.interp2d(gdat.xposimagpsfn, gdat.yposimagpsfn, z, kind='cubic', fill_value=0.)(gdat.xposimag, gdat.yposimag)

            if gmod.typepsfnshap == 'pfre':
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
        
        if gdat.booldiag:
            if not np.isfinite(cntpmodl).all():
                print('WARNING: cntpmodl was infinite!!!!')
                raise Exception('')

    return cntpmodl


def retr_dictderipsfn(para, gdat):
    
    dictvarbderi = dict()
    
    if gdat.typefittpsfnposi == 'fixd':
        xpos = gdat.fitt.catl['xpos']
        ypos = gdat.fitt.catl['ypos']
    if gdat.typefittpsfnposi == 'vari':
        xpos = para[gdat.indxparapsfnxpos] + gdat.fitt.catl['xpos']
        ypos = para[gdat.indxparapsfnypos] + gdat.fitt.catl['ypos']
    
    cnts = para[gdat.indxparapsfnflux]
    cntpbackscal = para[gdat.indxparapsfnback]
    parapsfn = para[gdat.indxparapsfnpsfn]
    
    cntpmodl = retr_cntpmodl(gdat, 'fitt', xpos, ypos, cnts, cntpbackscal, parapsfn)
    dictvarbderi['cntpmodlsexp'] = cntpmodl
    
    cntpmodlpnts = retr_cntpmodl(gdat, 'fitt', xpos, ypos, cnts, 0., parapsfn)
    dictvarbderi['cntpmodlpntssexp'] = cntpmodlpnts
    
    dictvarbderi['cntpresisexp'] = gdat.cntpdatasexp - cntpmodl
    
    if xpos.size > 1:
        dictvarbderi['raticont'] = retr_raticont(gdat, xpos, ypos, cnts, parapsfn)

    if xpos[0] == (gdat.numbside - 1.) / 2. and ypos[0] == (gdat.numbside - 1.) / 2. and xpos.size == 1:
        intg = int((gdat.numbside - 1.) / 2.)
        dictvarbderi['fraccent'] = cntpmodlpnts[intg, intg] / np.sum(cntpmodlpnts)

    return dictvarbderi


def retr_raticont(gdat, xpos, ypos, cnts, parapsfn):

    cntpmodltarg = retr_cntpmodl(gdat, 'fitt', xpos[0, None], ypos[0, None], cnts[0, None], 0., parapsfn)
    cntpmodlneig = retr_cntpmodl(gdat, 'fitt', xpos[1:], ypos[1:], cnts[1:], 0., parapsfn)
    raticont = np.sum(cntpmodltarg * cntpmodlneig) / np.sum(cntpmodltarg)**2
    
    return raticont


def retr_llik(para, gdat):
    
    # parse the parameter vector
    if gdat.typefittpsfnposi == 'fixd':
        xpos = gdat.fitt.catl['xpos']
        ypos = gdat.fitt.catl['ypos']
    if gdat.typefittpsfnposi == 'vari':
        xpos = para[:gdat.fitt.numbsour]
        ypos = para[gdat.fitt.numbsour:2*gdat.fitt.numbsour]
    
    cnts = para[gdat.indxparapsfnflux]
    cntpbackscal = para[gdat.indxparapsfnback]
    parapsfn = para[gdat.indxparapsfnpsfn]
    
    cntpmodl = retr_cntpmodl(gdat, 'fitt', xpos, ypos, cnts, cntpbackscal, parapsfn)
    
    chi2 = np.sum((gdat.cntpdatasexp - cntpmodl)**2 / gdat.cntpdatasexp)
    
    llik = -0.5 * chi2
    
    #print('xpos')
    #print(xpos)
    #print('ypos')
    #print(ypos)
    #print('cnts')
    #print(cnts)
    #print('cntpbackscal')
    #print(cntpbackscal)
    #print('parapsfn')
    #print(parapsfn)
    #print('gdat.cntpdatatmed')
    #print(gdat.cntpdatatmed)
    #print('cntpmodl')
    #print(cntpmodl)
    #print('')
    #print('')
    #print('')
    #print('')
    
    if gdat.booldiag and not np.isfinite(llik):
        print('')
        print('')
        print('Likelihood is Infinite!!')
        print('gdat.cntpdatasexp')
        summgene(gdat.cntpdatasexp)
        print('xpos')
        summgene(xpos)
        print('ypos')
        summgene(ypos)
        print('cnts')
        summgene(cnts)
        print('cntpbackscal')
        summgene(cntpbackscal)
        print('parapsfn')
        summgene(parapsfn)
        print('cntpmodl')
        summgene(cntpmodl)
        print('llik')
        print(llik)
        raise Exception('')

    return llik


#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def retr_mlikregr(cntpdataflat, matrdesi, variflat):
    '''
    Return the maximum likelihood estimate for linear regression.
    '''
    
    varitdim = np.diag(variflat)
    covafittcnts = np.linalg.inv(np.matmul(np.matmul(matrdesi.T, np.linalg.inv(varitdim)), matrdesi))
    mlikfittcnts = np.matmul(np.matmul(np.matmul(covafittcnts, matrdesi.T), np.linalg.inv(varitdim)), cntpdataflat)
    
    return mlikfittcnts, covafittcnts


def retr_strgsave(gdat, strgsecc, x, y, o):
    
    strgnumbside = '_n%03d' % gdat.numbside
    strgmaxmdmag = '_d%3.1f' % gdat.maxmdmag
    strgoffs = '_of%d%d' % (x, y)
    strgsave = '_%s_%s_%s%s%s%s' % (gdat.strgcnfg, strgsecc, gdat.typecade[o], strgnumbside, strgmaxmdmag, strgoffs)

    return strgsave


def setp_cntp(gdat, strg, typecntpscal):
    
    cntp = getattr(gdat, strg)
    
    vmin = np.nanpercentile(cntp, 0)
    vmax = np.nanpercentile(cntp, 100)
    if strg.startswith('resi'):
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    
    if typecntpscal == 'asnh':
        vmin = np.arcsinh(vmin)
        vmax = np.arcsinh(vmax)
    setattr(gdat, 'vmin' + strg + typecntpscal, vmin)
    setattr(gdat, 'vmax' + strg + typecntpscal, vmax)
    

def retr_strgsecc(gdat, o):
    
    strgsecc = '%02d%d%d' % (gdat.listtsec[o], gdat.listtcam[o], gdat.listtccd[o])

    return strgsecc


def init( \

         # data
         ## type of data: 'simugene', 'inje', 'obsd'
         typedata='obsd', \
         
         # selected TESS sectors
         listtsecsele=None, \
    
         # list of TESS sectors for which 
         #listtsec2min=None, \

         # data
         ## number of pixels on a side to cut out
         numbside=None, \
         ## Boolean flag to use Target Pixel Files (TPFs) at the highest cadence whenever possible
         booltpxflygo=True, \
         ## target
         ### a string to be used to search MAST for the target
         strgmast=None, \
         ### TIC ID of the target
         ticitarg=None, \
         ### TOI ID
         toiitarg=None, \
         ### RA of the target
         rasctarg=None, \
         ### DEC of the target
         decltarg=None, \
         
         # simulated data
         ## a dictionary describing the true generative model
         ### numbtime: number of time bins
         ### sourtype: 'dwar', \
         ### xpostarg: None, \
         ### ypostarg: None, \
         ### tmagtarg: None, \
         ### pmratarg: None, \
         ### pmdetarg: None, \
         ### pmxatarg: None, \
         ### pmyatarg: None, \
         dicttrue=None, \

         seedrand=0, \

         ## mask
         ### Boolean flag to put a cut on quality flag
         boolmaskqual=True, \
         ### limits of time between which the quality mask will be ignored
         limttimeignoqual=None, \
         ### ephemeris mask
         epocmask=None, \
         perimask=None, \
         duramask=None, \
        
         #Boolean flag to normalize the light curve by the median
         boolnorm=None, \

         # Boolean flag to merge fitting sources that are too close
         boolmerg=True, \

         # processing
         ## Boolean flag to turn on CBV detrending
         booldetrcbvs=False, \

         # string indicating the cluster of targets
         strgclus=None, \
         
         # reference time-series
         refrarrytser=None, \
         # list of labels for reference time-series
         refrlistlabltser=None, \

         # maximum delta magnitude of neighbor sources to be included in the model
         maxmdmag=4., \
        
         # model
         dictfitt=None, \
         ### RA of the target
         #fittrasctarg=None, \
         ### DEC of the target
         #fittdecltarg=None, \
         ## horizontal position of the target
         #fittxpostarg=None, \
         ## vertical position of the target
         #fittypostarg=None, \
         ## TESS magniude of the target
         #fitttmagtarg=None, \

         #fittxposneig=None, \
         #fittyposneig=None, \
         #fitttmagneig=None, \
        
         ### RA proper motion, used when RA and DEC are provided by the user
         #fittpmratarg=None, \
         ### DEC proper motion, used when RA and DEC are provided by the user
         #fittpmdetarg=None, \
         
         ## PSF
         ### type of inference
         #### 'fixd': fixed
         #### 'locl': based on only the image data to be analyzed
         #### 'osam': based on only the dithered image data collected during commissioning
         #### 'both': based on both
         typepsfninfe='locl', \
         
         ### type of sub-pixel interpolation of the point source emission model
         #### 'eval': evaluation of a functional form on the data grid (not to be used when point source positions are floating)
         #### 'regrcubi': cubic regression
         #### 'regrline': linear regression
         #### 'regrcubipixl': separate cubic regression in each pixel
         typepsfnsubp='eval', \
         
         catlextr=None, \
         lablcatlextr=None, \
            
         # Boolean flag to repeat the fit, putting the target to offset locations
         boolfittoffs=False, \
         
         # Boolean flag to use fast cadence (20 sec) when using target pixel files
         boolfasttpxf=False, \

         ## post-process
         ### Boolean flag to perform aperture photometry
         boolphotaper=True, \
         ### list of pixels for the aperture
         listpixlapertarg=None, \
         ### Boolean flag to output aperture photometry
         booloutpaper=False, \

         # visualization
         ## Boolean flag to make any plots
         boolplot=True, \
         ## Boolean flag to make relative flux plots
         boolplotrflx=None, \
         ## Boolean flag to make image plots
         boolplotcntp=None, \
         ## Boolean flag to plot the centroiod
         boolplotcent=None, \
         ## Boolean flag to plot the quaternions
         boolplotquat=None, \
         ## Boolean flag to make an animation
         boolanim=None, \
         ## Boolean flag to include all time bins in the animation
         boolanimframtotl=None, \
        
         ## Boolean flag to plot the histogram of the number of counts
         boolplothhistcntp=None, \

         ## time offset for time-series plots
         timeoffs=2457000., \

         ## list of limits for temporal zoom
         listlimttimetzom=None, \

         ## the time to indicate on the plots with a vertical line
         listtimeplotline=None, \
         
         # path for the target
         pathtarg=None, \
        
         # a string that will be used to name output files for this target
         strgtarg=None, \
         
         # a string that will appear in the plots to label the target, which can be anything the user wants
         labltarg=None, \
         
         # image color scale
         ## 'self': image is linear
         ## 'asnh': arcsinh of the image is linear
         listtypecntpscal=['asnh'], \
        
         # plot extensions
         typefileplot='pdf', \
        
         # verbosity level
         typeverb=1, \
        
         # Boolean flag to turn on diagnostic mode
         booldiag=True, \
         
        ):
   
    # start the timer
    timeinittotl = timemodu.time()
    
    # construct global object
    gdat = tdpy.util.gdatstrt()
    
    # copy unnamed inputs to the global object
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)
    
    if gdat.boolnorm is None:
        gdat.boolnorm = True
    
    if gdat.boolnorm:
        gdat.lablyaxilcur = 'Relative flux'
    else:
        gdat.lablyaxilcur = 'ADC counts [e$^-$/s]'
    
    # copy named arguments to the global object
    #for strg, valu in args.items():
    #    setattr(gdat, strg, valu)
    
    # set the seed
    np.random.seed(seedrand)
    
    if not gdat.typedata in ['simugene', 'inje', 'obsd']:
        raise Exception('')

    if gdat.boolplot:
        
        if gdat.boolplotrflx is None:
            gdat.boolplotrflx = True
        
        if gdat.boolplotcntp is None:
            gdat.boolplotcntp = True
         
        if gdat.boolplotquat is None:
            gdat.boolplotquat = True
         
        if gdat.boolanim is None:
            gdat.boolanim = True
         
        if gdat.boolanimframtotl is None:
            gdat.boolanimframtotl = False
        
        if gdat.boolplothhistcntp is None:
            gdat.boolplothhistcntp = True

    if gdat.numbside is None:
        gdat.boolinptnumbside = False
        gdat.numbside = 11
    else:
        gdat.boolinptnumbside = True
        
    if dicttrue is None:
        dicttrue = dict()
    
    if gdat.typedata == 'inje':
        if 'rasctarg' in dicttrue or 'decltarg' in dicttrue:
            raise Exception('When a sim. source is injected, only provide target RA (rasctarg) and dec (decltarg) and exclude them from the generative model dictionary (dicttrue).')
    
        dicttrue['rasctarg'] = gdat.rasctarg
        dicttrue['decltarg'] = gdat.rasctarg

    if not 'velrtarg' in dicttrue:  
        dicttrue['velrtarg'] = 0.
    if not 'veldtarg' in dicttrue:  
        dicttrue['veldtarg'] = 0.
    if not 'sigmpsfn' in dicttrue:  
        dicttrue['sigmpsfn'] = 1.05
    if not 'sigmpsfnxpos' in dicttrue:  
        dicttrue['sigmpsfnxpos'] = 1.05
    if not 'sigmpsfnypos' in dicttrue:  
        dicttrue['sigmpsfnypos'] = 1.05
    if not 'fracskewpsfnxpos' in dicttrue:  
        dicttrue['fracskewpsfnxpos'] = 0.
    if not 'fracskewpsfnypos' in dicttrue:  
        dicttrue['fracskewpsfnypos'] = 0.
    if not 'cntpbackscal' in dicttrue:  
        dicttrue['cntpbackscal'] = 100.
    if not 'xpostarg' in dicttrue:  
        dicttrue['xpostarg'] = (gdat.numbside - 1.) / 2.
    if not 'ypostarg' in dicttrue:  
        dicttrue['ypostarg'] = (gdat.numbside - 1.) / 2.
    if not 'tmagtarg' in dicttrue:  
        dicttrue['tmagtarg'] = 10.
    if not 'xposneig' in dicttrue:  
        dicttrue['xposneig'] = np.array([])
    if not 'yposneig' in dicttrue:  
        dicttrue['yposneig'] = np.array([])
    if not 'tmagneig' in dicttrue:  
        dicttrue['tmagneig'] = np.array([])
    
    ### type of template PSF model shape
    #### 'osam': based on the dithered image data collected during commissioning
    if not 'typepsfnshap' in dicttrue:
        dicttrue['typepsfnshap'] = 'gauscirc'
    
    gdat.true = tdpy.util.gdatstrt()
    for name, valu in dicttrue.items():
        setattr(gdat.true, name, valu)

    if dictfitt is None:
        dictfitt = dict()
    
    for name in ['typepsfnshap', 'sigmpsfn', 'sigmpsfnxpos', 'sigmpsfnypos', 'fracskewpsfnxpos', 'fracskewpsfnypos']:
        if not name in dictfitt:
            dictfitt[name] = dicttrue[name]
    
    gdat.fitt = tdpy.util.gdatstrt()
    for name, valu in dictfitt.items():
        setattr(gdat.fitt, name, valu)
    
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

    print('gdat.boolnorm')
    print(gdat.boolnorm)
    
    # check input
    ## ensure target identifiers are not conflicting
    if gdat.typedata == 'inje' or gdat.typedata == 'obsd':
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
    
    if gdat.typedata == 'simugene' and gdat.true.tmagtarg is None:
        print('gdat.typedata')
        print(gdat.typedata)
        print('gdat.true.tmagtarg')
        print(gdat.true.tmagtarg)
        raise Exception('truetmagtarg needs to be set when simulated data based on a generative model, are generated.')

    gdat.pathtoii = os.environ['EPHESUS_DATA_PATH'] + '/data/exofop_tess_tois.csv'
    print('Reading from %s...' % gdat.pathtoii)
    objtexof = pd.read_csv(gdat.pathtoii, skiprows=0)

    # conversion factors
    gdat.dictfact = ephesus.retr_factconv()
    
    # determine the MAST keyword and TOI ID of the target and its type
    if gdat.ticitarg is not None:
        gdat.typetarg = 'tici'
        print('A TIC ID was provided as target identifier.')
        indx = np.where(objtexof['TIC ID'].values == gdat.ticitarg)[0]
        if indx.size > 0:
            gdat.toiitarg = int(str(objtexof['TOI'][indx[0]]).split('.')[0])
            print('Matched the input TIC ID with TOI %d.' % gdat.toiitarg)
        gdat.strgmast = 'TIC %d' % gdat.ticitarg
    elif gdat.toiitarg is not None:
        gdat.typetarg = 'toii'
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
        gdat.typetarg = 'mast'
        print('A MAST key (%s) was provided as target identifier.' % gdat.strgmast)
    elif gdat.rasctarg is not None and gdat.decltarg is not None:
        gdat.typetarg = 'posi'
        print('RA and DEC (%g %g) are provided as target identifier.' % (gdat.rasctarg, gdat.decltarg))
        gdat.strgmast = '%g %g' % (gdat.rasctarg, gdat.decltarg)
    elif gdat.true.xpostarg is not None and gdat.true.ypostarg is not None:
        gdat.typetarg = 'posi'
        print('Horizontal and vertical positions (%g %g) are provided as target identifier for a simulated target.' % (gdat.true.xpostarg, gdat.true.ypostarg))
    else:
        raise Exception('')

    if gdat.typedata == 'inje' or gdat.typedata == 'obsd':
        print('gdat.typetarg')
        print(gdat.typetarg)
        print('gdat.strgmast')
        print(gdat.strgmast)
        print('gdat.toiitarg')
        print(gdat.toiitarg)
    
    if gdat.typedata != 'simugene':
        # temp -- check that the closest TIC to a given TIC is itself
        maxmradi = 1.1 * 20. * 1.4 * (gdat.numbside + 2) / 2.
        print('Querying the TIC within %d as to get the RA, DEC, Tmag, and TIC ID of the closest source to the MAST keywrod %s...' % (maxmradi, gdat.strgmast))
        catalogData = astroquery.mast.Catalogs.query_region(gdat.strgmast, radius='%ds' % maxmradi, catalog="TIC")
        print('Found %d TIC sources within 200 as.' % len(catalogData))
    
    gdat.refr = tdpy.util.gdatstrt()
    
    # get define reference catalogs
    gdat.refr.lablcatl = ['TIC']
    if gdat.catlextr is not None:
        gdat.refr.lablcatl.extend(gdat.lablcatlextr)

    gdat.refr.numbcatl = len(gdat.refr.lablcatl)
    gdat.refr.indxcatl = np.arange(gdat.refr.numbcatl)
    
    print('gdat.refr.numbcatl')  
    print(gdat.refr.numbcatl)

    gdat.liststrgfeat = ['tmag', 'labl', 'xpos', 'ypos']
    if gdat.typedata != 'simugene':
        gdat.liststrgfeat += ['tici', 'rasc', 'decl', 'pmde', 'pmra']
    
    gdat.refr.liststrgfeat = [[] for q in gdat.refr.indxcatl]
    gdat.refr.liststrgfeat[0] = gdat.liststrgfeat
    gdat.refr.catlbase = [dict() for q in gdat.refr.indxcatl]
    gdat.refr.numbsourbase = np.empty(gdat.refr.numbcatl, dtype=int)
    gdat.refr.indxsourbase = [[] for q in np.arange(gdat.refr.numbcatl)]
    
    
    for q in gdat.refr.indxcatl:
        print('Constructing reference catalog %d...' % q)
        if gdat.typedata == 'inje':
            gdat.refr.numbsourbase[q] = 1 + catalogData[:]['Tmag'].size
        if gdat.typedata == 'obsd':
            gdat.refr.numbsourbase[q] = catalogData[:]['Tmag'].size
        
        if gdat.typedata == 'inje':
            #gdat.true.rasctarg = (gdat.numbside - 1.) / 2.
            #gdat.true.decltarg = (gdat.numbside - 1.) / 2.
            gdat.true.velxtarg = 0.
            gdat.true.velytarg = 0.
            
        if gdat.typedata == 'simugene':
            gdat.true.velxtarg = 0.
            gdat.true.velytarg = 0.
            
        if gdat.typedata == 'simugene' or gdat.typedata == 'inje':
            if gdat.true.tmagneig.size > 0:
                # generate neighbors within 0.5 pixels of the edges
                gdat.true.velxneig = np.zeros(gdat.true.tmagneig.size)
                gdat.true.velyneig = np.zeros(gdat.true.tmagneig.size)
                gdat.refr.catlbase[q]['tmag'] = np.concatenate((np.array([gdat.true.tmagtarg]), gdat.true.tmagneig))
                if gdat.typedata == 'simugene':
                    gdat.refr.catlbase[q]['xpos'] = np.concatenate((np.array([gdat.true.xpostarg]), gdat.true.xposneig))
                    gdat.refr.catlbase[q]['ypos'] = np.concatenate((np.array([gdat.true.ypostarg]), gdat.true.yposneig))
                    gdat.refr.catlbase[q]['velx'] = np.concatenate((np.array([gdat.true.velxtarg]), gdat.true.velxneig))
                    gdat.refr.catlbase[q]['vely'] = np.concatenate((np.array([gdat.true.velytarg]), gdat.true.velyneig))
                if gdat.typedata == 'inje':
                    gdat.refr.catlbase[q]['rasc'] = np.concatenate((np.array([gdat.true.rasctarg]), catalogData[:]['ra']))
                    gdat.refr.catlbase[q]['decl'] = np.concatenate((np.array([gdat.true.decltarg]), catalogData[:]['dec']))
                    gdat.refr.catlbase[q]['velr'] = np.concatenate((np.array([gdat.true.velrtarg]), 0. * catalogData[:]['ra']))
                    gdat.refr.catlbase[q]['veld'] = np.concatenate((np.array([gdat.true.veldtarg]), 0. * catalogData[:]['dec']))
            else:
                if gdat.typedata == 'simugene':
                    gdat.refr.catlbase[q]['xpos'] = np.array([gdat.true.xpostarg])
                    gdat.refr.catlbase[q]['ypos'] = np.array([gdat.true.ypostarg])
                if gdat.typedata == 'inje':
                    gdat.refr.catlbase[q]['rasc'] = np.array([gdat.true.rasctarg])
                    gdat.refr.catlbase[q]['decl'] = np.array([gdat.true.decltarg])
                gdat.refr.catlbase[q]['tmag'] = np.array([gdat.true.tmagtarg])
                gdat.refr.catlbase[q]['velx'] = np.array([gdat.true.velxtarg])
                gdat.refr.catlbase[q]['vely'] = np.array([gdat.true.velytarg])
            
            # labels
            gdat.refr.catlbase[q]['labl'] = np.empty(gdat.refr.catlbase[q]['xpos'].size, dtype=object)
            for k in np.arange(gdat.refr.catlbase[q]['labl'].size):
                gdat.refr.catlbase[q]['labl'][k] = '%d' % k
            gdat.refr.numbsourbase[q] = gdat.refr.catlbase[q]['xpos'].size    
        
        gdat.refr.indxsourbase[q] = np.arange(gdat.refr.numbsourbase[q])
            
        if gdat.typedata == 'inje' or gdat.typedata == 'obsd':
            
            if gdat.typedata == 'inje':
                offs = 1
            if gdat.typedata == 'obsd':
                offs = 0
            
            for name in ['tmag', 'rasc', 'decl', 'tici', 'pmra', 'pmde', 'rascorig', 'declorig']:
                gdat.refr.catlbase[q][name] = np.empty(gdat.refr.numbsourbase[q])
            gdat.refr.catlbase[q]['tici'] = np.empty(gdat.refr.numbsourbase[q], dtype=int)
            gdat.refr.catlbase[q]['labl'] = np.empty(gdat.refr.numbsourbase[q], dtype=object)

            gdat.refr.catlbase[q]['tmag'][offs:] = catalogData[:]['Tmag']
            gdat.refr.catlbase[q]['rasc'][offs:] = catalogData[:]['ra']
            gdat.refr.catlbase[q]['decl'][offs:] = catalogData[:]['dec']
            gdat.refr.catlbase[q]['tici'][offs:] = catalogData[:]['ID']
            gdat.refr.catlbase[q]['pmde'][offs:] = catalogData[:]['pmDEC']
            gdat.refr.catlbase[q]['pmra'][offs:] = catalogData[:]['pmRA']
            gdat.refr.catlbase[q]['rascorig'][offs:] = catalogData[:]['RA_orig']
            gdat.refr.catlbase[q]['declorig'][offs:] = catalogData[:]['Dec_orig']
            for k in np.arange(catalogData[:]['ID'].size):
                gdat.refr.catlbase[q]['labl'][offs+k] = 'TIC %s' % catalogData[:]['ID'][k]
        gdat.refr.catlbase[q]['cnts'] = retr_fluxfromtmag(gdat.refr.catlbase[q]['tmag'])
    
    if gdat.typedata != 'simugene':
        if gdat.typetarg != 'posi':
            # ensure that the first source is the target
            print('catalogData[0][dstArcSec]')
            print(catalogData[0]['dstArcSec'])
            print('temp: webpage and astroquery dstArcSec values for TIC on MAST differ!')
            if catalogData[0]['dstArcSec'] < 0.2:
                gdat.ticitarg = int(catalogData[0]['ID'])
                gdat.rasctarg = catalogData[0]['ra']
                gdat.decltarg = catalogData[0]['dec']
                gdat.tmagtarg = catalogData[0]['Tmag']
        
        if gdat.rasctarg is not None:
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
            if gdat.typetarg != 'posi':
                print('gdat.tmagtarg')
                print(gdat.tmagtarg)
    
    if gdat.labltarg is None:
        if gdat.typedata == 'inje' or gdat.typedata == 'obsd':
            if gdat.typetarg == 'mast':
                gdat.labltarg = gdat.strgmast
            if gdat.typetarg == 'toii':
                gdat.labltarg = 'TOI %d' % gdat.toiitarg
            if gdat.typetarg == 'tici':
                gdat.labltarg = 'TIC %d' % gdat.ticitarg
            if gdat.typetarg == 'posi':
                gdat.labltarg = 'RA=%.4g, DEC=%.4g' % (gdat.rasctarg, gdat.decltarg)
        else:
            raise Exception('A label must be provided for targets when data are simulated based on a generative model.')
    if gdat.strgtarg is None:
        gdat.strgtarg = ''.join(gdat.labltarg.split(' '))
    
    print('Target label: %s' % gdat.labltarg) 
    print('Output folder name: %s' % gdat.strgtarg) 
    if gdat.typedata != 'simugene':
        print('RA and DEC: %g %g' % (gdat.rasctarg, gdat.decltarg))
        if gdat.typetarg == 'tici' or gdat.typetarg == 'mast':
            print('Tmag: %g' % gdat.tmagtarg)
   
    print('PSF inference type: %s' % gdat.typepsfninfe)
    print('PSF model shape type: %s' % gdat.fitt.typepsfnshap)
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
    gdat.pathimagtargsexp = gdat.pathimagtarg + 'sexp/'
    os.system('mkdir -p %s' % gdat.pathimagtargsexp)
   
    # header that will be added to the output CSV files
    gdat.strgheadtarg = 'time [BJD], relative flux, relative flux error'
    
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
    if gdat.typedata == 'simugene':
        gdat.numbtsec = 1
        gdat.indxtsec = np.arange(gdat.numbtsec)
        gdat.listtsec = [0]
        gdat.listtcam = [0]
        gdat.listtccd = [0]
    
    if gdat.typedata != 'simugene':
        
        # check all available TESS (FFI) cutout data 
        print('Calling TESSCut to get the data...')
        listhdundatatemp = astroquery.mast.Tesscut.get_cutouts('%g %g' % (gdat.rasctarg, gdat.decltarg), gdat.numbside)
        
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
        gdat.listtcamffim = np.array(gdat.listtcamffim)
        gdat.listtccdffim = np.array(gdat.listtccdffim)
        gdat.listhdundataffim = np.array(gdat.listhdundataffim)
        
        indxsort = np.argsort(gdat.listtsecffim)
        
        gdat.listtsecffim = gdat.listtsecffim[indxsort]
        gdat.listtcamffim = gdat.listtcamffim[indxsort]
        gdat.listtccdffim = gdat.listtccdffim[indxsort]
        gdat.listhdundataffim = gdat.listhdundataffim[indxsort]

        print('gdat.listtsecffim')
        print(gdat.listtsecffim)
        print('gdat.listtcamffim')
        print(gdat.listtcamffim)
        print('gdat.listtccdffim')
        print(gdat.listtccdffim)
        
        gdat.listtsecspoc = []
        gdat.listtcamspoc = []
        gdat.listtccdspoc = []
        print('booltpxflygo')
        print(booltpxflygo)
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
                    boolgood = boolfasttemp == gdat.boolfasttpxf

                    if boolgood and listprodspoctemp[a]['description'] == 'Target pixel files':
                        tsec = int(listprodspoctemp[a]['obs_id'].split('-')[1][1:])
                        gdat.listtsecspoc.append(tsec) 
                        
                        print('temp: assigning dummy Cam and CCD to the target for this sector.')
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
            
            gdat.numbtsecspoc = gdat.listtsecspoc.size

        if len(gdat.listtsecspoc) > 0:
            gdat.indxtsecspoc = np.arange(gdat.numbtsecspoc)
            
            # download data from MAST
            os.system('mkdir -p %s' % gdat.pathdatatarg)
        
            print('Downloading SPOC data products...')
            
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
            gdat.listhdundataspoc = [[] for o in gdat.indxtsecspoc]
            gdat.indxtimegoodspoc = [[] for o in gdat.indxtsecspoc]
            for oo in gdat.indxtsecspoc:
                gdat.listhdundataspoc[oo], gdat.indxtimegoodspoc[oo], gdat.listtsecspoc[oo], gdat.listtcamspoc[oo], \
                                                                gdat.listtccdspoc[oo] = ephesus.read_tesskplr_file(listpathdownspoctpxf[oo])
                
                if not np.isfinite(gdat.listhdundataspoc[oo][1].data['TIME'][gdat.indxtimegoodspoc[oo]]).all():
                    raise Exception('')
            
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
        
    gdat.listnameanls = ['psfn']
    if gdat.boolphotaper:
        gdat.listnameanls.append('aper')
        gdat.indxanlsaper = 1
    
    print('temp')
    gdat.listnameanls = ['aper']
    gdat.indxanlsaper = 0

    gdat.numbanls = len(gdat.listnameanls)
    gdat.indxanls = np.arange(gdat.numbanls)

    gdat.fitt.arryrflx = [[[] for e in gdat.indxanls] for o in gdat.indxtsec]
    gdat.refr.numbsour = np.empty((gdat.refr.numbcatl, gdat.numbtsec), dtype=int)
    gdat.typecade = np.zeros(gdat.numbtsec, dtype=object)
    if gdat.typedata != 'simugene':
        # determine whether sectors have TPFs
        gdat.booltpxf = ephesus.retr_booltpxf(gdat.listtsec, gdat.listtsecspoc)
        print('gdat.booltpxf')
        print(gdat.booltpxf)

        print('WARNING! SUPPRESSING NUMBSIDE ASSERTION.')
        if False and gdat.booldiag:
            for o in gdat.indxtsec:
                if gdat.booltpxf[o]:
                    if gdat.boolinptnumbside:
                        raise Exception('')

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
            strgtsec = 'sc%02d' % gdat.listtsec[o]
            for e, nameanls in enumerate(gdat.listnameanls):
                pathsaverflxtarg = gdat.pathdatatarg + 'rflx_%s%s.csv' % (nameanls, strgsave)
                gdat.dictoutp['pathsaverflxtarg_%s_%s' % (nameanls, strgtsec)] = pathsaverflxtarg
            
                if os.path.exists(pathsaverflxtarg):
                    print('Analysis of Sector %d previously completed...' % gdat.listtsec[o])
                    print('Reading from %s...' % pathsaverflxtarg)
                    gdat.fitt.arryrflx[o][e] = np.loadtxt(pathsaverflxtarg, delimiter=',', skiprows=1)
        
    if gdat.typedata == 'simugene':
        for o, tsec in enumerate(gdat.listtsec):
            gdat.typecade[o] = '10mn'
    
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
        
    gdat.sizeplotsour = 30
        
    gdat.listtypecatlplot = ['fitt', 'refr', 'refrfitt']
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
        
        if gdat.booldiag:
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

    # exposure time
    gdat.timeexpo = np.empty(gdat.numbtsec)
    for o in gdat.indxtsec:
        if gdat.typecade[o] == '20sc':
            gdat.timeexpo[o] = 20. # [sec]
        if gdat.typecade[o] == '2min':
            gdat.timeexpo[o] = 120. # [sec]
        if gdat.typecade[o] == '10mn':
            gdat.timeexpo[o] = 1600. # [sec]
        if gdat.typecade[o] == '30mn':
            gdat.timeexpo[o] = 1800. # [sec]
    # account for overhead
    gdat.timeexpo *= 0.8

    gdat.refr.catl = [[dict() for o in gdat.indxtsec] for q in gdat.refr.indxcatl]
    gdat.refr.cequ = [[[] for o in gdat.indxtsec] for q in gdat.refr.indxcatl]
    gdat.indxsourbrgt = [[[] for o in gdat.indxtsec] for q in gdat.refr.indxcatl]
    for q in gdat.refr.indxcatl:
        for o in gdat.indxtsec:
            print('Sector %d' % gdat.listtsec[o])
            for strgfeat in gdat.refr.catlbase[q].keys():#gdat.refr.liststrgfeat[q]:
                gdat.refr.catl[q][o][strgfeat] = np.array(gdat.refr.catlbase[q][strgfeat])
            print('Number of sources in the reference catalog %d: %d' % (q, len(gdat.refr.catl[q][o]['tmag'])))
            
            if gdat.typetarg != 'posi':
                dmag = gdat.refr.catl[q][o]['tmag'] - gdat.refr.catl[q][o]['tmag'][0]
                gdat.indxsourbrgt[q][o] = np.where(dmag < gdat.maxmdmag)[0]
                gdat.numbrefrbrgt = gdat.indxsourbrgt[q][o].size
                magtcutt = gdat.refr.catl[q][o]['tmag'][0] + gdat.maxmdmag
                print('%d of the reference catalog sources are brighter than the magnitude cutoff of %g.' % (gdat.numbrefrbrgt, magtcutt))
                
            gdat.refr.numbsour[q, o] = gdat.refr.catl[q][o]['cnts'].size

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
            if gdat.typedata != 'simugene' and gdat.booltpxf[o]:
                gdat.refr.labltser[o] += ['SPOC']
    
            # number of reference light curves
            gdat.refr.numbtser[o] = len(gdat.refr.labltser[o])
            gdat.refr.indxtser[o] = np.arange(gdat.refr.numbtser[o])
    
            gdat.refr.colrlcur[o] = ['r', 'orange'][:gdat.refr.numbtser[o]]
        
    gdat.refrtime = [[[] for k in gdat.refr.indxtser[o]] for o in gdat.indxtsec]
    gdat.refrrflx = [[[] for k in gdat.refr.indxtser[o]] for o in gdat.indxtsec]
    gdat.stdvrefrrflx = [[[] for k in gdat.refr.indxtser[o]] for o in gdat.indxtsec]
    
    # temporal types of image plots
    ## medians
    listtypeplotcntp = []
    if gdat.boolplotcntp:
        listtypeplotcntp += ['tmed']
    # cadence frames
    if gdat.boolanim:
        listtypeplotcntp += ['anim']
        
    gdat.liststrgfeat += ['xpos', 'ypos']
    # write metadata to file
    gdat.pathsavemetaglob = gdat.pathdatatarg + 'metatarg.csv'
    dicttranlabl = dict()
    dicttranlabl['rasc'] = 'RA'
    dicttranlabl['decl'] = 'Dec'
    
    dictmeta = dict()
    print('Writing to %s...' % gdat.pathsavemetaglob)
    objtfile = open(gdat.pathsavemetaglob, 'w')
    for key, value in dictmeta.items():
        objtfile.write('%s,%g\n' % (key, value))
    objtfile.close()

    # Boolean flag to indicate whether there is a reference time-series
    gdat.boolrefrtser = [[] for o in gdat.indxtsec]
    if gdat.refrarrytser is None:
        if gdat.typedata == 'simugene':
            for o in gdat.indxtsec:
                gdat.boolrefrtser[o] = False
        if gdat.typedata != 'simugene':
            cntr = 0
            for o in gdat.indxtsec:
                # get reference light curve
                if gdat.booltpxf[o]:
                    arry, tsecrefr, tcam, tccd = ephesus.read_tesskplr_file(listpathdownspoclcur[cntr], strgtypelcur='PDCSAP_FLUX')
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

    gdat.refr.indxsour = [[[] for o in gdat.indxtsec] for q in np.arange(gdat.refr.numbcatl)]
    
    gdat.listtime = [[] for o in gdat.indxtsec]
    gdat.indxtime = [[] for o in gdat.indxtsec]
    
    gdat.numbtime = np.empty(gdat.numbtsec, dtype=int)
    
    arrytemp = np.linspace(0., float(gdat.numbside - 1), gdat.numbside)
    gdat.xposimag, gdat.yposimag = np.meshgrid(arrytemp, arrytemp)
    
    if gdat.fitt.typepsfnshap == 'empi':
        gdat.intgpsfnfitt = 6
        gdat.numbsidepsfnfitt = gdat.numbside - 2 * gdat.intgpsfnfitt
        gdat.numbparapsfnempi = gdat.numbsidepsfnfitt**2
        gdat.indxparapsfnempi = np.arange(gdat.numbparapsfnempi)
        gdat.xposimagpsfn = gdat.xposimag[gdat.intgpsfnfitt:-gdat.intgpsfnfitt, gdat.intgpsfnfitt:-gdat.intgpsfnfitt]
        gdat.yposimagpsfn = gdat.yposimag[gdat.intgpsfnfitt:-gdat.intgpsfnfitt, gdat.intgpsfnfitt:-gdat.intgpsfnfitt]
    
    gdat.indxsourwthn = [[[] for o in gdat.indxtsec] for q in gdat.refr.indxcatl]
    gdat.indxsourwthnbrgt = [[[] for o in gdat.indxtsec] for q in gdat.refr.indxcatl]
    # get the WCS object for each sector
    gdat.listhdundata = [[] for o in gdat.indxtsec]
    gdat.listobjtwcss = [[] for o in gdat.indxtsec]
    
    gdat.listnois = [[[] for e in gdat.indxanls] for o in gdat.indxtsec]
    for o in gdat.indxtsec:
        strgsecc = retr_strgsecc(gdat, o)
        strgsave = retr_strgsave(gdat, strgsecc, 1, 1, o)
        
        if gdat.typedata != 'simugene':
            print('Sector: %d' % gdat.listtsec[o])
            print('Camera: %d' % gdat.listtcam[o])
            print('CCD: %d' % gdat.listtccd[o])
        
        if gdat.typedata != 'simugene':
            if gdat.booltpxf[o]:
                print('TPF data')
            else:
                print('FFI data')
        
        if gdat.boolplotcntp or gdat.boolplotrflx or gdat.boolanim:
            if gdat.typedata == 'simugene':
                gdat.strgtitlcntpplot = '%s' % (gdat.labltarg)
            else:
                gdat.strgtitlcntpplot = '%s, Sector %d, Cam %d, CCD %d' % (gdat.labltarg, gdat.listtsec[o], gdat.listtcam[o], gdat.listtccd[o])
        
        if gdat.typedata == 'simugene':
            if gdat.typecade[o] == '2min':
                difftime = 2. / 60. / 24. # [days]
            elif gdat.typecade[o] == '10mn':
                difftime = 10. / 60. / 24. # [days]
            else:
                raise Exception('')
            gdat.listtime[o] = 2457000. + np.arange(0., 30., difftime)
        
        if gdat.typedata == 'simugene' or gdat.typedata == 'inje':
            if gdat.true.typepsfnshap == 'gauscirc':
                gdat.true.parapsfn = np.array([gdat.true.sigmpsfn])
            if gdat.true.typepsfnshap == 'gauselli':
                gdat.true.parapsfn = np.array([gdat.true.sigmpsfnxpos, gdat.true.sigmpsfnypos, gdat.true.fracskewpsfnxpos, gdat.true.fracskewpsfnypos])
            
        if gdat.typedata == 'simugene':
            
            # generate data
            gdat.numbtime[o] = gdat.listtime[o].size
            gdat.cntpmodlsimu = np.empty((gdat.numbside, gdat.numbside, gdat.numbtime[o]))
        
            for t in np.arange(gdat.numbtime[o]):
                gdat.cntpmodlsimu[:, :, t] = retr_cntpmodl(gdat, 'true', gdat.refr.catl[q][o]['xpos'], gdat.refr.catl[q][o]['ypos'], gdat.refr.catl[q][o]['cnts'], \
                                                                                                                                gdat.true.cntpbackscal, gdat.true.parapsfn)
        
            gdat.cntpdata = np.random.poisson(gdat.cntpmodlsimu).astype(float)
            gdat.cntpdata += np.random.normal(size=gdat.cntpdata.size).reshape(gdat.cntpdata.shape) * gdat.cntpdata * 5e-3
        
        if gdat.typedata != 'simugene':
            
            # get data
            ## read the FITS files
            #print(gdat.listhdundata[o][1].data.names)
            ## time
            if gdat.booltpxf[o]:
                indxtsecspoctemp = np.where(gdat.listtsec[o] == gdat.listtsecspoc)[0][0]
                gdat.listhdundata[o] = gdat.listhdundataspoc[indxtsecspoctemp]
            else:
                indxtsectemp = np.where(gdat.listtsec[o] == gdat.listtsecffim)[0][0]
                gdat.listhdundata[o] = gdat.listhdundataffim[indxtsectemp]
            
            # times
            gdat.listtime[o] = gdat.listhdundata[o][1].data['TIME'] + 2457000
            
            ## count per pixel
            gdat.cntpdata = (gdat.listhdundata[o][1].data['FLUX'] + \
                                                        gdat.listhdundata[o][1].data['FLUX_BKG']).swapaxes(0, 2).swapaxes(0, 1)
            
            if gdat.booltpxf[o]:
                indxtsecspoctemp = np.where(gdat.listtsec[o] == gdat.listtsecspoc)[0][0]
                
                gdat.listtime[o] = gdat.listtime[o][gdat.indxtimegoodspoc[indxtsecspoctemp]]
                
                if gdat.booldiag:
                    if not np.isfinite(gdat.listtime[o]).all():
                        raise Exception('')
                
                gdat.cntpdata = gdat.cntpdata[:, :, gdat.indxtimegoodspoc[indxtsecspoctemp]]

                if gdat.cntpdata.shape[0] != 11:
                    raise Exception('')
                
                if gdat.numbside < 11:
                    intg = int((11 - gdat.numbside) / 2)
                    gdat.cntpdata = gdat.cntpdata[intg:11-intg, intg:11-intg, :]
                elif gdat.numbside > 11:
                    raise Exception('')

            print('Number of raw data points: %d' % gdat.listtime[o].size)
            
            if gdat.booltpxf[o]:
                gdat.numbside = gdat.cntpdata.shape[1]
            
            #booldatagood = np.isfinite(gdat.listtime[o])
            booldatagood = np.any(gdat.cntpdata > 0, axis=(0, 1))
            if gdat.boolmaskqual:
                print('Masking bad data with quality flags...')
                if limttimeignoqual is not None:
                    print('Ignoring the quality mask between %g and %g...' % (limttimeignoqual[0], limttimeignoqual[1]))
                    booldatagood = booldatagood & ((limttimeignoqual[0] < gdat.listtime[o]) & (gdat.listtime[o] < limttimeignoqual[1]))
            print('Masking out bad data...')
            indxtimedatagood = np.where(booldatagood)[0]
            fracgood = 100. * float(indxtimedatagood.size) / gdat.listtime[o].size
            print('Fraction of unmasked (good) times: %.4g percent' % fracgood)
            if indxtimedatagood.size == 0:
                print('No good data found for this sector. The returned list will have an empty element.')
                continue
    
            # keep good times and discard others
            gdat.listtime[o] = gdat.listtime[o][indxtimedatagood]
            gdat.cntpdata = gdat.cntpdata[:, :, indxtimedatagood]
        
        gdat.indxside = np.arange(gdat.numbside)

        gdat.numbtime[o] = gdat.listtime[o].size
        gdat.indxtime[o] = np.arange(gdat.numbtime[o])
        
        if gdat.typedata != 'simugene':
            gdat.listobjtwcss[o] = astropy.wcs.WCS(gdat.listhdundata[o][2].header)
        
        ## reference catalogs
        for q in gdat.refr.indxcatl:
            
            if gdat.typedata != 'simugene':
                gdat.refr.cequ[q][o] = np.empty((gdat.refr.catl[q][o]['rasc'].size, 2))
                gdat.refr.cequ[q][o][:, 0] = gdat.refr.catl[q][o]['rasc']
                gdat.refr.cequ[q][o][:, 1] = gdat.refr.catl[q][o]['decl']
                gdat.refr.cpix = gdat.listobjtwcss[o].all_world2pix(gdat.refr.cequ[q][o], 0)
                gdat.refr.catl[q][o]['xpos'] = gdat.refr.cpix[:, 0]
                gdat.refr.catl[q][o]['ypos'] = gdat.refr.cpix[:, 1]
                if gdat.booltpxf[o] and gdat.numbside < 11:
                    gdat.refr.catl[q][o]['xpos'] -= intg
                    gdat.refr.catl[q][o]['ypos'] -= intg
            
            gdat.refr.catlbase[q]['xpostime'] = np.zeros((gdat.numbtime[o], gdat.refr.numbsourbase[q])) + gdat.refr.catl[q][o]['xpos'][None, :]
            gdat.refr.catlbase[q]['ypostime'] = np.zeros((gdat.numbtime[o], gdat.refr.numbsourbase[q])) + gdat.refr.catl[q][o]['ypos'][None, :]
                
            gdat.refr.numbsour[q, o] = gdat.refr.catl[q][o]['xpos'].size
            gdat.refr.indxsour[q][o] = np.arange(gdat.refr.numbsour[q, o])
            print('Number of reference sources is %d.' % gdat.refr.numbsour[q, o])
            
            ## indices of the reference catalog sources within the cutout
            gdat.indxsourwthn[q][o] = np.where((gdat.refr.catl[q][o]['xpos'] > -0.5) & (gdat.refr.catl[q][o]['xpos'] < gdat.numbside - 0.5) & \
                                            (gdat.refr.catl[q][o]['ypos'] > -0.5) & (gdat.refr.catl[q][o]['ypos'] < gdat.numbside - 0.5))[0]

            
            print('Number of reference sources inside the cutout is %d...' % gdat.indxsourwthn[q][o].size)
            gdat.indxsourwthnbrgt[q][o] = np.intersect1d(gdat.indxsourwthn[q][o], gdat.indxsourbrgt[q][o])
            print('Number of bright reference sources inside the cutout is %d...' % gdat.indxsourwthnbrgt[q][o].size)
        
        if gdat.booldiag:
            for name in ['xpos', 'ypos']:
                if not np.isfinite(gdat.refr.catl[q][o][name]).all():
                    print('name')
                    print(name)
                    print('gdat.refr.catl[q][o][name]')
                    summgene(gdat.refr.catl[q][o][name])
                    print(gdat.refr.catl[q][o][name])
                    raise Exception('')

        if gdat.typedata == 'inje':
            
            # generate data
            gdat.numbtime[o] = gdat.listtime[o].size
            gdat.cntpmodlsimu = np.empty((gdat.numbside, gdat.numbside, gdat.numbtime[o]))
        
            for t in np.arange(gdat.numbtime[o]):
                gdat.cntpmodlsimu[:, :, t] = retr_cntpmodl(gdat, 'true', gdat.refr.catl[q][o]['xpos'], gdat.refr.catl[q][o]['ypos'], gdat.refr.catl[q][o]['cnts'], \
                                                                                                                                                0., gdat.true.parapsfn)
            gdat.cntpdata += gdat.cntpmodlsimu

        gdat.cntpdatasexp = gdat.cntpdata[:, :, 0]
    
        gdat.cntpdatatmed = np.nanmedian(gdat.cntpdata, axis=-1)
        
        if gdat.fitt.typepsfnshap == 'data':
            gdat.cntpdatapsfn = gdat.cntpdatatmed - np.percentile(gdat.cntpdatatmed, 90.)
            gdat.cntpdatapsfn[np.where(gdat.cntpdatapsfn < 0)] = 0
            gdat.cntpdatapsfn /= np.mean(gdat.cntpdatapsfn)
            #gdat.funcintppsfn = scipy.interpolate.interp2d(gdat.xposimagpsfn, gdat.yposimagpsfn, gdat.cntpdatapsfn, kind='cubic', fill_value=0.)(gdat.xposimag, gdat.yposimag)
    
        if len(gdat.cntpdata) == 0:
            raise Exception('')

        if not np.isfinite(gdat.cntpdata).all():
            raise Exception('')

        if gdat.booldiag:
            if gdat.cntpdatatmed.shape[0] != gdat.numbside:
                raise Exception('')
        
        if not np.isfinite(gdat.cntpdatatmed).all():
            raise Exception('')
        
        # plot data with initial catalogs (before PM correction)
        if gdat.boolplotcntp:
            strgtitl = gdat.strgtitlcntpplot
            for typecntpscal in gdat.listtypecntpscal:
                plot_cntp(gdat, gdat.cntpdatasexp, o, typecntpscal, gdat.pathimagtargsexp, 'cntpdatasexp_nopm', strgsave, 'refr', strgtitl=strgtitl)
        
            
        if gdat.typedata != 'simugene':
            
            if gdat.booldiag:
                for name in ['rasc', 'decl']:
                    if not np.isfinite(gdat.refr.catl[q][o][name]).all():
                        print('name')
                        print(name)
                        print('gdat.refr.catl[q][o][name]')
                        print(gdat.refr.catl[q][o][name])
                        summgene(gdat.refr.catl[q][o][name])
                        raise Exception('')
        
            print('Correcting the reference catalog for proper motion...')
            for q in gdat.refr.indxcatl:
                # epoch for correcting the RA and DEC for proper motion.
                gdat.epocpmot = (np.mean(gdat.listtime[o]) - 2433282.5) / 365.25 + 1950.
                print('Epoch: %g' % gdat.epocpmot)
                
                pmra = gdat.refr.catlbase[q]['pmra']
                pmde = gdat.refr.catlbase[q]['pmde']
                rascorig = gdat.refr.catlbase[q]['rascorig']
                declorig = gdat.refr.catlbase[q]['declorig']
                
                indx = np.where(np.isfinite(pmra) & np.isfinite(rascorig))[0]
                gdat.refr.catl[q][o]['rasc'][indx] = gdat.refr.catlbase[q]['rascorig'][indx] + gdat.refr.catlbase[q]['pmra'][indx] * (gdat.epocpmot - 2015.5) / (1000. * 3600.)
                
                if not np.isfinite(gdat.refr.catl[q][o]['rasc']).all():
                    raise Exception('')

                indx = np.where(np.isfinite(pmde) & np.isfinite(declorig))[0]
                gdat.refr.catl[q][o]['decl'][indx] = gdat.refr.catlbase[q]['declorig'][indx] + gdat.refr.catlbase[q]['pmde'][indx] * (gdat.epocpmot - 2015.5) / (1000. * 3600.)
                    
                if not np.isfinite(gdat.refr.catl[q][o]['decl']).all():
                    raise Exception('')

                gdat.refr.cequ[q][o] = np.empty((gdat.refr.catl[q][o]['rasc'].size, 2))
                gdat.refr.cequ[q][o][:, 0] = gdat.refr.catl[q][o]['rasc']
                gdat.refr.cequ[q][o][:, 1] = gdat.refr.catl[q][o]['decl']
                gdat.refr.cpix = gdat.listobjtwcss[o].all_world2pix(gdat.refr.cequ[q][o], 0)
                gdat.refr.catl[q][o]['xpos'] = gdat.refr.cpix[:, 0]
                gdat.refr.catl[q][o]['ypos'] = gdat.refr.cpix[:, 1]
                
                if not np.isfinite(gdat.refr.catl[q][o]['xpos']).all():
                    raise Exception('')

                if not np.isfinite(gdat.refr.catl[q][o]['ypos']).all():
                    raise Exception('')

                if gdat.booltpxf[o] and gdat.numbside < 11:
                    gdat.refr.catl[q][o]['xpos'] -= intg
                    gdat.refr.catl[q][o]['ypos'] -= intg
        
            if gdat.booldiag:
                for name in ['rasc', 'decl']:
                    if not np.isfinite(gdat.refr.catl[q][o][name]).all():
                        print('name')
                        print(name)
                        print('gdat.refr.catl[q][o][name]')
                        print(gdat.refr.catl[q][o][name])
                        summgene(gdat.refr.catl[q][o][name])
                        raise Exception('')
        
        if gdat.timeoffs != 0.:
            gdat.labltime = 'Time [BJD - %d]' % gdat.timeoffs
        else:
            gdat.labltime = 'Time [BJD]'

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
            
        if gdat.boolplotcent:
            # plot centroid
            for a in range(1):
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

        gdat.quat = np.zeros((gdat.numbtime[o], 3))
        #if len(os.listdir(gdat.pathcbvs)) == 0 and gdat.boolplotquat:
        if gdat.boolplotquat:
            print('Reading quaternions...')
            path = gdat.pathdatalygo + 'quat/'
            listfile = fnmatch.filter(os.listdir(path), 'tess*_sector%02d-quat.fits' % gdat.listtsec[o])
            if len(listfile) > 0:
                pathquat = path + listfile[0]
                listhdun = astropy.io.fits.open(pathquat)
                dataquat = listhdun[gdat.listtcam[o]].data
                headquat = listhdun[gdat.listtcam[o]].header
                #for k, key in enumerate(headquat.keys()):
                #    print(key + ' ' + str(headquat[k]))
                figr, axis = plt.subplots(3, 1, figsize=(12, 4), sharex=True)
                for k in range(1, 4):
                    strg = 'C%d_Q%d' % (gdat.listtcam[o], k)
                    gdat.quat[:, k-1] = scipy.interpolate.interp1d(dataquat['TIME'] + 2457000,  dataquat[strg], fill_value=0, bounds_error=False)(gdat.listtime[o])
                    minm = np.percentile(dataquat[strg], 0.05)
                    maxm = np.percentile(dataquat[strg], 99.95)
                    #axis[k-1].plot(dataquat['TIME'] + 2457000 - gdat.timeoffs, dataquat[strg], ls='', marker='.', ms=1)
                    axis[k-1].plot(gdat.listtime[o], gdat.quat[:, k-1], ls='', marker='.', ms=1)
                    axis[k-1].set_ylim([minm, maxm])
                    axis[k-1].set_ylabel('$Q_{%d}$' % k)
                axis[2].set_xlabel(gdat.labltime)
                path = gdat.pathimagtarg + 'quat_sc%02d.%s' % (gdat.listtsec[o], gdat.typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            else:
                print('Could not find any quaternion file.')

        if not np.isfinite(gdat.cntpdata).all():
            print('Warning! Not all data are finite.')
        
        if gdat.catlextr is not None:
            for qq in gdat.refr.indxcatl:
                gdat.refr.catl[qq+1]['rasc'] = gdat.catlextr[qq]['rasc']
                gdat.refr.catl[qq+1]['decl'] = gdat.catlextr[qq]['decl']

        if np.amin(gdat.cntpdata) < 0:
            print('Minimum of the image is negative.')

        # data variance
        gdat.vari = np.copy(gdat.cntpdata)

        
        gdat.fitt.catl = {}
    
        ## fitting catalog
        if gdat.typedata == 'simugene':
            for name in ['xpos', 'ypos', 'cnts']:
                gdat.fitt.catl[name] = gdat.refr.catl[0][o][name]
        if gdat.typedata != 'simugene':
            if gdat.typetarg == 'posi':
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
                gdat.fitt.catl['rasc'] = gdat.refr.catl[0][o]['rasc'][gdat.indxsourwthnbrgt[0][o]]
                gdat.fitt.catl['decl'] = gdat.refr.catl[0][o]['decl'][gdat.indxsourwthnbrgt[0][o]]
                gdat.fitt.catl['cntsesti'] = gdat.refr.catl[0][o]['cnts'][gdat.indxsourwthnbrgt[0][o]]
            
            skyyfitttemp = np.empty((gdat.fitt.catl['rasc'].size, 2))
            skyyfitttemp[:, 0] = gdat.fitt.catl['rasc']
            skyyfitttemp[:, 1] = gdat.fitt.catl['decl']
            if gdat.fitt.catl['rasc'].size == 0:
                raise Exception('')
            # transform sky coordinates into dedector coordinates and filter
            posifitttemp = gdat.listobjtwcss[o].all_world2pix(skyyfitttemp, 0)
            gdat.fitt.catl['xpos'] = posifitttemp[:, 0]
            gdat.fitt.catl['ypos'] = posifitttemp[:, 1]
            if gdat.booltpxf[o] and gdat.numbside < 11:
                gdat.fitt.catl['xpos'] -= intg
                gdat.fitt.catl['ypos'] -= intg
        else:
            gdat.fitt.catl['cntsesti'] = gdat.refr.catl[0][0]['cnts']

        if gdat.typedata != 'simugene':
            if gdat.booldiag:
                for name in ['rasc', 'decl']:
                    if not np.isfinite(gdat.fitt.catl[name]).all():
                        print('name')
                        print(name)
                        print('gdat.fitt.catl[name]')
                        print(gdat.fitt.catl[name])
                        summgene(gdat.fitt.catl[name])
                        raise Exception('')

                for name in ['xpos', 'ypos']:
                    if not np.isfinite(gdat.fitt.catl[name]).all():
                        print('name')
                        print(name)
                        print('gdat.fitt.catl[name]')
                        print(gdat.fitt.catl[name])
                        summgene(gdat.fitt.catl[name])
                        raise Exception('')

        if gdat.boolmerg:
            print('Checking whether it is necessary to merg fitting source pairs that are too close...')
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
                    
                    # delete the close source
                    gdat.fitt.catl['xpos'] = np.delete(gdat.fitt.catl['xpos'], m)
                    gdat.fitt.catl['ypos'] = np.delete(gdat.fitt.catl['ypos'], m)
            
                    print('Merging model point sources...')
                    
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
        
        # types of image plots
        listnameplotcntp = ['cntpdata', 'cntpmodl', 'cntpresi']
        
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

        gdat.strgheadtotl = 'time [BJD]'
        gdat.strgheadtotl += ', target rel flux, target rel flux err'
        for k in gdat.indxcomp:
            gdat.strgheadtotl += ', neig %d rel flux, neig %d rel flux err' % (k, k)
        gdat.strgheadtotl += ', bkg rel flux, bkg rel flux err'
    
        for e, nameanls in enumerate(gdat.listnameanls):
            if nameanls == 'aper':
                numbcomp = 2
            if nameanls == 'psfn':
                numbcomp = gdat.fitt.numbcomp
            gdat.fitt.arryrflx[o][e] = np.empty((gdat.numbtime[o], 1+2*numbcomp, 3, 3))
        
        gdat.stdvfittcnts = np.empty((gdat.numbtime[o], gdat.fitt.numbcomp))
        
        strgsave = retr_strgsave(gdat, strgsecc, 1, 1, o)
        
        if gdat.typepsfninfe != 'fixd':
            print('Modeling the data with a floating PSF...')
            # fit for the PSF
            numbsampwalk = 100000
            numbsampburnwalkinit = 0
            numbsampburnwalk = int(0.9 * numbsampwalk)
            
            gdat.typefittpsfnposi = 'fixd'
            gdat.typefittpsfncnts = 'vari'
            
            listnamepara = []
            listlablpara = []
            listminmpara = []
            listmaxmpara = []
            
            cntr = 0
            if gdat.typefittpsfnposi == 'vari':
                gdat.indxparapsfnxpos = np.empty(gdat.fitt.numbsour, dtype=int)
                gdat.indxparapsfnypos = np.empty(gdat.fitt.numbsour, dtype=int)
                for k in gdat.indxsour:
                    gdat.indxparapsfnxpos[k] = cntr
                    listminmpara.append(-1.)
                    listminmpara.append(1.)
                    listnamepara.append('xposdelt%04d' % k)
                    listlablpara.append(['$\Delta x_{%d}$' % k, ''])
                    cntr += 1

                    gdat.indxparapsfnypos[k] = cntr
                    listminmpara.append(-1.)
                    listminmpara.append(1.)
                    listnamepara.append('yposdelt%04d' % k)
                    listlablpara.append(['$\Delta y_{%d}$' % k, ''])
                    cntr += 1
            
            if gdat.typefittpsfncnts == 'vari':
                gdat.indxparapsfnflux = np.empty(gdat.fitt.numbsour, dtype=int)
                for k in gdat.indxsour:
                    gdat.indxparapsfnflux[k] = cntr
                    listminmpara.append(0.)
                    listmaxmpara.append(5. * np.amax(gdat.cntpdatasexp))
                    listnamepara.append('cnts%04d' % k)
                    listlablpara.append(['$C_{%d}$' % k, 'e$^-$/s'])
                    cntr += 1
            
            gdat.indxparapsfnback = np.empty(1, dtype=int)
            gdat.indxparapsfnback[0] = cntr
            listminmpara.append(0.)
            listmaxmpara.append(np.amax(gdat.cntpdatasexp))
            listnamepara.append('back')
            listlablpara.append(['$B$', 'e$^-$/s/px'])
            cntr += 1
            
            if gdat.fitt.typepsfnshap == 'gauscirc':
                gdat.indxparapsfnpsfn = np.empty(1, dtype=int)
                gdat.indxparapsfnpsfn[0] = cntr
                listminmpara.append(0.5)
                listmaxmpara.append(2.)
                listnamepara.append('sigmpsfn')
                listlablpara.append(['$\sigma$', 'px'])
                cntr += 1
            
            if gdat.fitt.typepsfnshap == 'gauselli':
                gdat.indxparapsfnpsfn = np.empty(4, dtype=int)
                gdat.indxparapsfnpsfn[0] = cntr
                listminmpara.append(0.5)
                listmaxmpara.append(2.)
                listnamepara.append('sigmpsfnxpos')
                listlablpara.append(['$\sigma_x$', 'px'])
                cntr += 1
            
                gdat.indxparapsfnpsfn[1] = cntr
                listminmpara.append(0.5)
                listmaxmpara.append(2.)
                listnamepara.append('sigmpsfnypos')
                listlablpara.append(['$\sigma_y$', 'px'])
                cntr += 1
            
                gdat.indxparapsfnpsfn[2] = cntr
                listminmpara.append(-10.)
                listmaxmpara.append(10.)
                listnamepara.append('fracskewpsfnxpos')
                listlablpara.append(['$\chi_x$', ''])
                cntr += 1
            
                gdat.indxparapsfnpsfn[3] = cntr
                listminmpara.append(-10.)
                listmaxmpara.append(10.)
                listnamepara.append('fracskewpsfnypos')
                listlablpara.append(['$\chi_y$', ''])
                cntr += 1
            
            if gdat.fitt.typepsfnshap == 'empi':
                gdat.indxparapsfnpsfn = np.empty(gdat.numbparapsfnempi, dtype=int)
                for r in gdat.indxparapsfnempi:
                    gdat.indxparapsfnpsfn[r] = cntr
                    listminmpara.append(0.)
                    listmaxmpara.append(np.amax(gdat.cntpdatasexp))
                    listnamepara.append('amplpsfnempi%04d' % r)
                    listlablpara.append(['$\rho_{%d}$' % r, ''])
                    cntr += 1
            
            gdat.numbparapsfn = cntr
            gdat.indxparapsfn = np.arange(gdat.numbparapsfn)
            listscalpara = ['self' for k in gdat.indxparapsfn]
            listmeangauspara = None
            liststdvgauspara = None
            
            listminmpara = np.array(listminmpara)
            listmaxmpara = np.array(listmaxmpara)
            
            strgextn = 'psfn_%s_%s' % (strgsecc, gdat.fitt.typepsfnshap)
           
            boolforcrepr = gdat.boolplotcntp

            dictlablscalparaderi = None
            if gdat.fitt.catl['xpos'].size > 1:
                if dictlablscalparaderi is None:
                    dictlablscalparaderi = dict()
                dictlablscalparaderi['raticont'] = [['$f_c$', ''], 'self']
            if gdat.fitt.catl['xpos'][0] == (gdat.numbside - 1.) / 2. and gdat.fitt.catl['ypos'][0] == (gdat.numbside - 1.) / 2. and gdat.fitt.catl['xpos'].size == 1:
                if dictlablscalparaderi is None:
                    dictlablscalparaderi = dict()
                dictlablscalparaderi['fraccent'] = [['$f_p$', ''], 'self']
            retr_dictderi = retr_dictderipsfn

            dictsamp = tdpy.samp(gdat, numbsampwalk, retr_llik, listnamepara, listlablpara, listscalpara, listminmpara, listmaxmpara, numbsampburnwalkinit=numbsampburnwalkinit, \
                                        retr_dictderi=retr_dictderi, \
                                        dictlablscalparaderi=dictlablscalparaderi, \
                                        numbsampburnwalk=numbsampburnwalk, \
                                        boolforcrepr=boolforcrepr, \
                                        pathbase=gdat.pathimagtargsexp, strgextn=strgextn, \
                                        #typefileplot=typefileplot, \
                                        )
            
            strgsavepsfn = strgsave + gdat.fitt.typepsfnshap

            if gdat.boolplotcntp:
                # plot the posterior median image, point source models and the residual
                for namevarb in ['cntpmodlpntssexp', 'cntpmodlsexp', 'cntpresisexp']:
                    setattr(gdat, namevarb, dictsamp[namevarb])
                    strgtitl = gdat.strgtitlcntpplot
                    boolresi = namevarb == 'cntpresisexp'
                    for typecntpscal in gdat.listtypecntpscal:
                        plot_cntp(gdat, np.median(dictsamp[namevarb], 0), o, typecntpscal, gdat.pathimagtargsexp, namevarb, strgsavepsfn, 'fitt', \
                                                                                                                                strgtitl=strgtitl, boolresi=boolresi)
                for typecntpscal in gdat.listtypecntpscal:
                    plot_cntp(gdat, gdat.cntpdatasexp, o, typecntpscal, gdat.pathimagtargsexp, 'cntpdatasexp', strgsavepsfn, 'fitt', strgtitl=gdat.strgtitlcntpplot)
            
            gdat.fitt.catl['cnts'] = np.empty(gdat.fitt.numbsour)
            for k in gdat.indxsour:
                gdat.fitt.catl['cnts'][k] = np.median(dictsamp['cnts%04d' % k])
            if gdat.fitt.typepsfnshap == 'gauscirc':
                gdat.fitt.parapsfn = np.empty(1)
                gdat.fitt.parapsfn[0] = np.median(dictsamp['sigmpsfn'])
            if gdat.fitt.typepsfnshap == 'gauselli':
                gdat.fitt.parapsfn = np.empty(2)
                gdat.fitt.parapsfn[0] = np.median(dictsamp['sigmpsfnxpos'])
                gdat.fitt.parapsfn[1] = np.median(dictsamp['sigmpsfnypos'])
                gdat.fitt.parapsfn[2] = np.median(dictsamp['fracskewpsfnxpos'])
                gdat.fitt.parapsfn[3] = np.median(dictsamp['fracskewpsfnypos'])
            if gdat.fitt.typepsfnshap == 'empi':
                gdat.fitt.parapsfn = np.empty(gdat.numbparapsfnempi)
                for r in gdat.indxparapsfnempi:
                    gdat.fitt.parapsfn[r] = np.median(dictsamp['amplpsfnempi%04d' % r])
        
        else:
            gdat.fitt.catl['cnts'] = gdat.refr.catl[0][o]['cnts']
            
            if gdat.fitt.typepsfnshap == 'gauscirc':
                gdat.fitt.parapsfn = np.empty(1)
                gdat.fitt.parapsfn[0] = gdat.fitt.sigmpsfn
            elif gdat.fitt.typepsfnshap == 'gauselli':
                gdat.fitt.parapsfn = np.empty(4)
                gdat.fitt.parapsfn[0] = gdat.fitt.sigmpsfnxpos
                gdat.fitt.parapsfn[1] = gdat.fitt.sigmpsfnypos
                gdat.fitt.parapsfn[2] = gdat.fitt.fracskewpsfnxpos
                gdat.fitt.parapsfn[3] = gdat.fitt.fracskewpsfnypos
            elif gdat.fitt.typepsfnshap == 'data':
                pass
            else:
                raise Exception('')

        if gdat.fitt.typepsfnshap != 'data':
            gdat.dictoutp['raticont'] = retr_raticont(gdat, gdat.fitt.catl['xpos'], gdat.fitt.catl['ypos'], gdat.fitt.catl['cnts'], gdat.fitt.parapsfn)
        
        if gdat.boolanim:
            if gdat.boolanimframtotl:
                numbplotanim = gdat.numbtime[o]
            else:
                numbplotanim = 100
            # time indices to be included in the animation
            gdat.indxtimeanim = np.linspace(0., gdat.numbtime[o] - 1., numbplotanim).astype(int)
            # get time string
            if gdat.typedata != 'simugene':
                objttime = astropy.time.Time(gdat.listtime[o], format='jd', scale='utc')#, out_subfmt='date_hm')
                listtimelabl = objttime.iso
            else:
                listtimelabl = gdat.listtime[o].astype(str)
        
        # plot data with initial catalogs (after PM correction)
        if gdat.boolplotcntp:
            for typecntpscal in gdat.listtypecntpscal:
                strgtitl = gdat.strgtitlcntpplot
                plot_cntp(gdat, gdat.cntpdatasexp, o, typecntpscal, gdat.pathimagtargsexp, 'cntpdatasexp', strgsave, 'refr', strgtitl=strgtitl)
        
        # plot a histogram of data counts
        if gdat.boolplothhistcntp:
            plot_histcntp(gdat, gdat.cntpdata, gdat.pathimagtarg, 'cntpdata', strgsave) 
        
        gdat.offsposi = 0.25
        gdat.listoffsposi = np.linspace(-gdat.offsposi, gdat.offsposi, 3)
        gdat.numboffs = gdat.listoffsposi.size
        gdat.indxoffs = np.arange(gdat.numboffs)
        for e in gdat.indxanls:
            gdat.listnois[o][e] = np.zeros((gdat.numboffs, gdat.numboffs))
        for x in gdat.indxoffs:
            for y in gdat.indxoffs:
                
                if not gdat.boolfittoffs and (x != 1 or y != 1):
                    continue
                
                print('Offset indices: %d %d' % (x, y))
                strgsave = retr_strgsave(gdat, strgsecc, x, y, o)
                
                # PSF photometry
                if 'psfn' in gdat.listnameanls:
                    ## paths
                    pathsaverflxpsfn = gdat.pathdatatarg + 'rflx_psfn' + strgsave + '.csv'
                    pathsaverflxpsfntarg = gdat.pathdatatarg + 'rflxtarg_psfn' + strgsave + '.csv'
                    pathsavemeta = gdat.pathdatatarg + 'metaregr' + strgsave + '.csv'
                        
                    ## introduce the positional offset
                    xpostemp = np.copy(gdat.fitt.catl['xpos'])
                    ypostemp = np.copy(gdat.fitt.catl['ypos'])
                    xpostemp[0] = gdat.fitt.catl['xpos'][0] + gdat.listoffsposi[x]
                    ypostemp[0] = gdat.fitt.catl['ypos'][0] + gdat.listoffsposi[y]
                    xpostemp = xpostemp[None, :] + gdat.quat[:, None, 1]
                    ypostemp = ypostemp[None, :] + gdat.quat[:, None, 0]
                    
                    if not os.path.exists(pathsaverflxpsfn):
                        
                        print('Performing PSF photometry...')
                        timeinit = timemodu.time()
                        
                        gdat.cntpdataflat = gdat.cntpdata.reshape((-1, gdat.numbtime[o]))
                        gdat.variflat = gdat.vari.reshape((-1, gdat.numbtime[o]))
                        gdat.indxpixlnzer = np.where((gdat.cntpdataflat != 0).all(axis=1))[0]
                        
                        gdat.covafittcnts = np.empty((gdat.numbtime[o], gdat.fitt.numbsour + 1, gdat.fitt.numbsour + 1))
                        gdat.mlikfittcntsinit = np.empty((gdat.numbtime[o], gdat.fitt.numbsour + 1))
                                
                        print('Solving for the best-fit raw light curves of the sources...')
                        # solve the linear system
                        cntstemp = np.ones((gdat.numbtime[o], gdat.fitt.numbsour))
                        matrdesi = np.ones((gdat.indxpixlnzer.size, gdat.fitt.numbsour + 1))
                        
                        if gdat.indxpixlnzer.size != gdat.numbpixl:
                            raise Exception('')

                        for t in gdat.indxtime[o]:
                            
                            if gdat.fitt.typepsfnshap == 'data':
                                matrdesi[:, 0] = gdat.cntpdatapsfn.flatten()
                            else:
                                for k in np.arange(gdat.fitt.numbsour):
                                    matrdesi[:, k] = retr_cntpmodl(gdat, 'fitt', xpostemp[t, k, None], ypostemp[t, k, None], cntstemp[t, k, None], 0., \
                                                                                                                            gdat.fitt.parapsfn).flatten()[gdat.indxpixlnzer]
                            
                            gdat.mlikfittcntsinit[t, :], gdat.covafittcnts[t, :, :] = retr_mlikregr(gdat.cntpdataflat[gdat.indxpixlnzer, t], matrdesi, gdat.variflat[gdat.indxpixlnzer, t])
                        
                        if gdat.booldiag:
                            if not (gdat.mlikfittcntsinit > 0).all():
                                for k in gdat.indxcomp:
                                    print('k')
                                    print(k)
                                    print('gdat.mlikfittcntsinit[:, k]')
                                    summgene(gdat.mlikfittcntsinit[:, k])
                                raise Exception('')
                            
                        for g in range(len(matrdesi)):
                            print('%g %g %g %g' % (np.median(gdat.cntpdataflat[g, :]), np.median(gdat.variflat[g, :]), matrdesi[g, 0], matrdesi[g, 1]))
                        
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
                                print('gdat.mlikfittcntsinit[:, k]')
                                summgene(gdat.mlikfittcntsinit[:, k])
                                #raise Exception('')
                        
                        gdat.varifittcnts = gdat.stdvfittcnts**2
        
                        if False and gdat.booldetrcbvs:
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
                        
                        print('Normalizing by the median flux...')
                        gdat.fitt.arryrflx[o][0][:, :, 0, x, y] = gdat.listtime[o][:, None]
                        
                        gdat.fitt.arryrflx[o][0][:, :, 1:, x, y] = gdat.mlikfittcnts
                        gdat.fitt.arryrflx[o][0][:, :, 2, x, y] = gdat.stdvfittcnts
                        # normalize fluxes to get relative fluxes
                        if gdat.boolnorm:
                            for k in gdat.indxcomp:
                                gdat.fitt.arryrflx[o][0][:, 1+2*k:1+2*k+2, x, y] /= gdat.medifittcnts[k]
                        
                        # write the meta data to the disk
                        print('Writing meta data to %s...' % pathsavemeta)
                        arry = gdat.medifittcnts
                        np.savetxt(pathsavemeta, arry, delimiter=',', header='Temporal median counts for each component')
                        
                        # write the light curves to the disk
                        print('Writing all light curves to %s...' % pathsaverflxpsfn)
                        np.savetxt(pathsaverflxpsfn, gdat.fitt.arryrflx[o][0][:, :, x, y], delimiter=',', header=gdat.strgheadtotl)
                        
                        # write the target light curve to the disk
                        print('Writing the target light curve to %s...' % pathsaverflxpsfntarg)
                        np.savetxt(pathsaverflxpsfntarg, gdat.fitt.arryrflx[o][0][:, :3, x, y], delimiter=',', header=gdat.strgheadtarg)
                        
                    else:
                        print('Skipping the regression...')
                    
                        print('Reading from %s...' % pathsavemeta)
                        gdat.medifittcnts = np.loadtxt(pathsavemeta, delimiter=',', skiprows=1)
                        
                        print('Reading from %s...' % pathsaverflxpsfn)
                        gdat.fitt.arryrflx[o][0][:, :, x, y] = np.loadtxt(pathsaverflxpsfn, delimiter=',', skiprows=1)
                        gdat.listtime[o] = gdat.fitt.arryrflx[o][0][:, 0, x, y]
                        
                        for k in gdat.indxcomp:
                            if gdat.boolnorm:
                                gdat.mlikfittcnts[:, k] = gdat.medifittcnts[k] * gdat.fitt.arryrflx[o][0][:, 1+2*k, x, y]
                            else:
                                gdat.mlikfittcnts[:, k] = gdat.fitt.arryrflx[o][0][:, 1+2*k, x, y]

                    if gdat.fitt.typepsfnshap != 'data':
                        print('Evaluating the model at all time bins...')
                        cntpbacktser = gdat.mlikfittcnts[:, -1]
                        timeinit = timemodu.time()
                        gdat.cntpmodl = np.empty_like(gdat.cntpdata)
                        gdat.cntpmodlpnts = np.empty_like(gdat.cntpdata)
                        for t in gdat.indxtime[o]:
                            gdat.cntpmodlpnts[:, :, t] = retr_cntpmodl(gdat, 'fitt', xpostemp[t, :], ypostemp[t, :], gdat.mlikfittcnts[t, :-1], 0., gdat.fitt.parapsfn)
                            gdat.cntpmodl[:, :, t] = retr_cntpmodl(gdat, 'fitt', xpostemp[t, :], ypostemp[t, :], gdat.mlikfittcnts[t, :-1], cntpbacktser[t], gdat.fitt.parapsfn)
                        
                        timefinl = timemodu.time()
                        print('Done in %g seconds.' % (timefinl - timeinit))
                            
                        gdat.cntpresi = gdat.cntpdata - gdat.cntpmodl
                        
                        chi2 = np.mean(gdat.cntpresi**2 / gdat.cntpdata) + 2 * gdat.fitt.numbsour
                        
                        ## temporal medians
                        for strg in ['modl', 'modlpnts', 'resi']:
                            cntp = getattr(gdat, 'cntp' + strg)
                            setattr(gdat, 'cntp' + strg + 'tmed', np.nanmedian(cntp, axis=-1))
                
                        # plot a histogram of data counts
                        if gdat.boolplothhistcntp:
                            plot_histcntp(gdat, gdat.cntpresi, gdat.pathimagtarg, 'cntpresi', strgsave) 
                
                if (x == 1 and y == 1) and 'aper' in gdat.listnameanls:
                    print('Performing aperture photometry...')
                    
                    ## paths
                    pathsaverflxpsfn = gdat.pathdatatarg + 'rflx_aper' + strgsave + '.csv'
                    pathsaverflxpsfntarg = gdat.pathdatatarg + 'rflxtarg_aper' + strgsave + '.csv'
                    pathsavemeta = gdat.pathdatatarg + 'metaregr' + strgsave + '.csv'
                    
                    if gdat.listpixlapertarg is None:
                        gdat.listpixlapertarg = [[]]
                        gdat.listpixlapertarg[0] = np.where(gdat.cntpdatatmed > np.percentile(gdat.cntpdata, 60.))
                        gdat.listpixlaperback = [[]]
                        gdat.listpixlaperback[0] = np.where(gdat.cntpdatatmed < np.percentile(gdat.cntpdata, 60.))
                    gdat.numbaper = len(gdat.listpixlapertarg)
                    gdat.indxaper = np.arange(gdat.numbaper)
                    
                    #gdat.listpixlapertargpair = [[] for p in gdat.indxaper]
                    #for p in gdat.indxaper:
                    #    for w in range(len(gdat.listpixlaperpair[p])):
                    #        gdat.listpixlapertargpair[p] = [gdat.listpixlapertarg[p][0][w], gdat.listpixlapertarg[p][1][w]]
                    #    for x in gdat.indxside:
                    #        for y in gdat.indxside:
                    #            if not (x, y) in (gdat.listpixlaperpair[p]:
                    #                gdat.listpixlaperback[p][0].append(x)
                    #                gdat.listpixlaperback[p][1].append(y)
                    #

                    gdat.cntpaper = np.zeros((gdat.numbtime[o], gdat.numbaper, 2))
                    for p in gdat.indxaper:
                        gdat.cntpaper[:, p, 0] = np.sum(gdat.cntpdata[gdat.listpixlapertarg[p][0], gdat.listpixlapertarg[p][1], :], axis=0)
                        gdat.cntpaper[:, p, 1] = np.sum(gdat.cntpdata[gdat.listpixlaperback[p][0], gdat.listpixlaperback[p][1], :], axis=0)
                        
                        vari = gdat.cntpaper[:, p, 0] * 0. + np.mean(gdat.cntpaper[:, p, 0])
                        matrdesi = np.ones((gdat.numbtime[o], 2))
                        matrdesi[:, 0] = gdat.cntpaper[:, p, 1]
                        matrdesi[:, 1] = 1. + 0 * gdat.cntpaper[:, p, 1]
                        coef, cova = retr_mlikregr(gdat.cntpaper[:, p, 0], matrdesi, vari)
                        gdat.cntpaper[:, p, 0] -= coef[0] * gdat.cntpaper[:, p, 1]
                    if gdat.boolnorm:
                        gdat.cntpaper /= np.median(gdat.cntpaper, 0)
                    gdat.cntpapertarg = gdat.cntpaper[:, :, 0]
                    gdat.cntpaperback = gdat.cntpaper[:, :, 1]

                    gdat.fitt.arryrflx[o][gdat.indxanlsaper][:, 0, x, y] = gdat.listtime[o]
                    gdat.fitt.arryrflx[o][gdat.indxanlsaper][:, 1, x, y] = gdat.cntpapertarg[:, 0]
                    gdat.fitt.arryrflx[o][gdat.indxanlsaper][:, 2, x, y] = 0.01 * gdat.cntpapertarg[:, 0]
                    gdat.fitt.arryrflx[o][gdat.indxanlsaper][:, 3, x, y] = gdat.cntpaperback[:, 0]
                    gdat.fitt.arryrflx[o][gdat.indxanlsaper][:, 4, x, y] = 0.01 * gdat.cntpaperback[:, 0]
                    print('Writing all light curves to %s...' % pathsaverflxpsfn)
                    np.savetxt(pathsaverflxpsfn, gdat.fitt.arryrflx[o][gdat.indxanlsaper][:, :, x, y], delimiter=',', header=gdat.strgheadtotl)
                    
                    print('Writing the target light curve to %s...' % pathsaverflxpsfntarg)
                    np.savetxt(pathsaverflxpsfntarg, gdat.fitt.arryrflx[o][gdat.indxanlsaper][:, :3, x, y], delimiter=',', header=gdat.strgheadtarg)
                
                if gdat.booldiag:
                    for a in range(gdat.listtime[o].size):
                        if a != gdat.listtime[o].size - 1 and gdat.listtime[o][a] >= gdat.listtime[o][a+1]:
                            raise Exception('')
                
                # assess noise in the light curve
                for e in gdat.indxanls:
                    if x == 1 and y == 1:
                        print('gdat.fitt.arryrflx[o][e][:, 1, x, y]')
                        summgene(gdat.fitt.arryrflx[o][e][:, 1, x, y])
                        arryrflxrbn = np.copy(gdat.fitt.arryrflx[o][e][:, :3, x, y])
                        arryrflxrbn[:, 1] = gdat.fitt.arryrflx[o][e][:, 1, x, y] - scipy.ndimage.median_filter(gdat.fitt.arryrflx[o][e][:, 1, x, y], size=50)
                        print('arryrflxrbn[:, 1]')
                        summgene(arryrflxrbn[:, 1])
                        arryrflxrbn = ephesus.rebn_tser(arryrflxrbn, delt=1. / 24.)
                        print('arryrflxrbn[:, 1]')
                        summgene(arryrflxrbn[:, 1])
                        gdat.listnois[o][e][x, y] = 1e6 * np.nanstd(arryrflxrbn[:, 1]) / np.median(gdat.fitt.arryrflx[o][e][:, 1, x, y])
                        print('gdat.listnois[o][e][x, y]')
                        summgene(gdat.listnois[o][e][x, y])
                        print('FUDGE FACTOR!!!')
                        gdat.listnois[o][e][x, y] /= 30.
                
                    if gdat.booldiag:
                        if len(gdat.fitt.arryrflx[o][e]) == 0:
                            raise Exception('')
                    
                if gdat.boolplotrflx:
                    print('Plotting light curves...')
                    if gdat.listlimttimetzom is not None:
                        gdat.indxtimelimt = []
                        for limttimeplot in gdat.listlimttimetzom:
                            gdat.indxtimelimt.append(np.where((gdat.listtime[o] > limttimeplot[0]) & (gdat.listtime[o] < limttimeplot[1]))[0])
                    
                    # plot light curve derived from aperture photometry
                    if 'aper' in gdat.listnameanls:
                        for p in gdat.indxaper:
                            plot_lcur(gdat, gdat.cntpapertarg[:, p], 0.01 * gdat.cntpapertarg[:, p], 0, o, '_' + strgsecc, strgsave, 'aper%04d' % p)
                            plot_lcur(gdat, gdat.cntpaperback[:, p], 0.01 * gdat.cntpaperback[:, p], 1, o, '_' + strgsecc, strgsave, 'aper%04d' % p)
                        
                    #if gdat.booldetrcbvs:
                    #    plot_lcur(gdat, gdat.fitt.arryrflx[o][:, 0, 1, x, y], gdat.fitt.arryrflx[o][:, 0, 2, x, y], 0, o, '_' + strgsecc, strgsave, 'detrcbvs')
                        
                    # plot the light curve of the sources and background
                    if 'psfn' in gdat.listnameanls:
                        for k in gdat.indxcomp:
                            
                            if x == 1 and y == 1 or k == 0:
                                
                                plot_lcur(gdat, gdat.fitt.arryrflx[o][0][:, 1+2*k, x, y], gdat.fitt.arryrflx[o][0][:, 1+2*k+1, x, y], k, o, '_' + strgsecc, \
                                                                                                                            strgsave, 'stan', cdpp=gdat.listnois[o][0][x, y])
                                if gdat.boolrefrtser[o] and x == 1 and y == 1:
                                    plot_lcurcomp(gdat, gdat.fitt.arryrflx[o][0][:, 1+2*k, x, y], gdat.fitt.arryrflx[o][0][:, 1+2*k+1, x, y], k, o, '_' + strgsecc, strgsave, 'stan')
                                
                                if gdat.listlimttimetzom is not None:
                                    for p in range(len(gdat.listlimttimetzom)):
                                        plot_lcur(gdat, gdat.fitt.arryrflx[o][0][:, 1+2*k, x, y], gdat.fitt.arryrflx[o][0][:, 1+2*k+1, x, y], k, o, '_' + strgsecc, \
                                                                                              strgsave, 'zoom', indxtimelimt=gdat.indxtimelimt[p], indxtzom=p)
                            
                if 'psfn' in gdat.listnameanls and gdat.fitt.typepsfnshap != 'data':
                    for typeplotcntp in listtypeplotcntp:
                        for nameplotcntp in ['cntpmodl', 'cntpmodlpnts', 'cntpresi']:
                            
                            # make animation plot
                            if typeplotcntp == 'anim':
                                pathanim = retr_pathvisu(gdat, gdat.pathimagtarg, nameplotcntp, strgsave, typevisu='anim')
                                if os.path.exists(pathanim):
                                    continue

                                print('Making an animation of frame plots...')
                                strg = ''
                            else:
                                strg = 'tmed'
                            
                            cntptemp = getattr(gdat, nameplotcntp + strg)
                            
                            boolresi = nameplotcntp.startswith('cntpresi')
                            
                            for typecntpscal in gdat.listtypecntpscal:
                            
                                if typeplotcntp == 'tmed':
                                    if nameplotcntp == 'cntpdata':
                                        listtypecatlplot = gdat.listtypecatlplot
                                    else:
                                        listtypecatlplot = ['fitt']
                                    for typecatlplot in listtypecatlplot:
                                        strgtitl = gdat.strgtitlcntpplot
                                        plot_cntp(gdat, cntptemp, o, typecntpscal, gdat.pathimagtarg, nameplotcntp + strg, strgsave, typecatlplot, \
                                                                                                                            xposoffs=gdat.listoffsposi[x], \
                                                                                                                            yposoffs=gdat.listoffsposi[y], \
                                                                                                                            strgtitl=strgtitl, boolresi=boolresi)
                                if x == 1 and y == 1 and typeplotcntp == 'anim' and nameplotcntp == 'cntpresi':
                                    # color scales
                                    setp_cntp(gdat, nameplotcntp, typecntpscal)
                                    
                                    vmin = getattr(gdat, 'vmin' + nameplotcntp + typecntpscal)
                                    vmax = getattr(gdat, 'vmax' + nameplotcntp + typecntpscal)
                            
                                    args = [gdat, cntptemp, o, typecntpscal, nameplotcntp, strgsave]
                                    kwag = { \
                                            'boolresi': boolresi, \
                                            'listindxpixlcolr': gdat.listpixlaper, \
                                            'listtimelabl':listtimelabl, \
                                            'vmin':vmin, 'vmax':vmax, \
                                            'lcur':gdat.fitt.arryrflx[o][gdat.indxanlsutil][:, 1, x, y], \
                                            'time':gdat.listtime[o], \
                                           }
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
     
    for e in gdat.indxanls:
        gdat.listnoiscent = np.array([gdat.listnois[o][e][1, 1] for o in gdat.indxtsec])
        nameanls = gdat.listnameanls[e]
        gdat.dictoutp['meannois%s' % nameanls] = np.mean(gdat.listnoiscent)
        gdat.dictoutp['medinois%s' % nameanls] = np.median(gdat.listnoiscent)
        gdat.dictoutp['stdvnois%s' % nameanls] = np.std(gdat.listnoiscent)
        gdat.dictoutp['listarry%s' % nameanls] = [gdat.fitt.arryrflx[o][e][:, :, 1, 1] for o in gdat.indxtsec]
        
    for name, valu in gdat.true.__dict__.items():
        gdat.dictoutp['true'+name] = valu
    
    for name, valu in gdat.fitt.__dict__.items():
        gdat.dictoutp['fitt'+name] = valu
    
    for name, valu in gdat.__dict__.items():
        gdat.dictoutp[name] = valu
    
    timefinltotl = timemodu.time()
    print('lygos ran in %g seconds.' % (timefinltotl - timeinittotl))
    print('')                
    
    return gdat.dictoutp


