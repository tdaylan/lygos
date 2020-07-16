import numpy as np

from scipy import ndimage

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
    
    pathdata = os.environ['TCAT_DATA_PATH'] + '/tesspsfn/'
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


def retr_pathcntp(strgtemp, gdat, strgcolr, strgsecc):
    
    path = gdat.pathimagobjt + 'cntp%s_%s%s_%s' % (strgtemp, gdat.strgcnfg, strgcolr, strgsecc)
    
    return path


def plot_cntpwrap(gdat, cntp, o, strgsecc, boolresi=False, listindxpixlcolr=None, indxstarcolr=None, indxtimeplot=None, \
                                                                                time=None, lcur=None, \
                                                                                vmin=None, vmax=None, \
                                                                                listtime=None, listtimelabl=None, \
                                                                                ):
    
    if time is None:
        time = [None for t in gdat.indxtime]

    if listtimelabl is None:
        listtimelabl = [None for t in gdat.indxtime]

    if listindxpixlcolr is None:
        strgcolr = ''
    else:
        strgcolr = 'cl%02d_' % indxstarcolr
    
    if indxtimeplot is None:
        indxtimeplot = np.arange(cntp.shape[3])
    
    listpath = []
    if boolresi:
        strgtemp = 'resi'
    else:
        strgtemp = 'data'
    pathbase = retr_pathcntp(strgtemp, gdat, strgcolr, strgsecc)

    for tt, t in enumerate(indxtimeplot):
        path = '%s_%06d.%s' % (pathbase, t, gdat.strgplotextn)
        
        # make title 
        strgtitl = gdat.strgtitlcntpplot
        if listtimelabl[t] is not None:
            strgtitl += ', %s' % listtimelabl[t]
        
        if not os.path.exists(path):
            plot_cntp(gdat, cntp[0, :, :, t], path, o, strgtitl=strgtitl, boolresi=boolresi, listindxpixlcolr=listindxpixlcolr, \
                                                                                            timelabl=listtimelabl[t], thistime=time[t], \
                                                                                                vmin=vmin, vmax=vmax, lcur=lcur, time=time)
        
        listpath.append(path)

    return listpath


def plot_cntp(gdat, cntp, path, o, cbar='Greys_r', strgtitl='', boolresi=False, xposoffs=None, yposoffs=None, \
                                                                                           lcur=None, boolanno=True, \
                                                                                           time=None, timelabl=None, thistime=None, \
                                                                                           vmin=None, vmax=None, listindxpixlcolr=None):
    
    if gdat.cntpscaltype == 'asnh':
        cntp = np.arcsinh(cntp)
    
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
    
        axis[0].scatter(gdat.catlrefr[0]['xpos'], gdat.catlrefr[0]['ypos'], alpha=1., s=gdat.sizeplotsour, color='r')
        indx = np.where(gdat.catlrefr[0]['tmag'] < 14.)[0]
        for indxtemp in indx:
            axis[0].text(gdat.catlrefr[0]['xpos'][indxtemp], gdat.catlrefr[0]['ypos'][indxtemp], '%s' % ('TIC %s' % gdat.catlrefr[0]['tici'][indxtemp]), \
                                                                                                                                                  color='g')
        axis[0].scatter(gdat.catlrefr[0]['xpos'][indx], gdat.catlrefr[0]['ypos'][indx], alpha=1., s=3*gdat.sizeplotsour, color='r')
        indx = np.where(gdat.catlrefr[0]['tmag'] < 10.)[0]
        for indxtemp in indx:
            print('gdat.catlrefr[0][tici][indxtemp]')
            print(gdat.catlrefr[0]['tici'][indxtemp])
            axis[0].text(gdat.catlrefr[0]['xpos'][indxtemp], gdat.catlrefr[0]['ypos'][indxtemp], '%s' % ('TIC %d' % gdat.catlrefr[0]['tici'][indxtemp]), \
                                                                                                                                                  color='g')
        axis[0].scatter(gdat.catlrefr[0]['xpos'][indx], gdat.catlrefr[0]['ypos'][indx], alpha=1., s=9*gdat.sizeplotsour, color='r')
        
        xposfitt = np.copy(gdat.xposfitt)
        yposfitt = np.copy(gdat.yposfitt)
        if xposoffs is not None:
            # add the positional offset, if any
            xposfitt += xposoffs
            yposfitt += yposoffs
        # target
        axis[0].scatter(xposfitt[0], yposfitt[0], alpha=1., color='b', s=gdat.sizeplotsour*2, marker='x')
        # neighbors
        axis[0].scatter(xposfitt[1:], yposfitt[1:], alpha=1., s=gdat.sizeplotsour, color='b')
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
    if gdat.cntpscaltype == 'asnh':
        tick = cbar.get_ticks()
        tick = np.sinh(tick)
        labl = ['%d' % tick[k] for k in range(len(tick))]
        cbar.set_ticklabels(labl)
    
    if lcur is not None:
        axis[1].plot(time, lcur, color='black', ls='', marker='o', markersize=1)
        axis[1].set_xlabel('Time [BJD]') 
        axis[1].set_ylabel('Relative flux') 
        axis[1].axvline(thistime)
        
    print('Writing to %s...' % path)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_lcur(gdat, lcurmodl, stdvlcurmodl, k, indxsectplot, strgsecc, strgoffs, timeedge=None, strgyaxi='rflx', \
                    boolzoom=False, strgextn='', boolphas=False, indxtimelimt=None, indxzoom=None, boolerrr=False):
    
    if k == 0:
        lablcomp = 'Target source, ' % k
    elif k == gdat.numbcomp - 1:
        lablcomp = 'Background, '
    else:
        lablcomp = 'Source %d, ' % k

    timedatatemp = np.copy(gdat.timedata[indxsectplot])
    # temp
    for q in gdat.indxrefrlcur:
        timerefrtemp = gdat.refrtime[q][indxsectplot]
    
    if indxtimelimt is not None:
        timedatatemp = timedatatemp[indxtimelimt]
        stdvlcurmodl = stdvlcurmodl[indxtimelimt]
        lcurmodl = lcurmodl[indxtimelimt]

    if boolphas:
        timedatatemp = ((timedatatemp - gdat.epoc) / gdat.peri + 0.25) % 1.
        timerefrtemp = ((timerefrtemp - gdat.epoc) / gdat.peri + 0.25) % 1.
        lablxaxi = 'Phase'
    else:
        lablxaxi = 'Time [BJD]'

    for a in gdat.listmodeplot:
        
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
        temp, listcaps, temp = axis[0].errorbar(timedatatemp, lcurmodl, yerr=yerr, color='b', ls='', markersize=5, \
                                                                                marker='o', lw=3, alpha=0.3, label='TCAT')
        for caps in listcaps:
            caps.set_markeredgewidth(3)
        
        if timeedge is not None:
            for timeedgetemp in timeedge[1:-1]:
                axis[0].axvline(timeedgetemp, ls='--', color='grey', alpha=0.5)
        if a == 1 and k == 0:
            for q in gdat.indxrefrlcur:
                if boolerrr:
                    yerr = gdat.stdvrefrrflx[q][indxsectplot]
                else:
                    yerr = None
                temp, listcaps, temp = axis[0].errorbar(timerefrtemp, gdat.refrrflx[q][indxsectplot], \
                                                    yerr=yerr, color=gdat.colrrefrlcur[q], ls='', markersize=5, \
                                                                                            marker='o', lw=3, alpha=0.3, label=gdat.lablrefrlcur[q])
                for caps in listcaps:
                    caps.set_markeredgewidth(3)
            
            ## residual
            if lcurmodl.size == gdat.refrrflx[q][indxsectplot][:gdat.numbtimecutt].size:
                for q in gdat.indxrefrlcur:
                    ydat = lcurmodl - gdat.refrrflx[q][indxsectplot][:gdat.numbtimecutt]
                    if boolerrr:
                        yerr = None
                    else:
                        yerr = None
                    axis[1].errorbar(timedatatemp[:gdat.numbtimecutt], ydat, yerr=yerr, label=gdat.lablrefrlcur[q], \
                                                        color='k', ls='', marker='o', markersize=1, alpha=0.3)
        axis[0].set_title(lablcomp + gdat.labltarg) 
        if gdat.listtimeplotline is not None:
            for timeplotline in gdat.listtimeplotline:
                axis[0].axvline(timeplotline, ls='--')
        strgphas = ''
        if boolphas:
            strgphas = '_phas'
        strgzoom = ''
        if boolzoom:
            axis[0].set_ylim([-1, 2.])
            #axis[0].set_ylim([np.percentile(lcurmodl, 1.), np.percentile(lcurmodl, 99.)])
            strgzoom += '_zoom'
        
        if gdat.numbrefrlcur > 0:
            axis[0].legend()

        strglimt = ''
        if indxtimelimt is not None:
            axis[a].set_xlim(gdat.listlimttimeplot[indxzoom])
            strglimt = 'zom%d_' % indxzoom

        #plt.tight_layout()
        path = retr_pathplot(gdat, a, strglimt, strgsecc, strgzoom, strgextn, strgphas, strgoffs, k, strgyaxi)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()


def retr_pathplot(gdat, a, strglimt, strgsecc, strgzoom, strgextn, strgphas, strgoffs, k, strgyaxi):
    
    if a == 0:
        pathplot = gdat.pathimagobjt + '%s_%s%s%s%s%s%s_%s_%04d.%s' % \
                            (strgyaxi, strglimt, gdat.strgsaveextn, strgsecc, strgzoom, strgextn, strgphas, strgoffs, k, gdat.strgplotextn)
    if a == 1:
        pathplot = gdat.pathimagobjt + '%srefr_%s%s%s%s%s%s_%s_%04d.%s' % \
                            (strgyaxi, strglimt, gdat.strgsaveextn, strgsecc, strgzoom, strgextn, strgphas, strgoffs, k, gdat.strgplotextn)
    
    return pathplot


def retr_cntpmodl(gdat, coef, xpos, ypos, cntpback, flux, verbtype=1):
    
    cntpmodl = np.zeros_like(gdat.cntpdatamedi[0, :, :]) + cntpback

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
        
        if False and verbtype > 0:
            print('k')
            print(k)
            print('xpos[k]')
            print(xpos[k])
            print('ypos[k]')
            print(ypos[k])
            print('coef')
            summgene(coef)
            print('cntpback')
            print(cntpback)
            print('deltxpos')
            print(deltxpos)
            print('deltypos')
            print(deltypos)
            print('psfnsour')
            print(psfnsour)
            print('flux[k]')
            print(flux[k])
            print('')

    return cntpmodl


def retr_llik(gdat, para):
    
    coef = para[:gdat.numbcoef]
    cntpback = para[gdat.numbcoef]
    
    if gdat.psfnshaptype == 'gfreffix':
        flux = gdat.cntsfitt * para[gdat.numbcoef+1]
    elif gdat.psfnshaptype == 'gfrefinf':
        flux = gdat.cntsfitt * para[gdat.numbcoef+1] * para[gdat.numbcoef+2:]
    elif gdat.psfnshaptype == 'gfreffre':
        flux = gdat.cntsfitt * para[gdat.numbcoef+1:]
    else:
        flux = para[gdat.numbcoef+1:]
    cntpmodl = retr_cntpmodl(gdat, coef, gdat.xposfitt, gdat.yposfitt, cntpback, flux)
    
    chi2 = np.sum((gdat.cntpdatamedi[0, :, :] - cntpmodl)**2 / gdat.cntpdatamedi[0, :, :])
    llik = -0.5 * chi2
    
    #if True:#verbtype > 0:
    if False:#verbtype > 0:
        print('cntpmodl')
        summgene(cntpmodl)
        print('llik')
        print(llik)
        print('gdat.cntpdatamedi[0, :, :]')
        summgene(gdat.cntpdatamedi[0, :, :])
        print('')
        print('')
        print('')
    
    return llik


def test():
    
    from astropy.io import fits
    import numpy as np
    import os
    import matplotlib.pyplot as plt





def retr_mlikflux(cntpdata, matrdesi, vari):
    
    varitdim = np.diag(vari.flatten())
    covafittflux = np.linalg.inv(np.matmul(np.matmul(matrdesi.T, np.linalg.inv(varitdim)), matrdesi))
    mlikfittflux = np.matmul(np.matmul(np.matmul(covafittflux, matrdesi.T), np.linalg.inv(varitdim)), cntpdata.flatten())
    
    return mlikfittflux, covafittflux


def main( \

         # data
         ## type of data
         datatype='obsd', \
         
         # ffim for FFI, tpxf for Target pixel file
         strgdata=None, \

         ## mock data
         truesourtype='dwar', \
         ### number of time bins
         numbtime=None, \
         ### Boolean flag
         boolcenttran=False, \
        
         ## TIC ID of the object of interest
         ticitarg=None, \
         ## RA of the object of interest
         rasctarg=None, \
         ## DEC of the object of interest
         decltarg=None, \

         ## path to feed external data
         listpathtescfile=None, \
         
         ## number of pixels on a side to cut out
         numbside=11, \
         
         ## mask
         ### Boolean flag to put a cut on quality flag
         boolcuttqual=True, \
         ### masking region
         epocmask=None, \
         perimask=None, \
         duramask=None, \

         strgbase=None, \
         
         # diagnostics
         booldiagmode=True, \

         # model
         ## factor by which to rebin the data along time
         facttimerebn=1., \
         ## maximum number stars in the fit
         maxmnumbstar=None, \
        
         ## PSF evaluation
         psfntype='lion', \
         psfnshaptype='gfrefinf', \

         catlextr=None, \
         lablcatlextr=None, \
    
         # Boolean flag to repeat the fit, putting the target to offset locations
         boolfittoffs=False, \

         ## post-process
         ## baseline-detrending
         boolbdtr=True, \
         bdtrtype='spln', \
         weigsplnbdtr=1., \
         booladdddiscbdtr=False, \
         durakernbdtrmedi=1., \
         ## optional BLS
         booltlss=False, \
         
         ## optinal phase-folding
         ### epoch
         epoc=None, \
         ### period
         peri=None, \
        
         listpixlaper=None, \

         # plotting
         ## Boolean flag to plot the quaternions
         boolplotquat=False, \

         ## animations
         ### Boolean flag to make an animation
         boolmakeanim=True, \

         ### Boolean flag to include all time bins in the animation
         boolplotframtotl=False, \
         
         # plot extensions
         strgplotextn='png', \

         # epoch for correcting the RA and DEC for proper motion
         epocpmot=None, \
         ## RA proper motion, used when RA and DEC are provided
         pmratarg=None, \
         ## DEC proper motion, used when RA and DEC are provided
         pmdetarg=None, \
         
         ## time limits for a zoom-in plot
         listlimttimeplot=None, \

         ## the time to indicate on the plots with a vertical line
         listtimeplotline=None, \
         
         # a string to be used to search MAST for the target
         strgmast=None, \

         # a string that will appear in the plots to label the target, which can be anything the user wants
         labltarg=None, \
         
         # a string that will be used to name output files for this target
         strgtarg=None, \
        
         # image color scale
         cntpscaltype='asnh', \
         
        ):
   
    # construct global object
    gdat = tdpy.util.gdatstrt()
    
    # copy unnamed inputs to the global object
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print('TCAT initialized at %s...' % gdat.strgtimestmp)
   
    # determine the target
    if gdat.ticitarg is not None and (rasctarg is not None or decltarg is not None) or \
                gdat.ticitarg is not None and gdat.strgmast is not None or \
                (rasctarg is not None or decltarg is not None) and gdat.strgmast is not None:
        raise Exception('Either a TIC ID (ticitarg) or RA&DEC (rasctarg and decltarg) or a string to be searched on MAST (strgmast) \
                                                                                                         should be provided; not more than one of these.')
    if gdat.ticitarg is None and (rasctarg is None or decltarg is None) and gdat.strgmast is None:
        raise Exception('Either a TIC ID (ticitarg) or RA&DEC (rasctarg and decltarg) or a string to be searched on MAST (strgmast) should be provided.')
    if gdat.ticitarg is not None:
        strgtargtype = 'tici'
        print('A TIC ID is provided as input.')
        catalogData = astroquery.mast.Catalogs.query_object('TIC ' + str(gdat.ticitarg), radius='200s', catalog="TIC")
        rasctarg = catalogData[0]['ra']
        decltarg = catalogData[0]['dec']
        print('rasctarg')
        print(rasctarg)
        print('decltarg')
        print(decltarg)
        print('TIC ID: %d' % gdat.ticitarg)
    else:
        if rasctarg is not None and decltarg is not None:
            strgtargtype = 'posi'
            print('RA and DEC are provided as input.')
            catalogData = astroquery.mast.Catalogs.query_region('%g %g' % (rasctarg, decltarg), radius='200s', catalog="TIC")
        else:
            strgtargtype = 'mast'
            print('A MAST keyword is provided as input: %s' % gdat.strgmast)
            catalogData = astroquery.mast.Catalogs.query_region(gdat.strgmast, radius='200s', catalog="TIC")
            rasctarg = catalogData[0]['ra']
            decltarg = catalogData[0]['dec']
            gdat.ticitarg = int(catalogData[0]['ID'])
    
    if gdat.labltarg is None:
        if gdat.strgmast is not None:
            gdat.labltarg = gdat.strgmast
        else:
            gdat.labltarg = 'TIC %d' % gdat.ticitarg
    if gdat.strgtarg is None:
        if gdat.strgmast is not None:
            gdat.strgtarg = gdat.strgmast
        else:
            gdat.strgtarg = 'TIC%d' % gdat.ticitarg
    print('Object label: %s' % gdat.labltarg) 
    print('Output folder name: %s' % gdat.strgtarg) 
    print('RA and DEC: %g %g' % (rasctarg, decltarg))
   
    print('PSF model:')
    print(gdat.psfntype)
    if gdat.psfntype == 'ontf':
        print('PSF model shape:')
        print(gdat.psfnshaptype)
    # paths
    gdat.pathbase = os.environ['TCAT_DATA_PATH'] + '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathimag = gdat.pathbase + 'imag/'
    if gdat.strgbase is None:
        gdat.pathobjt = gdat.pathbase + '%s/' % gdat.strgtarg
    else:
        gdat.pathobjt = gdat.pathbase + '%s/%s/' % (gdat.strgbase, gdat.strgtarg)
    gdat.pathdataobjt = gdat.pathobjt + 'data/'
    gdat.pathimagobjt = gdat.pathobjt + 'imag/'
    gdat.pathimagobjtpsfn = gdat.pathimagobjt + 'psfn/'
    os.system('mkdir -p %s' % gdat.pathdata)
    os.system('mkdir -p %s' % gdat.pathimag)
    os.system('mkdir -p %s' % gdat.pathdataobjt)
    os.system('mkdir -p %s' % gdat.pathimagobjt)
    os.system('mkdir -p %s' % gdat.pathimagobjtpsfn)
   
    # fix the seed
    np.random.seed(0)
    
    gdat.pathdatasumm = gdat.pathbase + '%s/summary/' % gdat.strgbase
    os.system('mkdir -p %s' % gdat.pathdatasumm) 
    
    # exposure time
    gdat.timeexpo = 1440. # [sec]
    
    pathbasedown = gdat.pathdataobjt + 'mastDownload/TESS/'
    if gdat.strgdata is None:
        if gdat.ticitarg is None:
            gdat.strgdata = 'ffim'
        else:
            if not os.path.exists(pathbasedown):
                print('Trying to download the SPOC data...')
                pathdown = tesstarg.util.down_spoclcur(gdat.pathdataobjt, strgmast='TIC' + str(gdat.ticitarg), boollcuronly=False)
                if len(pathdown) == 0:
                    gdat.strgdata = 'ffim'
                else:
                    gdat.strgdata = 'tpxf'
            else:
                print('SPOC data already downloaded at %s. Will read the available data...' % pathbasedown)
                gdat.strgdata = 'tpxf'
    
    print('gdat.strgdata')
    print(gdat.strgdata)
    gdat.strgcnfg = '%s_%04d_%04d_%s_%s' % (gdat.datatype, gdat.numbside, gdat.facttimerebn, gdat.strgtarg, gdat.strgdata)
    
    if gdat.strgdata == 'tpxf' and gdat.numbside != 11:
        raise Exception('numbside should be 11 for target pixel file (TPF) data.')

    print('Reading the data...')
    if gdat.datatype != 'mock':
        
        # get data
        if gdat.strgdata == 'ffim':
            cutout_coord = astropy.coordinates.SkyCoord(rasctarg, decltarg, unit="deg")
            if gdat.listpathtescfile is not None:
                listhdundata = []
                for pathtescfile in gdat.listpathtescfile:
                    listhdundata.append(astropy.io.fits.open(pathtescfile))
            else:
                try:
                    listhdundata = astroquery.mast.Tesscut.get_cutouts(cutout_coord, gdat.numbside)
                except:
                    print('TESSCut failed to get data.')
                    print('Returning 1...')
                    print
                    return 1
        else:
            listfilespoc = fnmatch.filter(os.listdir(gdat.pathdataobjt + 'mastDownload/TESS/'), 'tess*-s*-%016d-*-s' % gdat.ticitarg)
            # ensure sectors are analyzed and concatenated in time-order
            listfilespoc.sort()

            listhdundata = []
            for k, pathtemp in enumerate(listfilespoc): 
                path = gdat.pathdataobjt + 'mastDownload/TESS/%s/%s_tp.fits' % (pathtemp, pathtemp)
                listhdundata.append(astropy.io.fits.open(path))
            
        gdat.listisec = []
        gdat.listicam = []
        gdat.listiccd = []
        listobjtwcss = []
        for hdundata in listhdundata:
            gdat.listisec.append(hdundata[0].header['SECTOR'])
            gdat.listicam.append(hdundata[0].header['CAMERA'])
            gdat.listiccd.append(hdundata[0].header['CCD'])
            listobjtwcss.append(astropy.wcs.WCS(hdundata[2].header))

        gdat.numbsect = len(listhdundata)
    
    gdat.listmeta = [gdat.listisec, gdat.listicam, gdat.listiccd]

    print('Found %d sectors of data.' % gdat.numbsect)
    gdat.indxsect = np.arange(gdat.numbsect)
    
    # get reference light curve
    gdat.lablrefrlcur = []
    gdat.colrrefrlcur = []
    if gdat.strgdata == 'tpxf':
        gdat.lablrefrlcur += ['SPOC']
        gdat.colrrefrlcur = ['r']
    gdat.numbrefrlcur = len(gdat.lablrefrlcur)
    gdat.indxrefrlcur = np.arange(gdat.numbrefrlcur)
    
    # get reference catalog
    gdat.lablrefrcatl = ['TIC']
    if gdat.catlextr is not None:
        gdat.lablrefrcatl.extend(gdat.lablcatlextr)

    gdat.numbrefrcatl = len(gdat.lablrefrcatl)
    gdat.indxrefrcatl = np.arange(gdat.numbrefrcatl)
    
    # check for an earlier TCAT run
    gdat.boolskip = True
    for o in gdat.indxsect:
        strgsecc = '%02d%d%d' % (gdat.listisec[o], gdat.listicam[o], gdat.listiccd[o])
        strgrflxextn = '_%s_%s_%s.csv' % (gdat.strgcnfg, strgsecc, 'of11')
        pathsavefluxtarg = gdat.pathdataobjt + 'fluxtarg' + strgrflxextn
        if not os.path.exists(pathsavefluxtarg):
            gdat.boolskip = False
    if False and gdat.boolskip:
        print('Run previously completed...')
        print
        gdat.listarry = np.loadtxt(pathsavefluxtarg)
        return gdat.listarry, gdat.listmeta

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
        gdat.gdatlion.indxtime = np.arange(1)
        gdat.gdatlion.numbtime = 1
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

    print('Searching the TIC catalog for nearby sources...')
    catlrefrskyy = np.empty((catalogData[:]['ra'].size, 2))
    catlrefrskyy[:, 0] = catalogData[:]['ra']
    catlrefrskyy[:, 1] = catalogData[:]['dec']
    
    gdat.catlrefr = [{}]
    if gdat.catlextr is not None:
        gdat.numbcatlextr = len(gdat.catlextr)
        gdat.indxcatlextr = np.arange(gdat.numbcatlextr)
        for k in gdat.indxcatlextr:
            gdat.catlrefr.append({})
    
    print('Including the nearby TIC sources to the fitting catalog...')
    gdat.catlrefr[0]['tmag'] = catalogData[:]['Tmag']
    indxcatlsortbrgt = np.argsort(gdat.catlrefr[0]['tmag'])
    gdat.catlrefr[0]['rasc'] = catalogData[:]['ra'][indxcatlsortbrgt]
    gdat.catlrefr[0]['decl'] = catalogData[:]['dec'][indxcatlsortbrgt]
    
    gdat.catlrefr[0]['tici'] = np.empty(len(catalogData), dtype=int)
    gdat.catlrefr[0]['tici'][:] = catalogData[:]['ID'][indxcatlsortbrgt]
    gdat.catlrefr[0]['pmde'] = catalogData[:]['pmDEC'][indxcatlsortbrgt]
    gdat.catlrefr[0]['pmra'] = catalogData[:]['pmRA'][indxcatlsortbrgt]
    
    print('Number of sources in the reference catalog: %d' % len(gdat.catlrefr[0]['rasc']))
    if gdat.rasctarg is not None:
        maxmdmag = 5.
        print('Filtering nearbry sources based on delta magnitude (maximum delta magnitude of %g)' % maxmdmag)
        gdat.indxrefrbrgt = np.where(gdat.catlrefr[0]['tmag'] - gdat.catlrefr[0]['tmag'][0] < maxmdmag)[0]
    else:
        gdat.indxrefrbrgt = np.where(gdat.catlrefr[0]['tmag'] < 18.)[0]
    
    gdat.numbrefrbrgt = gdat.indxrefrbrgt.size

    print('%d of the reference catalog sources are brighter than the magnitude cutoff.' % gdat.numbrefrbrgt)
    gdat.rasccatl = gdat.catlrefr[0]['rasc'][gdat.indxrefrbrgt]
    gdat.declcatl = gdat.catlrefr[0]['decl'][gdat.indxrefrbrgt]
    gdat.tmagcatl = gdat.catlrefr[0]['tmag'][gdat.indxrefrbrgt]
    
    gdat.cntscatl = 10**(-(gdat.tmagcatl - 20.4) / 2.5)

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
        gdat.rasccatl += pmra * (gdat.epocpmot - 2000.) / (1000. * 3600.)
        gdat.declcatl += pmde * (gdat.epocpmot - 2000.) / (1000. * 3600.)
    
    if gdat.rasctarg is not None:
        gdat.rascfitt = np.empty(1 + len(gdat.rasccatl))
        gdat.declfitt = np.empty(1 + len(gdat.rasccatl))
        gdat.rascfitt[0] = gdat.rasctarg
        gdat.declfitt[0] = gdat.decltarg
        gdat.rascfitt[1:] = gdat.rasccatl
        gdat.declfitt[1:] = gdat.declcatl
        #np.concatenate((np.array([gdat.rasctarg]), gdat.rasccatl))
        #gdat.declfitt = np.concatenate((np.array([gdat.decltarg]), gdat.declcatl))
        if gdat.psfntype == 'ontf':
            gdat.cntsfitt = np.concatenate((np.array([0.]), gdat.cntscatl))
            print('Do not know what counts to assume for the target (for the on-the-fly PSF fit). Assuming 0!')
    else:
        gdat.rascfitt = gdat.rasccatl
        gdat.declfitt = gdat.declcatl
        if gdat.psfntype == 'ontf':
            gdat.cntsfitt = gdat.cntscatl
    
    print('gdat.rascfitt')
    summgene(gdat.rascfitt)
    gdat.refrtime = [[] for k in gdat.indxrefrlcur]
    gdat.refrrflx = [[] for k in gdat.indxrefrlcur]
    gdat.stdvrefrrflx = [[] for k in gdat.indxrefrlcur]
    gdat.boolrefr = False
    if gdat.strgdata == 'tpxf':
        for k, pathtemp in enumerate(listfilespoc): 
            path = gdat.pathdataobjt + 'mastDownload/TESS/%s/%s_lc.fits' % (pathtemp, pathtemp)
            arry, indxtimequalgood, indxtimenanngood, isec, icam, iccd = tesstarg.util.read_tesskplr_file(path, strgtype='PDCSAP_FLUX')
            gdat.refrtime[0].append(arry[:, 0])
            gdat.refrrflx[0].append(arry[:, 1])
            gdat.stdvrefrrflx[0].append(arry[:, 2])
        gdat.boolrefr = True

    print('Looping over sectors...')
    
    gdat.timedata = [[] for o in gdat.indxsect]
    
    gdat.listarry = [[] for o in gdat.indxsect]
    gdat.numbtime = np.empty(gdat.numbsect, dtype=int)
    for o in gdat.indxsect:
        strgsecc = '%02d%d%d' % (gdat.listisec[o], gdat.listicam[o], gdat.listiccd[o])
        print('Sector: %d' % gdat.listisec[o])
        print('Camera: %d' % gdat.listicam[o])
        print('CCD: %d' % gdat.listiccd[o])
        print('strgsecc')
        print(strgsecc)
        
        gdat.strgtitlcntpplot = '%s, Sector %d, Cam %d, CCD %d' % (gdat.labltarg, gdat.listisec[o], gdat.listicam[o], gdat.listiccd[o])
        
        if gdat.boolplotquat:
            print('Reading quaternions...')
            pathquat = gdat.pathdata + 'quat/tess2018330083923_sector%02d-quat.fits' % gdat.listisec[o]
            listhdun = astropy.io.fits.open(pathquat)
            listhdun.info()
            dataquat = listhdun[0].data
            print('dataquat')
            summgene(dataquat)
            print

        # get data
        if gdat.datatype == 'mock':
            gdat.listtime = np.linspace(0, gdat.numbtime - 1, gdat.numbtime)
            indxtimedatagood = np.arange(gdat.numbtime)
        else:
            # read the FITS files
            ## time
            gdat.listtime = listhdundata[o][1].data['TIME']
            ## count per pixel
            gdat.cntpdata = gdat.timeexpo * (listhdundata[o][1].data['FLUX'] + \
                                             listhdundata[o][1].data['FLUX_BKG']).swapaxes(0, 2).swapaxes(0, 1)[None, :, :, :]
            
            if gdat.strgdata == 'tpxf':
                gdat.numbside = gdat.cntpdata.shape[1]
            ## filter good times
            
            #indxtimedatagood = np.isfinite(np.mean(gdat.cntpdata, axis=(0, 1, 2))) & np.isfinite(gdat.listtime)
            indxtimedatagood = np.isfinite(gdat.listtime)
            if gdat.boolcuttqual:
                indxtimedatagood = indxtimedatagood & (listhdundata[o][1].data['QUALITY'] == 0)
            indxtimedatagood = np.where(indxtimedatagood)[0]
            
            if indxtimedatagood.size == 0:
                print('No good data found.')
                print('Returning 2...')
                print
                return 2, None
    
        print('gdat.cntpdata')
        summgene(gdat.cntpdata)
        
        gdat.listtime = gdat.listtime[indxtimedatagood]
        gdat.cntpdata = gdat.cntpdata[:, :, :, indxtimedatagood]
        
        gdat.numbtimecutt = gdat.listtime.size
        
        gdat.listtime = gdat.listtime[:gdat.numbtimecutt]
        gdat.cntpdata = gdat.cntpdata[:, :, :, :gdat.numbtimecutt]
       
        gdat.numbtime[o] = gdat.listtime.size
        gdat.indxtime = np.arange(gdat.numbtime[o])

        if gdat.listpixlaper is not None:
            gdat.cntpaper = np.zeros(gdta.numbtime[o])
            for pixlaper in gdat.listpixlaper:
                gdat.cntpaper += gdgat.cntpdata[0, pixlaper[0], pixlaper[1], :]
            gdat.cntpaper /= np.median(gdat.cntpaper)
        
        gdat.cntpdatamedi = np.median(gdat.cntpdata, 3)

        # make mock data
        if gdat.datatype == 'mock':
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
            cntpbacktrue = np.zeros((1, gdat.numbside, gdat.numbside, gdat.numbtime[o])) + 1800 * 100.
            if gdat.psfntype == 'lion':
                cntpdatatemp = lionmain.eval_modl(gdat.gdatlion, xpostrue, ypostrue, fluxtrue, cntpbacktrue)
            
            gdat.cntpdata = np.random.poisson(cntpdatatemp).astype(float)
        
        # filter input time series
        gdat.indxtimefitt = np.arange(gdat.cntpdata.shape[3])
        gdat.cntpdata = gdat.cntpdata[:, :, :, gdat.indxtimefitt]
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

        if gdat.datatype != 'mock':
            objttimedata = astropy.time.Time(gdat.listtime + 2457000, format='jd', scale='utc')
            gdat.timedata[o] = objttimedata.jd
        else:
            gdat.timedata[o] = gdat.listtime
            gdat.refrtime[o] = gdat.listtime
        
        if not np.isfinite(gdat.cntpdata).all():
            print('Warning. Not all data are finite.')
            print('Returning 3...')
            print
            return 3, None
        
        if gdat.catlextr is not None:
            for qq in gdat.indxcatlextr:
                gdat.catlrefr[qq+1]['rasc'] = gdat.catlextr[qq]['rasc']
                gdat.catlrefr[qq+1]['decl'] = gdat.catlextr[qq]['decl']

        ## reference catalogs
        for q in gdat.indxrefrcatl:
            if gdat.datatype == 'mock':
                gdat.catlrefr[q]['xpos'] = xpostrue
                gdat.catlrefr[q]['ypos'] = ypostrue
            else:
                catlrefrposi = listobjtwcss[o].all_world2pix(catlrefrskyy, 0)
                gdat.catlrefr[q]['xpos'] = catlrefrposi[:, 0]
                gdat.catlrefr[q]['ypos'] = catlrefrposi[:, 1]
                
            gdat.catlrefr[q]['flux'] = gdat.catlrefr[q]['xpos'] * 0

            ## filter the reference catalog
            indxsourgood = (gdat.catlrefr[q]['xpos'] > -0.5) & (gdat.catlrefr[q]['xpos'] < gdat.numbside - 0.5) & \
                           (gdat.catlrefr[q]['ypos'] > -0.5) & (gdat.catlrefr[q]['ypos'] < gdat.numbside - 0.5)
            gdat.catlrefr[q]['xpos'] = gdat.catlrefr[q]['xpos'][indxsourgood]
            gdat.catlrefr[q]['ypos'] = gdat.catlrefr[q]['ypos'][indxsourgood]
            gdat.catlrefr[q]['tmag'] = gdat.catlrefr[q]['tmag'][indxsourgood]
        
        if gdat.facttimerebn != 1:
            listcadetemp = np.empty(indxtimerebn.size - 1)
            cntpdatatemp = np.empty((1, gdat.numbside, gdat.numbside, indxtimerebn.size - 1))
            for t in range(len(indxtimerebn) - 1):
                temp = gdat.cntpdata[0, :, :, indxtimerebn[t]:indxtimerebn[t+1]]
                cntpdatatemp[0, :, :, t] = np.sum(temp, axis=2)
                listcadetemp[t] = gdat.timeexpo * temp.shape[2]
            gdat.listcade = np.copy(listcadetemp)
            gdat.cntpdata = np.copy(cntpdatatemp)
        
        if np.amin(gdat.cntpdata) < 0:
            print('Minimum of the image is negative.')

        # data weights
        gdat.vari = np.zeros_like(gdat.cntpdata) + np.mean(gdat.cntpdata)

        # contruct a reference catalog
            
        # transform sky coordinates into dedector coordinates and filter
        ## fitting catalog
        print('gdat.rascfitt')
        summgene(gdat.rascfitt)
        skyyfitttemp = np.empty((gdat.rascfitt.size, 2))
        skyyfitttemp[:, 0] = gdat.rascfitt
        skyyfitttemp[:, 1] = gdat.declfitt
        if gdat.rascfitt.size == 0:
            raise Exception('')
        posifitttemp = listobjtwcss[o].all_world2pix(skyyfitttemp, 0)
        gdat.xposfitt = posifitttemp[:, 0]
        gdat.yposfitt = posifitttemp[:, 1]
        indxsourgood = (gdat.xposfitt > -0.5) & (gdat.xposfitt < gdat.numbside - 0.5) & \
                       (gdat.yposfitt > -0.5) & (gdat.yposfitt < gdat.numbside - 0.5)
        gdat.xposfitt = gdat.xposfitt[indxsourgood]
        gdat.yposfitt = gdat.yposfitt[indxsourgood]
        if gdat.psfntype == 'ontf':
            gdat.cntsfitt = gdat.cntsfitt[indxsourgood]

        print('Number of sources in the cut-out: %d' % gdat.xposfitt.size)
         
        if gdat.xposfitt.size == 0:
            print('No fitting source found...')
            print('Returning 4...')
            print
            return 4, None
        
        numbtemp = gdat.xposfitt.size
        
        if gdat.rascfitt.size == 0:
            raise Exception('')
        
        # filter the fitting sources
        if gdat.maxmnumbstar is not None:
            print('Taking the first %d sources among the neighbors.' % gdat.maxmnumbstar)
            gdat.xposfitt = gdat.xposfitt[:gdat.maxmnumbstar]
            gdat.yposfitt = gdat.yposfitt[:gdat.maxmnumbstar]
        
        gdat.numbstar = gdat.xposfitt.size
        gdat.indxstar = np.arange(gdat.numbstar)
            
        print('Number of point sources to fit: %d' % gdat.numbstar)
    
        print('gdat.numbstar')
        print(gdat.numbstar)

        gdat.numbpixl = gdat.numbside**2
        
        # plotting settings
        gdat.vmaxcntpdata = np.percentile(gdat.cntpdata, 100)
        gdat.vmincntpdata = np.percentile(gdat.cntpdata, 0)
    
        gdat.sizeplotsour = 10
        
        if not np.isfinite(gdat.cntpdata).all():
            print('There is NaN in the data!')
        

        gdat.vmaxcntpresi = 100. * np.sqrt(np.percentile(gdat.cntpdata[np.where(np.isfinite(gdat.cntpdata))], 100))
        
        print('gdat.vmaxcntpresi')
        print(gdat.vmaxcntpresi)
        if gdat.cntpscaltype == 'asnh':
            gdat.vmincntpdata = np.arcsinh(gdat.vmincntpdata)
            gdat.vmaxcntpdata = np.arcsinh(gdat.vmaxcntpdata)
            gdat.vmaxcntpresi = np.arcsinh(gdat.vmaxcntpresi)
        gdat.vmincntpresi = -gdat.vmaxcntpresi

        if not np.isfinite(gdat.vmincntpresi):
            print('Minimum of the residual counts is not finite.')
            raise Exception('')

        gdat.numbener = 1
        gdat.indxener = np.arange(gdat.numbener)
        
        gdat.boolbackoffs = True
        gdat.boolposioffs = False
            
        if gdat.ticitarg is None:
            gdat.strgsaveextn = '%s_%04d_%s_%s' % \
                                            (gdat.datatype, gdat.numbside, gdat.strgtarg, gdat.strgdata)
        else:
            gdat.strgsaveextn = '%s_%04d_%016d_%s' % \
                                            (gdat.datatype, gdat.numbside, gdat.ticitarg, gdat.strgdata)
   
        # plot a histogram of data counts
        figr, axis = plt.subplots(figsize=(12, 4))
        bins = np.logspace(np.log10(np.amin(gdat.cntpdata)), np.log10(np.amax(gdat.cntpdata)), 200)
        axis.hist(gdat.cntpdata.flatten(), bins=bins)
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_ylabel('N')
        axis.set_xlabel('C [e$^{-}$]')
        plt.tight_layout()
        path = gdat.pathimagobjt + 'histcntpdata_%s.%s' % (gdat.strgsaveextn, gdat.strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        gdat.cntpback = np.zeros_like(gdat.cntpdata)

        if np.amin(gdat.cntpdata) < 1.:
            print('Minimum of the data is not positive.')

        if not np.isfinite(gdat.cntpback).all():
            raise Exception('')
        
        if gdat.cntpdata.shape[1] != gdat.numbside:
            raise Exception('')

        if gdat.cntpdata.shape[2] != gdat.numbside:
            raise Exception('')

        if gdat.psfntype == 'lion':
            gdat.gdatlion.numbener = 1
            gdat.gdatlion.numbtime = 1
            gdat.gdatlion.indxtime = [0]
        
        # number of components
        gdat.numbcomp = gdat.numbstar + 1
        gdat.indxcomp = np.arange(gdat.numbcomp)

        gdat.stdvfittflux = np.empty((gdat.numbtime[o], gdat.numbcomp))
        
        #print('Evaluating the PSFs for contamination ratio...')
        #numbstarneig = 10
        #indxstarneig = np.arange(numbstarneig)
        #
        #for k in indxstarneig:
        #    matrdesi[:, k] = lionmain.eval_modl(gdat.gdatlion, gdat.xposfitt[k, None], gdat.yposfitt[k, None], \
        #                                                    gdat.fluxdumm[None, :, k, None], cntpback)[0, :, :, 0].flatten()
                

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
            path = '%scntpdatamedi.%s' % (gdat.pathimagobjtpsfn, gdat.strgplotextn)
            strgtitl = gdat.strgtitlcntpplot
            plot_cntp(gdat, gdat.cntpdatamedi[0, :, :], path, o, strgtitl=strgtitl)

            listlablpara = []
            for k in indxcoef:
                listlablpara.append(['$c_{%d}$' % k, ''])
            listlablpara.append(['B', 'e$^{-}$'])
            
            arrytemp = np.linspace(0., float(gdat.numbside - 1), gdat.numbside)
            gdat.xpos, gdat.ypos = np.meshgrid(arrytemp, arrytemp)
            
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
            
            print('gdat.indxstar')
            summgene(gdat.indxstar)
            print('gdat.numbparapsfn')
            print(gdat.numbparapsfn)
            print('gdat.cntsfitt')
            summgene(gdat.cntsfitt)
            print('')
            if gdat.psfnshaptype == 'gaus' or gdat.psfntype == 'gfre' or gdat.psfntype == 'pfre':
                # solve for the maximum likelihood fluxes for the median image
                matrdesi = np.ones((gdat.numbpixl, gdat.numbstar + 1))
                for k in np.arange(gdat.numbstar):
                    matrdesi[:, k] = retr_cntpmodl(gdat, coef, gdat.xposfitt[k, None], gdat.yposfitt[k, None], np.array([0.]), np.array([1.])).flatten()
                    print('matrdesi[:, k]')
                    summgene(matrdesi[:, k])
                    print('')
                gdat.mlikfittfluxmedi, gdat.covafittfluxmedi = retr_mlikflux(gdat.cntpdatamedi[0, :, :], matrdesi, gdat.cntpdatamedi[0, :, :])
            
            # background
            listminmpara[gdat.numbcoef] = 0.
            listmaxmpara[gdat.numbcoef] = np.amax(gdat.cntpdatamedi)#abs(gdat.mlikfittfluxmedi[-1]) * 2.
            
            if gdat.psfnshaptype == 'gfreffix' or gdat.psfntype =='gfrefinf':
                listminmpara[gdat.numbcoef+1] = 0.
                listmaxmpara[gdat.numbcoef+1] = 10000.
                if gdat.psfnshaptype == 'gfrefinf':
                    listminmpara[gdat.numbcoef+2:] = 0.
                    listmaxmpara[gdat.numbcoef+2:] = 10.
            elif gdat.psfnshaptype == 'gfreffre':
                listminmpara[gdat.numbcoef+1:] = 0.
                listmaxmpara[gdat.numbcoef+1:] = np.amax(gdat.cntpdatamedi)#abs(gdat.mlikfittfluxmedi[:-1]) * 2.
            else:
                listminmpara[gdat.numbcoef+1:] = 0.
                listmaxmpara[gdat.numbcoef+1:] = np.amax(gdat.cntpdatamedi)#abs(gdat.mlikfittfluxmedi[:-1]) * 2.
            
            strgextn = 'psfn'
            numbdata = gdat.numbpixl
            strgsaveextn = gdat.pathdataobjt + gdat.psfnshaptype + '_' + gdat.strgsaveextn + '.txt'
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
                
            # plot the posterior median PSF model
            xpossour = np.array([(gdat.numbside - 1.) / 2.])
            ypossour = np.array([(gdat.numbside - 1.) / 2.])
            flux = np.array([1.])
            gdat.cntpmodlpsfn = retr_cntpmodl(gdat, coefmedi, xpossour, ypossour, 0., flux, verbtype=1)[None, :, :, None]
            path = '%scntpmodlpsfnmedi.%s' % (gdat.pathimagobjtpsfn, gdat.strgplotextn)
            strgtitl = gdat.strgtitlcntpplot
            plot_cntp(gdat, gdat.cntpmodlpsfn[0, :, :, 0], path, o, strgtitl=strgtitl, boolanno=False)

            # plot the posterior median image model
            if gdat.psfnshaptype == 'gfreffix':
                flux = gdat.cntsfitt * amplmedi
            elif gdat.psfnshaptype == 'gfrefinf':
                flux = gdat.cntsfitt * ampltotlmedi * amplrelamedi
            elif gdat.psfnshaptype == 'gfreffre':
                flux = fluxmedi
            gdat.cntpmodlpsfn = retr_cntpmodl(gdat, coefmedi, gdat.xposfitt, gdat.yposfitt, cntpbackmedi, flux)[None, :, :, None]
            path = '%scntpmodlmedi.%s' % (gdat.pathimagobjtpsfn, gdat.strgplotextn)
            strgtitl = gdat.strgtitlcntpplot
            plot_cntp(gdat, gdat.cntpmodlpsfn[0, :, :, 0], path, o, strgtitl=strgtitl)

            # plot the posterior median residual
            path = '%scntpresimedi.%s' % (gdat.pathimagobjtpsfn, gdat.strgplotextn)
            strgtitl = gdat.strgtitlcntpplot
            plot_cntp(gdat, gdat.cntpdatamedi[0, :, :] - gdat.cntpmodlpsfn[0, :, :, 0], path, o, strgtitl=strgtitl, boolresi=True)

        gdat.offs = 0.5
        gdat.listoffsxpos = np.linspace(-gdat.offs, gdat.offs, 3)
        gdat.listoffsypos = np.linspace(-gdat.offs, gdat.offs, 3)
        gdat.numboffs = gdat.listoffsxpos.size
        gdat.indxoffs = np.arange(gdat.numboffs)
        for x in gdat.indxoffs:
            for y in gdat.indxoffs:
                
                if not gdat.boolfittoffs and (x != 1 or y != 1):
                    continue
                
                strgoffs = 'of%d%d' % (x, y)
                strgrflxextn = '_%s_%s_%s.csv' % (gdat.strgcnfg, strgsecc, strgoffs)
                pathsaveflux = gdat.pathdataobjt + 'flux' + strgrflxextn
                pathsavefluxtarg = gdat.pathdataobjt + 'fluxtarg' + strgrflxextn
                pathsavefluxtargbdtr = gdat.pathdataobjt + 'fluxtargbdtr' + strgrflxextn
                if True or not os.path.exists(pathsaveflux):
                    
                    # introduce the positional offset
                    xpostemp = np.copy(gdat.xposfitt)
                    ypostemp = np.copy(gdat.yposfitt)
                    xpostemp[0] = gdat.xposfitt[0] + gdat.listoffsxpos[x]
                    ypostemp[0] = gdat.yposfitt[0] + gdat.listoffsypos[y]
                
                    gdat.covafittflux = np.empty((gdat.numbtime[o], gdat.numbstar + 1, gdat.numbstar + 1))
                    gdat.mlikfittflux = np.empty((gdat.numbtime[o], gdat.numbstar + 1))
                    gdat.fluxdumm = np.ones((gdat.numbtime[o], gdat.numbstar))
                    matrdesi = np.ones((gdat.numbpixl, gdat.numbstar + 1))

                    print('Evaluating the PSFs to create templates for the design matrix...')
                    for k in np.arange(gdat.numbstar):
                        if gdat.psfntype == 'lion':
                            matrdesi[:, k] = lionmain.eval_modl(gdat.gdatlion, xpostemp[k, None], ypostemp[k, None], \
                                                                        gdat.fluxdumm[None, :, k, None], gdat.cntpback)[0, :, :, 0].flatten()
                        else:
                            matrdesi[:, k] = retr_cntpmodl(gdat, coefmedi, xpostemp[k, None], ypostemp[k, None], np.array([0.]), np.array([1.])).flatten()
                            
                    # solve the linear system
                    print('Solving the linear systems of equations...')
                    timeinit = timemodu.time()
                    for t in gdat.indxtime:
                        gdat.mlikfittflux[t, :], gdat.covafittflux[t, :, :] = retr_mlikflux(gdat.cntpdata[0, :, :, t], matrdesi, gdat.vari[0, :, :, t])

                        #varitdim = np.diag(gdat.vari[0, :, :, t].flatten())
                        #gdat.covafittflux[t, :, :] = np.linalg.inv(np.matmul(np.matmul(matrdesi.T, np.linalg.inv(varitdim)), matrdesi))
                        #gdat.mlikfittflux[t, :] = np.matmul(np.matmul(np.matmul(gdat.covafittflux[t, :, :], matrdesi.T), \
                        #                                                             np.linalg.inv(varitdim)), gdat.cntpdata[0, :, :, t].flatten())

                    timefinl = timemodu.time()
                    print('Done in %g seconds.' % (timefinl - timeinit))
                    
                    if gdat.boolmakeanim:
                        cntpbackneww = gdat.mlikfittflux[:, -1][None, None, None, :]
                        print('Evaluating the model at all time bins...')
                        timeinit = timemodu.time()
                        if gdat.psfntype == 'lion':
                            gdat.gdatlion.numbtime = gdat.numbtime[o]
                            gdat.gdatlion.indxtime = gdat.indxtime
                            cntpmodl = lionmain.eval_modl(gdat.gdatlion, xpostemp, ypostemp, gdat.mlikfittflux[None, :, :-1], cntpbackneww)
                        else:
                            cntpmodl = np.empty_like(gdat.cntpdata)
                            for t in gdat.indxtime:
                                cntpmodl[0, :, :, t] = retr_cntpmodl(gdat, coefmedi, xpostemp, ypostemp, \
                                                                                        cntpbackneww[0, 0, 0, t], gdat.mlikfittflux[t, :-1])
                        timefinl = timemodu.time()
                        print('Done in %g seconds.' % (timefinl - timeinit))
                    
                        gdat.vmaxcntpmodl = np.percentile(cntpmodl, 100)
                        gdat.vmincntpmodl = np.percentile(cntpmodl, 0)
                        if gdat.cntpscaltype == 'asnh':
                            gdat.vmaxcntpmodl = np.arcsinh(gdat.vmaxcntpmodl)
                            gdat.vmincntpmodl = np.arcsinh(gdat.vmincntpmodl)
                        
                        cntpresi = gdat.cntpdata - cntpmodl
                        chi2 = np.mean(cntpresi**2 / gdat.cntpdata) + 2 * gdat.numbstar
                        
                    for k in gdat.indxcomp:
                        gdat.stdvfittflux[:, k] = np.sqrt(gdat.covafittflux[:, k, k])
                    
                    print('Normalizing by the median flux...')
                    gdat.medifittflux = np.median(gdat.mlikfittflux, 0)
                    print('Median flux of the central source is %g ADU.' % gdat.medifittflux[0])
                    
                    # normalize fluxes to get relative fluxes
                    gdat.mlikfittrflx = gdat.mlikfittflux / gdat.medifittflux[None, :]
                    gdat.stdvfittrflx = gdat.stdvfittflux / gdat.medifittflux[None, :]
                    if gdat.medifittflux[0] < 0:
                        print('MEDIAN FLUX OF THE TARGET IS NEGATIVE.')
                        gdat.mlikfittrflx = 2 - gdat.mlikfittrflx
                    
                    if gdat.booldiagmode:
                        for a in range(gdat.timedata[o].size):
                            if a != gdat.timedata[o].size - 1 and gdat.timedata[o][a] >= gdat.timedata[o][a+1]:
                                raise Exception('')

                    # write the light curve to file
                    arry = np.empty((gdat.numbtime[o], 2*gdat.numbcomp+1))
                    arry[:, 0] = gdat.timedata[o]
                    for k in gdat.indxcomp:
                        arry[:, 2*k+1] = gdat.mlikfittrflx[:, k]
                        arry[:, 2*k+2] = gdat.stdvfittrflx[:, k]
                    print('Writing all light curves to %s...' % pathsaveflux)
                    np.savetxt(pathsaveflux, arry)
                    print('Writing the target light curve to %s...' % pathsavefluxtarg)
                    np.savetxt(pathsavefluxtarg, arry[:, :3])
                    gdat.listarry[o] = arry[:, :3]

                    # if running over multiple targets, copy the target light curve to a summary folder
                    if gdat.strgbase is not None:
                        pathsummorig = pathsavefluxtarg
                        pathsummdest = gdat.pathdatasumm + '%s_%s.csv' % (gdat.strgtarg, strgsecc)
                        print('Writing to %s...' % pathsummdest)
                        os.system('cp %s %s' % (pathsummorig, pathsummdest))

                    if gdat.booldiagmode:
                        for a in range(gdat.listarry[o][:, 0].size):
                            if a != gdat.listarry[o][:, 0].size - 1 and gdat.listarry[o][a, 0] >= gdat.listarry[o][a+1, 0]:
                                raise Exception('')

                else:
                    print('Skipping the analysis...')
                    
                path = retr_pathplot(gdat, 0, '', strgsecc, '', '', '', 'of11', 0, 'rflx')
                if not os.path.exists(path):
                    
                    gdat.mlikfittrflx = np.empty((gdat.numbtime[o], gdat.numbcomp))
                    gdat.stdvfittrflx = np.empty((gdat.numbtime[o], gdat.numbcomp))
                    print('Reading from %s...' % pathsaveflux)
                    arry = np.loadtxt(pathsaveflux)
                    gdat.timedata[o] = arry[:, 0]
                    for k in gdat.indxcomp:
                        gdat.mlikfittrflx[:, k] = arry[:, 2*k+1]
                        gdat.stdvfittrflx[:, k] = arry[:, 2*k+2]
                
                    print('Plotting light curves...')
                    if gdat.listlimttimeplot is not None:
                        gdat.indxtimelimt = []
                        for limttimeplot in gdat.listlimttimeplot:
                            gdat.indxtimelimt.append(np.where((gdat.timedata[o] > limttimeplot[0]) & (gdat.timedata[o] < limttimeplot[1]))[0])
                
                    if gdat.boolrefr and x == 1 and y == 1:
                        gdat.listmodeplot = [0, 1]
                    else:
                        gdat.listmodeplot = [0]
       
                    if gdat.listpixlaper is not None:
                        plot_lcur(gdat, gdat.cntpaper, 0.01 * gdat.cntpaper, k, o, '_' + strgsecc, '_aper')
                        
                    # plot the light curve of the target stars and background
                    for k in gdat.indxcomp:

                        if gdat.boolbdtr and (x == 1 and y == 1) or not gdat.boolbdtr and (k == 0 and x == 1 and y == 1):

                            plot_lcur(gdat, gdat.mlikfittrflx[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, strgoffs)
                            
                            if gdat.listlimttimeplot is not None:
                                for p in range(len(gdat.listlimttimeplot)):
                                    plot_lcur(gdat, gdat.mlikfittrflx[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, strgoffs, \
                                                                                                    indxtimelimt=gdat.indxtimelimt[p], indxzoom=p)
                                    if k == 0:
                                        plot_lcur(gdat, gdat.mlikfittflux[:, k], gdat.stdvfittflux[:, k], k, o, '_' + strgsecc, strgoffs, \
                                                                                          indxtimelimt=gdat.indxtimelimt[p], indxzoom=p, strgyaxi='flux')
                            
                    if gdat.boolbdtr:
                        gdat.mlikrflxbdtr = np.empty_like(gdat.mlikfittrflx)
                        gdat.mlikrflxspln = np.empty_like(gdat.mlikfittrflx)
                        for k in gdat.indxcomp:
                            
                            if k > 0 and (x != 1 or y != 1):
                                continue
                        
                            lcurbdtrregi, indxtimeregi, indxtimeregioutt, listobjtspln, timeedge = \
                                                tesstarg.util.bdtr_lcur(gdat.timedata[o], gdat.mlikfittrflx[:, k], \
                                                weigsplnbdtr=gdat.weigsplnbdtr, booladdddiscbdtr=gdat.booladdddiscbdtr, \
                                                epocmask=gdat.epocmask, perimask=gdat.perimask, duramask=gdat.duramask, bdtrtype=gdat.bdtrtype, verbtype=1)
                            gdat.mlikrflxbdtr[:, k] = np.concatenate(lcurbdtrregi)
                            gdat.mlikrflxspln[:, k] = 1. + gdat.mlikfittrflx[:, k] - gdat.mlikrflxbdtr[:, k]
                            #gdat.mlikrflxspln[:, k] = np.concatenate([listobjtspln[i](gdat.timedata[o][indxtimeregi[i]]) for i in range(len(listobjtspln))])
                            #numbregi = len(lcurbdtrregi)
                            #indxregi = np.arange(numbregi)
                            #rflxspln = [[] for l in indxregi]
                            #for l in indxregi:
                            #    rflxspln[l] = listobjtspln[l](gdat.timedata[o][indxtimeregi[l]])
                            #rflxspln = np.concatenate(rflxspln)
                            
                            if k == 0:
                                arry = np.empty((gdat.numbtime[o], 3))
                                arry[:, 0] = gdat.timedata[o]
                                arry[:, 1] = gdat.mlikrflxbdtr[:, 0]
                                arry[:, 2] = gdat.stdvfittrflx[:, 0]
                                print('Writing the target light curve to %s...' % pathsavefluxtargbdtr)
                                np.savetxt(pathsavefluxtargbdtr, arry)
                            
                            plot_lcur(gdat, gdat.mlikrflxspln[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, strgoffs, strgextn='_spln', \
                                                                                                                                timeedge=timeedge)
                            plot_lcur(gdat, gdat.mlikrflxbdtr[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, strgoffs, strgextn='_bdtr')
                            if gdat.listlimttimeplot is not None:
                                for p in range(len(gdat.listlimttimeplot)):
                                    plot_lcur(gdat, gdat.mlikrflxbdtr[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, strgoffs, \
                                                                                    indxtimelimt=gdat.indxtimelimt[p], strgextn='_bdtr', indxzoom=p)

                    else:
                        gdat.mlikrflxbdtr = gdat.mlikfittrflx[:, 0]
                    
                    if gdat.epoc is not None:
                        for k in gdat.indxcomp:
                            plot_lcur(gdat, gdat.mlikrflxbdtr[:, k], gdat.stdvfittrflx[:, k], k, o, '_' + strgsecc, strgoffs, boolphas=True)
                    
                    if np.median(gdat.mlikrflxbdtr) < 0:
                        print('Median of the light curve is negative. Skipping TLS...')
                    else:
                        if gdat.booltlss:

                            print('Performing TLS...')
                            print('gdat.mlikrflxbdtr')
                            summgene(gdat.mlikrflxbdtr)
                            print('gdat.timedata[o]')
                            summgene(gdat.timedata[o])
                            model = transitleastsquares(gdat.timedata[o], gdat.mlikrflxbdtr)
                            gdat.results = model.power()
                            gdat.sdee = gdat.results.SDE
                            gdat.perimaxm = gdat.results.period
                            gdat.peri = gdat.results.periods
                            gdat.dept = gdat.results.depth
                            gdat.dura = gdat.results.duration
                            gdat.powr = gdat.results.power
                            gdat.timetran = gdat.results.transit_times
                            gdat.phasmodl = gdat.results.model_folded_phase
                            gdat.pcurmodl = 2. - gdat.results.model_folded_model
                            gdat.phasdata = gdat.results.folded_phase
                            gdat.pcurdata = 2. - gdat.results.folded_y
       
                            print('Period', format(gdat.results.period, '.5f'), 'd')
                            print(len(gdat.results.transit_times), 'transit times in time series:', \
                                    ['{0:0.5f}'.format(i) for i in gdat.results.transit_times])
                            print('Transit depth', format(gdat.results.depth, '.5f'))
                            print('Best duration (days)', format(gdat.results.duration, '.5f'))
                            print('Signal detection efficiency (SDE):', gdat.results.SDE)

                            # plot TLS spectrum
                            figr, axis = plt.subplots(figsize=(12, 4))
                            axis.axvline(gdat.results.period, alpha=0.4, lw=3)
                            axis.set_xlim(np.min(gdat.results.periods), np.max(gdat.results.periods))
                            for n in range(2, 10):
                                axis.axvline(n*gdat.results.period, alpha=0.4, lw=1, linestyle="dashed")
                                axis.axvline(gdat.results.period / n, alpha=0.4, lw=1, linestyle="dashed")
                            axis.set_ylabel('SDE')
                            axis.set_xlabel('Period [days]')
                            axis.plot(gdat.results.periods, gdat.results.power, color='black', lw=0.5)
                            axis.set_xlim(0, max(gdat.results.periods))
                            plt.tight_layout()
                            path = gdat.pathimagobjt + 'spectlss_%s.%s' % (gdat.strgsaveextn, gdat.strgplotextn)
                            print('Writing to %s...' % path)
                            plt.savefig(path)
                            plt.close()
                            
                            # plot phase folded
                            for a in range(2):
                                figr, axis = plt.subplots(figsize=(12, 4))
                                axis.plot(gdat.results.model_folded_phase, gdat.results.model_folded_model, color='red')
                                axis.scatter(gdat.results.folded_phase, gdat.results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
                                if a == 0:
                                    axis.set_xlim(0.45, 0.55)
                                axis.set_xlabel('Phase')
                                axis.set_ylabel('Relative flux');
                                plt.tight_layout()
                                path = gdat.pathimagobjt + 'pcurtls%d_%s.%s' % (a, gdat.strgsaveextn, gdat.strgplotextn)
                                print('Writing to %s...' % path)
                                plt.savefig(path)
                                plt.close()
    
        # make animations
        # time indices to be included in the animation
        if gdat.boolplotframtotl:
            numbplotanim = gdat.numbtime[o].size
        else:
            numbplotanim = 100
        indxtimeplot = np.linspace(0., gdat.numbtime[o] - 1., numbplotanim).astype(int)
        
        # get time string
        objttime = astropy.time.Time(gdat.timedata[o], format='jd', scale='utc', out_subfmt='date_hm')
        listtimelabl = objttime.iso
        
        liststrgplotdata = ['resi', 'data', 'datazoom', 'datanbkg', 'modl']
        for strgplotdata in liststrgplotdata:
                
            # make animation plot
            pathbasecntp = retr_pathcntp(strgplotdata, gdat, '', strgsecc)
            if not os.path.exists(pathbasecntp + '.gif') and gdat.boolmakeanim:
            
                if strgplotdata == 'resi':
                    boolresi = True
                    vmin = gdat.vmincntpresi
                    vmax = gdat.vmaxcntpresi
                else:
                    boolresi = False
                    if strgplotdata == 'modl':
                        vmin = gdat.vmincntpmodl
                        vmax = gdat.vmaxcntpmodl
                    elif strgplotdata == 'zoom':
                        vmin = gdat.vmincntpdatazoom
                        vmax = gdat.vmaxcntpdatazoom
                    else:
                        vmin = gdat.vmincntpdata
                        vmax = gdat.vmaxcntpdata
                if strgplotdata == 'data' or strgplotdata == 'datazoom':
                    cntptemp = gdat.cntpdata
                if strgplotdata == 'modl':
                    cntptemp = cntpmodl
                if strgplotdata == 'datanbkg':
                    cntptemp = gdat.cntpdata - gdat.mlikfittrflx[None, None, None, :, -1] * gdat.medifittflux[-1]
                if strgplotdata == 'resi':
                    cntptemp = cntpresi
                args = [gdat, cntptemp, o, strgsecc]
                kwag = {'indxtimeplot': indxtimeplot, \
                                    'boolresi': boolresi, 'listindxpixlcolr': gdat.listpixlaper, \
                                    'listtimelabl':listtimelabl, \
                                    'vmin':vmin, 'vmax':vmax, 'lcur':gdat.mlikfittrflx[:, 0], 'time':gdat.timedata[o]}
            
                listpath = plot_cntpwrap(*args, **kwag)
            
                # make animation
                cmnd = 'convert -delay 20 '
                for path in listpath:
                    cmnd += '%s ' % path
                cmnd += '%s.gif' % pathbasecntp
                os.system(cmnd)
                
                # delete images
                for path in listpath:
                    os.system('rm %s' % path)
        
            else:
                print('Skipping the animation of frame plots...')
                
    return gdat.listarry, gdat.listmeta


def init_list(pathfile, strgbase, **kwag):
    
    pathbase = os.environ['TCAT_DATA_PATH'] + '/%s/' % strgbase
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


