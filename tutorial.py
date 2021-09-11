import os, sys
import numpy as np

import lygos
import miletos
from tdpy import summgene

# first, make sure that the environment variable $TCAT_DATA_PATH is set to the folder, where you would like the output plots and data to appear

def cnfg_HATP19():

    strgmast = 'HAT-P-19'
    lygos.init( \
               strgmast=strgmast, \
              )
        

def IRAS090263817():
   
    labltarg = 'IRAS09026-3817'
    rasctarg = 136.1388872
    decltarg = -38.4895431

    for k in range(1, 4):
        lygos.init(rasctarg=rasctarg, decltarg=decltarg, labltarg=labltarg, maxmnumbstar=k, boolplotrflx=True, boolplotcntp=True)


def ASASNN():
    
    # a string that will be used to query the target on MAST, which should be resolvable on MAST
    liststrgmast = [ \
                    'ASASSN-19bt', \
                    #'ASASNN-19fp', \
                   ]
    for strgmast in liststrgmast:
        lygos.init( \
                   strgmast=strgmast, \
                   strgclus='ASASSN', \
                   boolplotrflx=True, \
                  )


def cnfg_Ross619():
    # Ben Rackham
    lygos.init( \
                   strgmast='Ross 619', \
                   boolcbvs=False, \
                   #listtsecsele=[28, 29, 30], \
                   #typedatatess='lygos-best', \
                  )


def tutorial():
    
    lygos.init(strgmast='WASP-121')
    lygos.init(toiitarg=1233)
    lygos.init(rasctarg=124.343, decltarg=decltarg)


def cnfg_TOI2406():

    lygos.init( \
                   toiitarg=2406, \
                   boolplotrflx=True, \
                   #ticitarg=212957629, \
                   #labltarg='TOI-2406', \
                  )


def cnfg_WASP121():
    
    lygos.init( \
               strgmast='WASP-121', \
               boolplotquat=True, \
               boolplot=True, \
               boolcalcconr=True, \
               boolanim=True, \
               boolanimframtotl=False, \
              )
        

def cnfg_TOI1233():
    
    lygos.init( \
         toiitarg=1233, \
         numbside=21, \
         boolplotquat=True, \
         boolplot=True, \
         boolplotrflx=True, \
         boolanim=True, \
         boolanimframtotl=False, \
        )
        

def cnfg_V563Lyr():
    
    lygos.init( \
         strgmast='V563 Lyr', \
        )


def cnfg_Pleides():
    
    lygos.init( \
         strgmast='Electra', \
        )


def cnfg_KOI1003():
    
    lygos.init( \
         ticitarg=122374527, \
         abltarg='KOI 1003', \
         #strgmast='KOI-1003', \
         epocpmot=2019.3, \
        )


def cnfg_Luhman16():
   
    pmratarg = -2762.16 # [mas/yr]
    pmdetarg = 357.79 # [mas/yr]
    
    # exofop (J2015.5)
    #rasctarg = 162.328259
    #decltarg = -53.319372°
    
    # TICv8 (J2000)
    rasctarg = 162.328814
    decltarg = -53.319466
    
    lygos.init( \
         rasctarg=rasctarg, \
         decltarg=decltarg, \
         labltarg='Luhman 16', \
         booladdddiscdetr=True, \
         maxmnumbstar=5, \
         epocpmot=2019., \
         pmratarg=pmratarg, \
         pmdetarg=pmdetarg, \
         psfntype='lion', \
        )
        

def cnfg_WD1856():

    ticitarg = 267574918
    strgmast = 'TIC 267574918'
    labltarg = 'WD 1856'
    
    lygos.init( \
         labltarg=labltarg, \
         #strgmast=strgmast, \
         ticitarg=ticitarg, \
         boolanim=True, \
         #datatype='sapp', \
        )


def cnfg_contamination():
    
    typedata = 'toyy'
    #typedata = 'obsd'
    
    dictpopl = miletos.retr_dictcatltic8('ffimhcon')
    if typedata == 'toyy':
        print('temp -- assigning random TESS magnitudes!')
        tmagtarg = np.random.rand() * 10 + 7
        #tmagtarg = dictpopl['tmag']
    else:
        tmagtarg = None
    
    for k in range(2):
        lygos.init( \
             ticitarg=dictpopl['ID'].astype(int)[k], \
             strgclus='contamination', \
             typepsfn='ontf', \
             tmagtarg=tmagtarg, \
             boolfittoffs=True, \
             boolplot=True, \
             boolplotquat=True, \
             boolanim=True, \
             typedata=typedata, \
            )
        

def cnfg_GJ299():
    
    lygos.init( \
         #boolfittoffs=True, \
         labltarg='GJ 299', \
         #typepsfn='ontf', \
         boolplotquat=True, \
         boolplot=True, \
         boolanim=True, \
         ticitarg=334415465, \
         epocpmot=2019.3, \
        )
        

def cnfg_spec():
    
    path = os.environ['TCAT_DATA_PATH'] + '/data/List_for_MIT_pilot.txt'
    data = np.loadtxt(path, delimiter='\t', skiprows=1)
    numbtarg = data.shape[0]
    indxtarg = np.arange(numbtarg)
    for k in indxtarg:
        ticitarg = int(data[k, 2])
        lygos.init( \
             ticitarg=ticitarg, \
             labltarg='TIC %s' % ticitarg, \
             strgtarg='speculus_%s' % ticitarg, \
            )
        
    
def cnfg_saul():
    
    path = os.environ['TCAT_DATA_PATH'] + '/data/list_saul.txt'
    strgclus = 'saul'

    pathbase = os.environ['LYGOS_DATA_PATH'] + '/%s/' % strgbase
    os.system('mkdir -p %s' % pathbase)

    listticitarg = []
    listlabl = []
    listrasc = []
    listdecl = []
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


def cnfg_KeplerEBs():
    # Amaury
    strgclus = 'KeplerEBs'
    cnfg_arry(strgclus) 


def cnfg_SPECULOOS():
    
    strgclus = 'SPECULOOS'
    cnfg_arry(strgclus) 


def cnfg_arry(strgclus):
    
    pathdata = os.environ['TCAT_DATA_PATH'] + '/data/'
    if strgclus == 'KeplerEBs':
        path = pathdata + 'KeplerEBs/Kepler_binaries_priority.csv'
    else:
        path = pathdata + 'SPECULOOS/final_catalog.csv'
    
    cntr = 0
    indx
    listtici = []
    print('Reading from %s...' % path)
    for line in open(path, 'r'):
        if cntr == 0:
            cntr += 1
            continue
        linesplt = line.split(',')
        if strgclus == 'KeplerEBs':
            listtici.append(int(linesplt[1]))
        else:
            listtici.append(int(linesplt[23]))
    
    numbtarg = len(listtici)
    indxtarg = np.arange(numbtarg)
    listintgresu = np.empty(numbtarg)
    for k in indxtarg:
        temp = lygos.init( \
                              ticitarg=listtici[k], \
                              strgclus=strgclus, \
                             )
        

def chec_runs():
    
    strg = 'spec1313'

    path = os.environ['TCAT_DATA_PATH'] + '/'
    liststrgfile = fnmatch.filter(os.listdir(path), '%s_*' % strg)
    numb = len(liststrgfile)
    listbool = np.zeros(numb, dtype=bool)
    for k, strgfile in enumerate(liststrgfile):
        liststrgextn = fnmatch.filter(os.listdir(path + strgfile + '/data/'), 'rflx_*')
        if len(liststrgextn) == 1:
            listbool[k] = True
    print('numb')
    print(numb)
    print('np.where(listbool).size')
    print(np.where(listbool)[0].size)
    print(float(np.where(listbool)[0].size) / numb)


def cnfg_test347543557():

    ticitarg = 347543557
    labltarg = 'TIC 347543557'
    strgtarg = 'test347543557'
    listlimttimeplot = [[2458428, 2458430]]
    lygos.init( \
         ticitarg=ticitarg, \
         labltarg=labltarg, \
         strgtarg=strgtarg, \
         boolplotframtotl=True, \
         listlimttimeplot=listlimttimeplot, \
        )


def cnfg_GRB191016A():
    
    rasctarg = 30.2695
    decltarg = 24.5099
    labltarg = 'GRB191016A'
    listlimttimeplot = []
    for timedelt in [1.]:
        listlimttimeplot.append(2458772.67 + np.array([-timedelt, timedelt]))
    listtimeplotline = [2458772.67291666667]
    boolfittoffs = False#True
    boolcuttqual = False
    lygos.init( \
         rasctarg=rasctarg, \
         decltarg=decltarg, \
         labltarg=labltarg, \
         boolcuttqual=boolcuttqual, \
         boolfittoffs=boolfittoffs, \
         numbside=9, \
         boolbdtr=False, \
         listtimeplotline=listtimeplotline, \
         listlimttimeplot=listlimttimeplot, \
        )



def cnfg_lindsey():
    
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    pathbase = os.environ['LYGOS_DATA_PATH'] + '/'
    pathdata = pathbase + 'data/'
    pathimag = pathbase + 'imag/'
    os.system('mkdir -p %s' % pathdata)
    os.system('mkdir -p %s' % pathimag)
    for strgclus in [ \
                     #'targets_vf', \
                     #'sne_matched_03172021', \
                     'lygos_rerun_1', \
                     #'SNe-nominal-mission-all', 'SNe-extendedmission-27-32', 'SNe-S28-TNS-TESS', 'SNe_fausnaugh18' \
                    ]:

        pathcsvv = pathdata + '%s.csv' % strgclus
        print('Reading from %s...' % pathcsvv)
        objtfile = open(pathcsvv, 'r')
        k = 0
        for line in objtfile:
            if k == 0:
                k += 1
                continue
            linesplt = line.split(',')
            labltarg = linesplt[2]
            c = SkyCoord('%s %s' % (linesplt[3], linesplt[4]), unit=(u.hourangle, u.deg))
            rasctarg = c.ra.degree
            decltarg = c.dec.degree
            
            print('labltarg')
            print(labltarg)
            print('linesplt[1]')
            print(linesplt[1])
            print('linesplt[2]')
            print(linesplt[2])
            print('rasctarg')
            print(rasctarg)
            print('decltarg')
            print(decltarg)
            
            dictoutp = lygos.init( \
                                       rasctarg=rasctarg, \
                                       decltarg=decltarg, \
                                       labltarg=labltarg, \
                                       boolcbvs=False, \
                                       strgclus=strgclus, \
                                       booltpxflygo=False, \
                                       boolbdtr=False, \
                                       boolplotrflx=True, \
                                      )
            
            listtsec = dictoutp['listtsec']
            numbtsec = len(listtsec)
            indxtsec = np.arange(numbtsec)
            for o in indxtsec:
                cmnd = 'cp %s%s/%s/imag/* %s%s/imag/' % (pathbase, strgclus, dictoutp['strgtarg'], pathbase, strgclus)
                print(cmnd)
                os.system(cmnd)
                pathsaverflxtarg = dictoutp['pathsaverflxtargsc%02d' % listtsec[o]]
                cmnd = 'cp %s %s%s/data/' % (pathsaverflxtarg, pathbase, strgclus)
                print(cmnd)
                os.system(cmnd)
            k += 1


def cnfg_DJ():
    
    listname = ['quasar_transients_2', 'quasar_gaia_locs', 'galRaDec', 'raDecs_transient']
    numbname = len(listname)
    indxname = np.arange(numbname)
    for n in indxname:
        strgclus = listname[n]
        path = os.environ['LYGOS_DATA_PATH'] + '/data/djjj/' + listname[n]
        print('Reading from %s...' % path)
        arry = np.loadtxt(path, delimiter=' ')
        rasc = arry[:, 0]
        decl = arry[:, 1]
        numbtarg = rasc.size
        indxtarg = np.arange(numbtarg)
        for k in indxtarg:
            labltarg = 'Target %04d' % k
            strgtarg = 'lygos_DJ_target_%04d' % k
            rasctarg = rasc[k]
            decltarg = decl[k]
            lygos.init( \
                 rasctarg=rasctarg, \
                 decltarg=decltarg, \
                 labltarg=labltarg, \
                 strgclus=strgclus, \
                 maxmnumbstar=1, \
                 boolplotrflx=True, \
                 #boolplotquat=True, \
                 #boolanim=True, \
                 strgtarg=strgtarg, \
                 #boolbdtr=False, \
                )
            #break
            #k += 1



def cnfg_TIC284856863():
    '''
    12 July 2021, young star with accretion disk from Max (TYC 2597-735-1)
    '''

    lygos.init( \
               boolplotrflx=True, \
               boolplotcntp=True, \
               boolplotquat=True, \
               boolregrforc=True, \
               boolplotforc=True, \
               maxmnumbstar=999, \
               maxmdmag=6, \
               ticitarg=284856863, \
               labltarg='TYC 2597-735-1', \
              )


def cnfg_ASASSN20qc():
    '''
    13 July 2021, AGN from DJ
    '''
    rasctarg = 63.260208 
    decltarg = -53.0727

    labltarg = 'ASASSN-20qc'

    lygos.init( \
               boolplotrflx=True, \
               boolplotcntp=True, \
               boolfittoffs=True, \

               labltarg=labltarg, \
               
               rasctarg=rasctarg, \
               decltarg=decltarg, \
               
               boolregrforc=True, \
               boolplotforc=True, \
              )





globals().get(sys.argv[1])()


