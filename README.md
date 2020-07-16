# pandora

## Introduction
pandora is an analysis framework to perform Point Source Function (PSF) modeling of time domain astronomical images such as the Transition Exoplanet Survey Satellite (TESS) and the Kepler telescope. Given a series of images, it enables computationally efficient and reliable inference of point source fluxes.


## Usage
```
import pandora
    
ticitarg = 22529346
labltarg = 'WASP-121'
strgtarg = 'wasp0121'
pandora.main( \
     ticitarg=ticitarg, \
     labltarg=labltarg, \
     strgtarg=strgtarg, \
    )
        
```
