# lygos

## Introduction
lygos is an inference framework to model Transition Exoplanet Survey Satellite (TESS) data using Point Source Function (PSF) photometry. It extracts light curves from a time-series of images with well-characterized uncertainties.

## Usage
```
import lygos
    
ticitarg = 22529346
labltarg = 'WASP-121'
strgtarg = 'wasp0121'
pandora.main( \
     ticitarg=ticitarg, \
     labltarg=labltarg, \
     strgtarg=strgtarg, \
    )
        
```
