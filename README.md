# TCAT (TESS Cataloger)

## Introduction
TCAT is an analysis tool to model Transition Exoplanet Survey Satellite (TESS) data with Point Source Function (PSF) photometry.


## Usage
```
import tcat
    
ticitarg = 22529346
labltarg = 'WASP-121'
strgtarg = 'wasp0121'
tcat.main( \
     ticitarg=ticitarg, \
     labltarg=labltarg, \
     strgtarg=strgtarg, \
    )
        
```
