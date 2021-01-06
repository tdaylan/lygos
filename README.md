# lygos

## Introduction
lygos is an inference framework to model Transition Exoplanet Survey Satellite (TESS) data using Point Source Function (PSF) photometry. It extracts light curves from a time-series of images with well-characterized uncertainties.

## Usage
First, import the lygos module.
```
import lygos
```
Then, you can extract light curves of certain targets by a single function lygos.main.init(). There are two ways to indicate the path where you want the plot and data files to be produced. You can either set the environment variable 'LYGOS_DATA_PATH' (i.e., export LYGOS_DATA_PATH=/your/path/) or provide the pathbase argument.

WASP-121b:
```
lygos.main.init(strgmast='WASP-121')
```

TOI-1233:
```
lygos.main.init(toiitarg=1233)
```

The light curve at an arbitrary RA and DEC (124.343, 138.927):
```
rasctarg = 124.343
decltarg = 138.927
lygos.main.init(rasctarg=rasctarg, decltarg=decltarg)
```
