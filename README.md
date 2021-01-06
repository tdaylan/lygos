# lygos

## Introduction
lygos is an inference framework to model Transition Exoplanet Survey Satellite (TESS) data using Point Source Function (PSF) photometry. It extracts light curves from a time-series of images with well-characterized uncertainties.

## Usage
First, import the lygos module.
```
import lygos
```

Then, you can extract light curves of certain targets by a single function.

WASP-121b:
```
lygos.main(strgmast='WASP-121')
```

TOI-1233:
```
lygos.main(toiitarg=1233)
```

The light curve at an arbitrary RA and DEC ():
```
rasctarg = 124.343
decltarg = 138.927
lygos.main(rasctarg=rasctarg, decltarg=decltarg)
```
