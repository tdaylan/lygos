import lygos.main

# first, make sure that the environment variable $TCAT_DATA_PATH is set to the folder, where you would like the output plots and data to appear

# a string that will be used to query the target on MAST, which should be resolvable on MAST
strgmast = 'HAT-P-19'

lygos.main.main( \
               strgmast=strgmast, \
              )
        
