from utils import *
from os import listdir, makedirs
from os.path import isfile, join, exists
import os

granule = 'ATL03_20210715182907_03381203_005_01.h5'
shapefile = '/shapefiles/jakobshavn_small.shp'

datadir = '/IS2data'

c = edc()
download_granule_nsidc(granule, shapefile, datadir, c.u, c.p)

filelist = [datadir[1:]+'/'+f for f in listdir(datadir[1:]) if isfile(join(datadir[1:], f)) & ('.h5' in f)]
print('\nNumber of processed ATL03 granules to read in: ' + str(len(filelist)))
    
photon_data, bckgrd_data, ancillary = read_atl03(filelist[0], geoid_h=True)
print_granule_stats(photon_data, bckgrd_data, ancillary, outfile='stats.txt')