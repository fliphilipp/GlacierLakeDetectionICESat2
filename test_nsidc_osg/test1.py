from utils import *
from os import listdir, makedirs
from os.path import isfile, join, exists
import os
import rsa
import argparse

# move input files
# os.system("mkdir misc && mkdir geojsons")
# os.system("mv test1 misc/ && mv test2 misc/ && mv *.geojson geojsons/")

# parse command line arguments
parser = argparse.ArgumentParser(description='Test script to print some stats for a given ICESat-2 ATL03 granule.')
parser.add_argument('--granule', type=str, default='ATL03_20210715182907_03381203_005_01.h5',
                    help='the producer_id of the input ATL03 granule')
parser.add_argument('--polygon', type=str, default='/geojsons/jakobshavn_small.geojson',
                    help='the file path of a geojson file for spatial subsetting')
args = parser.parse_args()

gtxs = 'all'
datadir = '/IS2data'

download_granule_nsidc(args.granule, gtxs, args.polygon, datadir, decedc(edc().u), decedc(edc().p))

filelist = [datadir[1:]+'/'+f for f in listdir(datadir[1:]) if isfile(join(datadir[1:], f)) & (args.granule in f)]
print('\nNumber of processed ATL03 granules to read in: ' + str(len(filelist)))
filename = filelist[0]
    
photon_data, bckgrd_data, ancillary = read_atl03(filename, geoid_h=True)
print_granule_stats(photon_data, bckgrd_data, ancillary, outfile='stats.txt')
