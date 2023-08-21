# run locally with: 
# $ conda activate icelakes-env
# $ python3 detect_lakes.py --granule <granule_producer_id> --polygon geojsons/<polygon_name.geojson>

import argparse
import os
import sys
import pickle
import subprocess
import traceback
import numpy as np
import icelakes
from icelakes.utilities import encedc, decedc, get_size
from icelakes.nsidc import download_granule, edc
from icelakes.detection import read_atl03, detect_lakes, melt_lake

parser = argparse.ArgumentParser(description='Test script to print some stats for a given ICESat-2 ATL03 granule.')
parser.add_argument('--granule', type=str, default='ATL03_20220714010847_03381603_006_02.h5',
                    help='The producer_id of the input ATL03 granule')
parser.add_argument('--polygon', type=str, default='geojsons/simplified_GRE_2500_CW.geojson',
                    help='The file path of a geojson file for spatial subsetting') # geojsons/west_greenland.geojson
parser.add_argument('--is2_data_dir', type=str, default='IS2data',
                    help='The directory into which to download ICESat-2 granules')
parser.add_argument('--download_gtxs', type=str, default='all',
                    help='String value or list of gtx names to download, also accepts "all"')
parser.add_argument('--out_data_dir', type=str, default='detection_out_data',
                    help='The directory to which to write the output data')
parser.add_argument('--out_plot_dir', type=str, default='detection_out_plot',
                    help='The directory to which to write the output plots')
parser.add_argument('--out_stat_dir', type=str, default='detection_out_stat',
                    help='The directory to which to write the granule stats')
args = parser.parse_args()

# try to figure out where the script is being executed (just to show those maps at conferences, etc...)
try:
    with open('location-wrapper.sh', 'rb') as file: script = file.read()
    geoip_out = subprocess.run(script, shell=True, capture_output=True)
    compute_latlon = str(geoip_out.stdout)[str(geoip_out.stdout).find('<x><y><z>')+9 : str(geoip_out.stdout).find('<z><y><x>')]
    print('\nThis job is running at the following lat/lon location:%s\n' % compute_latlon)
except:
    compute_latlon='0.0,0.0'
    print('\nUnable to determine compute location for this script.\n')

# shuffling files around for HTCondor
for thispath in (args.is2_data_dir, args.out_data_dir, args.out_plot_dir):
    if not os.path.exists(thispath): os.makedirs(thispath)

# download the specified ICESat-2 data from NSIDC
input_filename, request_status_code = download_granule(args.granule, args.download_gtxs, args.polygon, args.is2_data_dir, 
                                             decedc(edc().u), decedc(edc().p))

# perform a bunch of checks to make sure everything went alright with the nsidc api
print('Request status code:', request_status_code, request_status_code==200)
if request_status_code != 200:
    print('NSIDC API request failed.')
    sys.exit(127)
if request_status_code==200:
    with open('success.txt', 'w') as f: print('we got some sweet data', file=f)
    if input_filename == 'none': 
        print('granule seems to be empty. nothing more to do here.') 
        sys.exit(69)
if os.path.exists(input_filename):
    if os.path.getsize(input_filename) < 31457280:# 30 MB
        print('granule seems to be empty. nothing more to do here.') 
        sys.exit(69)

gtx_list, ancillary = read_atl03(input_filename, gtxs_to_read='none')

# detect melt lakes
lake_list = []
granule_stats = [0,0,0,0]

for gtx in gtx_list:
    lakes_found, gtx_stats = detect_lakes(input_filename, gtx, args.polygon, verbose=False)
    for i in range(len(granule_stats)): granule_stats[i] += gtx_stats[i]
    lake_list += lakes_found

if granule_stats[0] > 0:
    with open('success.txt', 'w') as f: print('we got some data from NSIDC!!', file=f)
    print('Sucessfully retrieved data from NSIDC!!')
    
# print stats for granule
print('\nGRANULE STATS (length total, length lakes, photons total, photons lakes):%.3f,%.3f,%i,%i\n' % tuple(granule_stats))

# for each lake call the surrf algorithm for depth determination
# if it fails, just skip the lake, but print trackeback for the logs 
print('---> determining depth for each lake')
for i, lake in enumerate(lake_list):
    try: 
        lake.surrf()
        print('   --> %3i/%3i, %s | %8.3fN, %8.3fE: %6.2fm deep / quality: %8.2f' % (i+1, len(lake_list), lake.gtx, lake.lat, 
                                                                                 lake.lon, lake.max_depth, lake.lake_quality))
    except:
        print('Error for lake %i (detection quality = %.5f) ... skipping:' % (i+1, lake.detection_quality))
        traceback.print_exc()
        lake.lake_quality = 0.0

# remove zero quality lakes
# lake_list[:] = [lake for lake in lake_list if lake.lake_quality > 0]

# for each lake 
for i, lake in enumerate(lake_list):
    lake.lake_id = '%s_%s_%s_%04i' % (lake.polygon_name, lake.granule_id[:-3], lake.gtx, i)
    filename_base = 'lake_%05i_%s_%s_%s' % (np.clip(1000-lake.lake_quality,0,None)*10, 
                                                       lake.ice_sheet, lake.melt_season, 
                                                       lake.lake_id)
    # plot each lake and save to image
    fig = lake.plot_lake(closefig=True)
    figname = args.out_plot_dir + '/%s.jpg' % filename_base
    if fig is not None: fig.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0)
    
    # export each lake to h5 and pickle
    try:
        h5name = args.out_data_dir + '/%s.h5' % filename_base
        datafile = lake.write_to_hdf5(h5name)
        print('Wrote data file: %s, %s' % (datafile, get_size(datafile)))
    except:
        print('Could not write hdf5 file <%s>' % lake.lake_id)
        # try:
        #     pklname = args.out_data_dir + '/%s.pkl' % filename_base
        #     with open(pklname, 'wb') as f: pickle.dump(vars(lake), f)
        #     print('Wrote data file: %s, %s' % (pklname, get_size(pklname)))
        # except:
        #     print('Could not write pickle file.')

statsfname = args.out_stat_dir + '/stats_%s_%s.csv' % (args.polygon[args.polygon.rfind('/')+1:].replace('.geojson',''), args.granule[:-4])
with open(statsfname, 'w') as f: print('%.3f,%.3f,%i,%i,%s' % tuple(granule_stats+[compute_latlon]), file=f)
    
# clean up the input data
os.remove(input_filename)

print('\n-------------------------------------------------')
print(  '----------->   Python script done!   <-----------')
print(  '-------------------------------------------------\n')