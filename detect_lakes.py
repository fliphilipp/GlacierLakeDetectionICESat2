# author: Philipp Arndt, UC San Diego / Scripps Institution of Oceanography
# 
# intended for use on OSG OSPool, called in run_py.sh, which is called in a submit file 
# submit file is based on a granule list queried locally in make_granule_list.ipynb 
# see examples for submit files in: HTCondor_submit/ 
# see examples for granule lists in:  granule_lists/
# 
# run locally with: 
# $ conda activate icelakes-env
# $ python3 detect_lakes.py --granule <NSIDC download link to granule> --polygon geojsons/<polygon_name.geojson>

# can also download granule in script by producer_id: 
# $ conda activate icelakes-env
# $ python3 detect_lakes.py --granule <granule producer_id> --polygon geojsons/<polygon_name.geojson>

# 30 ish lakes for testing (with NSIDC download)
# python3 detect_lakes.py --granule https://n5eil02u.ecs.nsidc.org/esir/5000005624137/297348002/processed_ATL03_20230806063138_07192003_006_02.h5 --polygon geojsons/simplified_GRE_2200_CW.geojson

# 40+ lakes for testing (with NSIDC download)
# python3 detect_lakes.py --granule https://n5eil02u.ecs.nsidc.org/esir/5000005624136/264467978/processed_ATL03_20220714010847_03381603_006_02.h5 --polygon geojsons/simplified_GRE_2200_CW.geojson

# the one that seems to have trouble for greenland --> finished on OSG!
# python3 detect_lakes.py --granule https://n5eil02u.ecs.nsidc.org/esir/5000005624139/271812658/processed_ATL03_20200825091749_09320803_006_01.h5 --polygon geojsons/simplified_GRE_2200_SE.geojson

# the one that keeps running out of memory for antarctica on OSG
# python3 detect_lakes.py --granule https://n5eil02u.ecs.nsidc.org/esir/5000005624275/273048326/processed_ATL03_20211110235432_07561311_006_01.h5 --polygon geojsons/simplified_ANT_1000_East_E-Ep.geojson

import argparse
import os
import gc
import sys
import time
import pickle
import requests
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
parser.add_argument('--polygon', type=str, default='geojsons/simplified_GRE_2000_CW.geojson',
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

print('\npython args:', args, '\n')

# shuffling files around for HTCondor
print('Shuffling files around for HTCondor...')
for thispath in (args.is2_data_dir, args.out_data_dir, args.out_plot_dir):
    if not os.path.exists(thispath): 
        os.makedirs(thispath)

# if the granule is provided via an NSIDC download link, just download it using that link
# this is how most granules are processed, by making async requests to NSIDC and collecting 
# download links when they finished processing
# see README.md, and request_data_and_make_granule_list.ipynb
if 'https://' in args.granule:
    # download the specified granule from NSIDC
    filename = args.granule.split('/')[-1]
    granule_id = filename.replace('processed_', '')
    input_filename = args.is2_data_dir + '/' + filename
    
    download_successful = False
    attempt_nr = 0
    
    while (not download_successful) & (attempt_nr < 100):
        attempt_nr += 1
        print('\nAttempting to download granule from NSIDC... (try %i)' % attempt_nr)
        capability_url = f'https://n5eil02u.ecs.nsidc.org/egi/capabilities/ATL03.006.xml'
        session = requests.session()
        s = session.get(capability_url)
        session.get(s.url,auth=(decedc(edc().u),decedc(edc().p)))
        response = session.get(args.granule)
        request_status_code = response.status_code
        
        if request_status_code == 200:
            print(' - Saving file %s...' % args.granule)
            with open(input_filename, 'wb') as file:
                file.write(response.content)
            print(' - Downloaded file %s: %s' % (input_filename, get_size(input_filename)))
            download_successful = True
            del response
        else:
            sleepytime = np.random.randint(low=60, high=300)
            print(' - This attempt unsuccessful. Trying again in %i seconds...' % sleepytime)
            time.sleep(sleepytime)
            
# if the granule is provided by producer ID, then download the granule via async request
# (with a synchronous/streaming request being the fallback option)
# this should just be needed for granules that failed processing at NSIDC
# in request_data_and_make_granule_list.ipynb these were saved to GLD3-NSIDC-processing-errors.csv
# and processed in GLD3-failedNSIDC_granules.ipynb
else:
    try_nr = 0
    request_status_code = 0
    granule_id = args.granule
    
    while (request_status_code != 200) & (try_nr <= 3):
        try_nr += 1
        try:
            print('\nDOWNLOADING GRANULE FROM NSIDC (try %i for asynchronous request)' % try_nr)
            input_filename, request_status_code = download_granule(
                args.granule, 
                args.download_gtxs, 
                args.polygon, 
                args.is2_data_dir, 
                decedc(edc().u), 
                decedc(edc().p), 
                spatial_sub=True,
                request_mode='async',
                sleep_time=10,
                max_try_time=3600
            )
            if request_status_code != 200:
                print('  --> Request unsuccessful (%i), trying again in a minute...\n' % request_status_code)
                time.sleep(60)
            
        except:
            print('  --> Request unsuccessful (error raised in code), trying again in a minute...\n')
            traceback.print_exc()
            time.sleep(60)

    # stream request as fallback
    if (request_status_code != 200):
        try_nr = 0
        while (request_status_code != 200) & (try_nr <= 10):
            try_nr += 1
            try:
                print('\nDOWNLOADING GRANULE FROM NSIDC (try %i for streaming request)' % try_nr)
                input_filename, request_status_code = download_granule(
                    args.granule, 
                    args.download_gtxs, 
                    args.polygon, 
                    args.is2_data_dir, 
                    decedc(edc().u), 
                    decedc(edc().p), 
                    spatial_sub=True,
                    request_mode='stream'
                )
                if request_status_code != 200:
                    print('  --> Request unsuccessful (%i), trying again in a minute...\n' % request_status_code)
                    time.sleep(60)
                
            except:
                print('  --> Request unsuccessful (error raised in code), trying again in a minute...\n')
                traceback.print_exc()
                time.sleep(60)
                
    download_successful = True if (request_status_code == 200) else False
    

if download_successful:
    print('\n---> Data download from NSIDC was successful!!!')
else:
    print('\nNSIDC API request failed. (Request status code: %i)' % request_status_code)
    sys.exit(127)

# perform a bunch of checks to make sure everything went alright with the nsidc api
print('Request status code:', request_status_code)
if download_successful:
    with open('success.txt', 'w') as f: 
        print('we got some sweet data', file=f)
    if not os.path.exists(input_filename): 
        print('no granule found. nothing more to do here.') 
        sys.exit(69)
    if os.path.exists(input_filename):
        if os.path.getsize(input_filename) < 500000: # 1 MB
            print('granule seems to be empty. nothing more to do here.') 
            sys.exit(69)

# read in the data
gtx_list, ancillary = read_atl03(input_filename, gtxs_to_read='none')

# detect melt lakes
lake_list = []
granule_stats = [0,0,0,0]

for gtx in gtx_list:
    try:
        lakes_found, gtx_stats = detect_lakes(input_filename, gtx, args.polygon, verbose=False)
        for i in range(len(granule_stats)): granule_stats[i] += gtx_stats[i]
        lake_list += lakes_found
        del lakes_found, gtx_stats
        gc.collect()
    except:
        print('Something went wrong for %s' % gtx)
        traceback.print_exc()

try:
    if granule_stats[0] > 0:
        with open('success.txt', 'w') as f: print('we got some useable data from NSIDC!!', file=f)
        print('\n_____________________________________________________________________________\nSucessfully got some useable data from NSIDC!!')
except:
    traceback.print_exc()
    
# print stats for granule
try:
    print('GRANULE STATS (length total, length lakes, photons total, photons lakes):%.3f,%.3f,%i,%i' % tuple(granule_stats))
    print('_____________________________________________________________________________\n')
except:
    traceback.print_exc()

try:
    max_lake_length = 30000 # meters (there are no lakes >30km and it's usually where something went wrong over the ocean)
    lake_list[:] = [lake for lake in lake_list if (lake.photon_data.xatc.max()-lake.photon_data.xatc.min()) <= max_lake_length]
except:
    traceback.print_exc()

# for each lake call the surrf algorithm for depth determination
# if it fails, just skip the lake, but print trackeback for the logs 
print('---> determining depth for each lake in the granule')
for i, lake in enumerate(lake_list):
    try: 
        lake.surrf()
        if hasattr(lake, 'depth_data'):
            lake.get_sorting_quality()
            print('     --> %3i/%3i, %s | %8.3fN, %8.3fE: %6.2fm deep / quality: %8.2f' % (i+1, len(lake_list), lake.gtx, lake.lat, 
                                                                                     lake.lon, lake.max_depth, lake.depth_quality_sort))
        else:
            lake.lake_quality = 0.0
            lake.depth_quality_sort = 0.0
            print('     xxx Lake %i  has no valid surface (detection quality = %.5f) ... skipping:' % (i+1, lake.detection_quality))
    except:
        lake.lake_quality = 0.0
        lake.depth_quality_sort = 0.0
        print('     xxx Error for lake %i (detection quality = %.5f) ... skipping:' % (i+1, lake.detection_quality))
        traceback.print_exc()

# remove zero quality lakes
# lake_list[:] = [lake for lake in lake_list if lake.lake_quality > 0]

# for each lake 
print('\n---> writing output data:')
for i, lake in enumerate(lake_list):
    if hasattr(lake, 'depth_data'):
        print('     lake %3i:  ' % (i+1), end='')
        try:
            lake.lake_id = '%s_%s_%s_%04i' % (lake.polygon_name, lake.granule_id[:-3], lake.gtx, i)
            filename_base = 'lake_%06i_%s_%s_%s' % (int(np.round(np.clip(1000-lake.depth_quality_sort,0,None)*100)),
                                                         lake.ice_sheet, lake.melt_season, 
                                                         lake.lake_id)
            figname = args.out_plot_dir + '/%s_quicklook.jpg' % filename_base
            figname_detail = args.out_plot_dir + '/%s_details.jpg' % filename_base
            h5name = args.out_data_dir + '/%s.h5' % filename_base
            
            # plot each lake and save to image
            start_print = ''
            try:
                fig = lake.plot_lake(closefig=True)
                if fig is not None: 
                    fig.savefig(figname, dpi=150, bbox_inches='tight', pad_inches=0)
                    print('quicklook plot, %s' % get_size(figname), end='')
                    start_print = ' | '
            except:
                print('Could not make QUICKLOOK figure for lake <%s>' % lake.lake_id)
                traceback.print_exc()
    
            # plot details for each lake and save to image
            try:
                fig = lake.plot_lake_detail(closefig=True)
                if fig is not None: 
                    fig.savefig(figname_detail, dpi=80, bbox_inches='tight', pad_inches=0)
                    print('%sdetail plot, %s' % (start_print,get_size(figname_detail)), end='')
                    start_print = ' | '
            except:
                print('detail_plotting_error:')
                print('Could not make DETAIL figure for lake <%s>' % lake.lake_id)
                traceback.print_exc()
            
            # export each lake to h5 and pickle
            try:
                datafile = lake.write_to_hdf5(h5name)
                print('%sh5 data file, %s' % (start_print, get_size(datafile)))
                start_print = ' | '
                print('                data: %s' % datafile)
            except:
                print('Could not write hdf5 file <%s>' % lake.lake_id)
                traceback.print_exc()
    
            # only keep files where it was possible to both write the figure and the data file
            if os.path.isfile(figname) and (not os.path.isfile(h5name)):
                os.remove(figname)
            if os.path.isfile(figname_detail) and (not os.path.isfile(h5name)):
                os.remove(figname_detail)
            if os.path.isfile(h5name) and ((not os.path.isfile(figname)) and (not os.path.isfile(figname_detail))):
                os.remove(h5name)
                
        except:
            traceback.print_exc()
    else:
        print('     lake %i:  no valid depth data! (skipping...)' % (i+1))

try:
    statsfname = args.out_stat_dir + '/stats_%s_%s.csv' % (args.polygon[args.polygon.rfind('/')+1:].replace('.geojson',''),
                                                           granule_id.replace('.h5',''))
    stats = [args.polygon[args.polygon.rfind('/')+1:].replace('simplified_', ''), granule_id]
    stats += granule_stats
    with open(statsfname, 'w') as f: print('%s,%s,%.3f,%.3f,%i,%i' % tuple(stats), file=f)
except:
    print("could not write stats file")
    traceback.print_exc()
    
# clean up the input data
os.remove(input_filename)

print('\n-------------------------------------------------')
print(  '----------->   Python script done!   <-----------')
print(  '-------------------------------------------------\n')