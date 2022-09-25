import argparse
import os
import pickle
import icelakes
from subprocess import call
from icelakes.utilities import encedc, decedc
from icelakes.nsidc import download_granule, edc
from icelakes.detection import read_atl03, detect_lakes, melt_lake

parser = argparse.ArgumentParser(description='Test script to print some stats for a given ICESat-2 ATL03 granule.')
parser.add_argument('--granule', type=str, default='ATL03_20210715182907_03381203_005_01.h5',
                    help='The producer_id of the input ATL03 granule')
parser.add_argument('--polygon', type=str, default='geojsons/jakobshavn_small.geojson',
                    help='The file path of a geojson file for spatial subsetting')
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
    with open('location-wrapper.sh', 'rb') as file:
        script = file.read()
    rc = call(script, shell=True)
except:
    print('\nUnable to determine compute location for this script.\n')

# shuffling files around for HTCondor
for thispath in (args.is2_data_dir, args.out_data_dir, args.out_plot_dir):
    if not os.path.exists(thispath): os.makedirs(thispath)

# download the specified ICESat-2 data from NSIDC
input_filename = download_granule(args.granule, args.download_gtxs, args.polygon, args.is2_data_dir, decedc(edc().u), decedc(edc().p))

gtx_list, ancillary = read_atl03(input_filename, gtxs_to_read='none')

# detect melt lakes
lake_list = []
granule_stats = {'length_total': 0.0, 'length_lakes': 0.0, 'n_photons_total': 0, 'n_photons_lakes': 0}
for gtx in gtx_list:
    lakes_found, gtx_stats = detect_lakes(input_filename, gtx, args.polygon, verbose=False)
    for k in granule_stats.keys(): granule_stats[k] += gtx_stats[k]
    lake_list += lakes_found

# print stats for granule
print('\nGRANULE STATS (length total, length lakes, photons total, photons lakes):%.3f,%.3f,%i,%i' % tuple(granule_stats.values()))

# save plots and lake data dictionaries
for lake in lake_list:
    filename_base = 'lake_%05i_%s_%s_%s_%s_%s' % ((1.0-lake.detection_quality)*10000, lake.ice_sheet, lake.melt_season, 
                                                  lake.polygon_name, lake.granule_id[:-4], lake.gtx)
    # plot each lake and save to image
    fig = lake.plot_detected(min_width=0.0, min_depth=0.0);
    figname = args.out_plot_dir + '/%s.jpg' % filename_base
    if fig is not None: fig.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0)
    
    # export each lake to pickle (TODO: add .h5 option soon)
    pklname = args.out_data_dir + '/%s.pkl' % filename_base
    with open(pklname, 'wb') as f: pickle.dump(vars(lake), f)

lk = lake_list[0]
statsfname = args.out_stat_dir + '/stats_%s_%s_%s_%s.csv' % (lk.ice_sheet, lk.melt_season, lk.polygon_name, lk.granule_id[:-4])
with open(statsfname, 'w') as f: print('%.3f,%.3f,%i,%i' % tuple(granule_stats.values()), file=f)
    
# clean up the input data
os.remove(input_filename)

print('\n--------------------------------------')
print('----------->   SUCCESS!   <-----------')
print('--------------------------------------\n')