"""Download the multispectral Sentinel-2 chip for the Figure 2 example lake.

Replicates the parameters used for the is2comp lakes in
GlacierLakeIS2ML/is2_comp_get_avg_features.ipynb (cell 21), so the Figure 2 chip is
produced the same way as the Figure 6 ones.
"""
import os
import sys
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
from shapely.geometry import box

ML_DIR = 'GlacierLakeDetectionICESat2/GlacierLakeIS2ML'
sys.path.insert(0, ML_DIR)
os.environ.setdefault('EE_PROJECT', 'ee-psarndt93')
import IS2ML_utils as U  # noqa: E402  (must follow sys.path/env setup)

LAKE_H5 = ('greenland-supraglacial-lake-depths-figures/data/is2/'
           'lake_094135_GrIS_2021_simplified_GRE_2200_NE_ATL03_20210720053125_04061205_006_01_gt3r_0045.h5')
OUT = 'greenland-supraglacial-lake-depths-figures/data/fig2/lake_094135_s2_10m.tif'

# exactly the band list and renaming used for the is2comp chips
SELECT_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12',
                'probability', 'AOT', 'WVP', 'SCL', 'MSK_CLDPRB', 'MSK_SNWPRB']
RENAME_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12',
                'cloudScore', 'AOT', 'WVP', 'SCL', 'MSK_CLDPRB', 'MSK_SNWPRB']
BUFFER_FACTOR = 1.0   # 1.5 overflows getDownloadURL at 10 m; 1.0 gives ~541 px
CRS_OUT = 'EPSG:3413'

lk = U.dictobj(U.read_melt_lake_h5(LAKE_H5))
df = lk.depth_data
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')

bbox_ll = gpd.GeoDataFrame(geometry=[box(*gdf.total_bounds)], crs=gdf.crs, index=[0])
xy = bbox_ll.to_crs(CRS_OUT).geometry.get_coordinates()
buff = BUFFER_FACTOR * np.max((xy.x.max() - xy.x.min(), xy.y.max() - xy.y.min()))
ll = bbox_ll.geometry.get_coordinates()
lon_c = ll.x.min() + (ll.x.max() - ll.x.min()) / 2
lat_c = ll.y.min() + (ll.y.max() - ll.y.min()) / 2

print(f'lake      : {os.path.basename(LAKE_H5)[:48]}...')
print(f'centre    : {lat_c:.4f} N, {lon_c:.4f} E')
print(f'ICESat-2  : {lk.date_time}')
print(f'buffer    : {buff:.0f} m  -> chip approx {2*buff/1000:.1f} km across')

props = U.download_S2_full_images(
    lon_c, lat_c, lk.date_time,
    buffer_m=buff,
    max_cloud_prob=20,
    days_buffer=7,
    min_sun_elevation=20,
    max_images=3,
    select_bands=SELECT_BANDS,
    select_bands_rename=RENAME_BANDS,
    imagery_filename=OUT,
    download_imagery=True,
)
print('\nreturned:', props if not hasattr(props, 'head') else f'\n{props}')
