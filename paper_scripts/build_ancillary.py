"""Build the ancillary archive layers: per-granule coverage statistics and the drainage basins."""
import argparse
import glob
import os
import re

import geopandas as gpd
import pandas as pd

RENAME = {
    'polygon': 'basin_file',
    'granule': 'granule_id',
    'length_granule_xatc_meters': 'length_granule_m',
    'length_FLUID_detected_lakes_xatc_meters': 'length_fluid_segments_m',
    'number_photons_granule': 'n_photons_granule',
    'number_photons_FLUID_detected_lakes': 'n_photons_fluid_segments',
}


def build_coverage(src, prefix, out):
    d = pd.read_csv(src).rename(columns=RENAME)
    d = d[d.basin_file.str.startswith(prefix)].copy()
    d['basin'] = d.basin_file.str.replace('.geojson', '', regex=False)
    # ATL03 granule id encodes acquisition time as ATL03_YYYYMMDDHHMMSS_...
    dt = d.granule_id.str.extract(r'ATL03_(\d{8})')[0]
    d['date'] = pd.to_datetime(dt, format='%Y%m%d').dt.strftime('%Y-%m-%d')
    d['melt_season'] = pd.to_datetime(dt, format='%Y%m%d').dt.year
    d['frac_length_lakes'] = d.length_fluid_segments_m / d.length_granule_m
    cols = ['granule_id', 'basin', 'melt_season', 'date', 'length_granule_m',
            'length_fluid_segments_m', 'frac_length_lakes',
            'n_photons_granule', 'n_photons_fluid_segments']
    d = d[cols].sort_values(['melt_season', 'basin', 'granule_id'], ignore_index=True)
    d.to_csv(out, index=False, float_format='%.6g')
    print(f'wrote {out}: {len(d):,} rows, {os.path.getsize(out)/1e6:.1f} MB')
    print(f'  unique granules {d.granule_id.nunique():,} | '
          f'along-track {d.length_granule_m.sum()/1e9:.2f} million km | '
          f'photons {d.n_photons_granule.sum()/1e9:.1f} billion')
    return d


def build_basins(geojson_dir, pattern, out, crs):
    frames = []
    for fn in sorted(glob.glob(os.path.join(geojson_dir, pattern))):
        name = os.path.basename(fn).replace('.geojson', '')
        g = gpd.read_file(fn)
        g['basin'] = name.replace('simplified_', '')
        g['variant'] = 'simplified' if name.startswith('simplified_') else 'full'
        frames.append(g[['basin', 'variant', 'geometry']])
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs='EPSG:4326').to_crs(crs)
    gdf.to_file(out, layer='drainage_basins', driver='GPKG')
    print(f'wrote {out}: {len(gdf)} polygons ({gdf.variant.value_counts().to_dict()}), CRS {crs}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stats-combined', required=True)
    p.add_argument('--prefix', required=True, help='basin filename prefix, e.g. GRE_2200')
    p.add_argument('--geojson-dir', required=True)
    p.add_argument('--basin-pattern', required=True, help="e.g. '*GRE_2200_*.geojson'")
    p.add_argument('--crs', default='EPSG:3413')
    p.add_argument('--out-coverage', required=True)
    p.add_argument('--out-basins', required=True)
    a = p.parse_args()
    build_coverage(a.stats_combined, a.prefix, a.out_coverage)
    build_basins(a.geojson_dir, a.basin_pattern, a.out_basins, a.crs)
