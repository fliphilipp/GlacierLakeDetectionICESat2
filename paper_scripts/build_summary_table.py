"""Build the archive lake-segment summary table from the FLUID/SuRRF HDF5 outputs.

Recomputes the per-segment depth statistics directly from each file's depth_data group so
that every number in the table has a stated, reproducible definition. Identity and basin
attribution are carried over from the existing lakestats CSV.

Differences from the original lakestats CSV (see data_dictionary.csv):
  * conf > 0.3     - every water and depth statistic in this table uses this single
                     bathymetry confidence threshold, matching the convention used in the
                     paper. Depths below it come from unconstrained lakebed fits. Users
                     wanting another threshold can recompute from depth_profiles.parquet.
  * length_water   - FIXED. The original used `dfc.xatc.diff().sum()`, which telescopes to
                     (max - min) and so measured segment extent, not water length. It
                     overstated the true value by ~1.31x and duplicated `total_length`.
                     Now computed as (number of water measurements) x the 5 m posting.
  * total_length   - renamed `depth_extent` to say what it actually is.
  * max_depth      - now the maximum depth at conf > 0.3. The algorithm's own internal
                     value is retained as `max_depth_surrf`, which uses a different
                     criterion and differs slightly.
"""
import argparse
import os

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd

POSTING_M = 5.0        # SuRRF along-track depth posting
CONF_MAIN = 0.3        # bathymetry confidence threshold for ALL water/depth statistics

# carried through unchanged from the existing lakestats CSV
IDENTITY_COLS = [
    'lake_id', 'label', 'ice_sheet', 'melt_season', 'date_time', 'lon', 'lat',
    'surface_elevation', 'depth_quality_sort', 'cycle_number', 'rgt', 'gtx',
    'beam_strength', 'beam_number', 'sc_orient', 'granule_id', 'polygon_name',
    'lat_max', 'lat_min', 'lon_max', 'lon_min', 'basin', 'sub_basin', 'glacier_type',
]


def segment_stats(path):
    """Per-segment depth statistics, recomputed from depth_data."""
    with h5py.File(path, 'r') as f:
        d = pd.DataFrame({k: f['depth_data'][k][()] for k in ('xatc', 'depth', 'conf')})
        segment_length = float(f['properties/length_extent'][()])
        max_depth = float(f['properties/max_depth'][()])

    water = d[(d.depth > 0) & (d.conf > CONF_MAIN)]

    return {
        'segment_length': segment_length,
        'n_depth_meas': len(d),
        'n_water_meas': len(water),
        'length_water': len(water) * POSTING_M,
        'depth_extent': (water.xatc.max() - water.xatc.min()) if len(water) else np.nan,
        'max_depth': water.depth.max() if len(water) else np.nan,
        'max_depth_surrf': max_depth,
        'depth_mean': water.depth.mean() if len(water) else np.nan,
        'depth_median': water.depth.median() if len(water) else np.nan,
        'ratio_water': len(water) / len(d) if len(d) else np.nan,
        'conf_mean': d.conf[d.depth > 0].mean() if (d.depth > 0).any() else np.nan,
        'conf_median': d.conf[d.depth > 0].median() if (d.depth > 0).any() else np.nan,
        'conf_q90': d.conf[d.depth > 0].quantile(0.90) if (d.depth > 0).any() else np.nan,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lakestats', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--labels', nargs='+', default=['good_lake', 'lake_bad_depth'])
    p.add_argument('--atl03-version', default='006')
    p.add_argument('--crs', default='EPSG:3413', help='3413 for Greenland, 3031 for Antarctica')
    p.add_argument('--out-csv', required=True)
    p.add_argument('--out-gpkg', required=True)
    args = p.parse_args()

    src = pd.read_csv(args.lakestats)
    src = src[src.label.isin(args.labels)].reset_index(drop=True)
    print(f'{len(src)} segments with labels {args.labels}')

    rows = []
    for i, r in enumerate(src.itertuples(index=False), 1):
        rows.append(segment_stats(os.path.join(args.data_dir, r.label, f'{r.lake_id}.h5')))
        if i % 2000 == 0:
            print(f'  {i}/{len(src)}', flush=True)

    out = pd.concat([src[IDENTITY_COLS], pd.DataFrame(rows)], axis=1)
    out.insert(out.columns.get_loc('granule_id') + 1, 'atl03_version', args.atl03_version)

    # Flag near-identical segments. A handful of granule-basin pairs were processed twice on
    # the compute cluster and both outputs were retained; the twins differ only at the level
    # of the algorithm's internal random jitter. Segments sharing a granule, ground track and
    # geolocation to 1e-6 deg get a shared duplicate_group id; everything else is NaN.
    key = (out.granule_id + '|' + out.gtx + '|'
           + out.lat.round(6).astype(str) + '|' + out.lon.round(6).astype(str))
    dup = key.duplicated(keep=False)
    out['duplicate_group'] = pd.Series(pd.factorize(key[dup])[0], index=key[dup].index).reindex(out.index)
    out['duplicate_group'] = out.duplicate_group.map(lambda v: f'dup_{int(v):03d}' if pd.notna(v) else '')

    out = out.sort_values(['melt_season', 'lake_id'], ignore_index=True)

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    out.to_csv(args.out_csv, index=False, float_format='%.6f')

    gdf = gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out.lon, out.lat),
                           crs='EPSG:4326').to_crs(args.crs)
    gdf.to_file(args.out_gpkg, layer='lake_segments', driver='GPKG')

    print(f'\nwrote {args.out_csv} ({os.path.getsize(args.out_csv)/1e6:.1f} MB)')
    print(f'wrote {args.out_gpkg} ({os.path.getsize(args.out_gpkg)/1e6:.1f} MB), CRS {args.crs}')
    print(f'\ncolumns ({len(out.columns)}): {list(out.columns)}')


if __name__ == '__main__':
    main()
