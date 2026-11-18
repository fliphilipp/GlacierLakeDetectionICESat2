"""Stage the fused ICESat-2 / Sentinel-2 training dataset for archiving.

Reads the Parquet tables produced by the ML pipeline, adds a lake_id column so every table
joins to the lake-depth archive, and writes them out under archive-friendly names.
"""
import argparse
import json
import os
import re

import pandas as pd

# lake_id -> IS2_id: IS2_id drops the quality-rank prefix, melt season and polygon prefix
LAKE_ID_RE = re.compile(
    r'^lake_\d+_(?P<sheet>\w+?)_\d{4}_simplified_GRE_\d+_(?P<basin>[A-Z]+)_ATL03_'
    r'(?P<stem>\d{14}_\d{8}_\d{3}_\d{2})_(?P<gtx>gt\d[lr])_(?P<idx>\d{4})$')

TABLES = {
    'fused_segments':   ('training_data_gris/lake_segment_properties.parquet', True),
    'fused_samples':    ('training_data_gris/lake_segments_depth_reflectance_S2.parquet', False),
    'training_segments': ('data_paper/lake_segments_info.parquet', True),
    'training_features': ('data_paper/lake_segments_depth_features.parquet', False),
}

# from RF-dev.ipynb, descriptor 'selected_features_angles' (chosen by recursive feature elimination)
MODEL = {
    'model': 'sklearn.ensemble.RandomForestRegressor',
    'target': 'depth',
    'features': ['B3_log', 'ratio_logB3_logB2', 'diff_logB3_logB4', 'diff_logB2_logB3',
                 'ratio_logB4_logB2', 'B2_log', 'ratio_logB4_logB3', 'B4_log', 'ndwi',
                 'B1_log', 'diff_logB2_logB4', 'B8_log', 'B5_log', 'PhA_B3', 'B6_log',
                 'AOT_log', 'B7_log', 'VZA_B3_cos', 'SZA_cos'],
    'hyperparameters': {'n_estimators': 1000, 'criterion': 'squared_error',
                        'max_features': 0.4, 'max_samples': 0.05, 'min_samples_leaf': 1},
    'feature_selection': 'recursive feature elimination, 19 of 62 candidate predictors',
    'cross_validation': '10-fold, grouped by ICESat-2 lake segment (IS2_id)',
    'training_filters': {
        'surrf_segment_quality_min': 2.0,
        'surrf_bathymetry_confidence_min': 0.3,
        'max_time_difference_hours': 48,
        'ndwi_surface_water_agreement_min': 0.85,
        'pearson_r_depth_vs_log_reflectance_min': 0.75,
    },
}


def build_crosswalk(seg_csv):
    seg = pd.read_csv(seg_csv, usecols=['lake_id', 'label'])
    m = seg.lake_id.str.extract(LAKE_ID_RE)
    seg['IS2_id'] = m.sheet + '_' + m.basin + '_' + m.stem + '_' + m.gtx + '_' + m.idx
    return seg.dropna(subset=['IS2_id'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ml-dir', required=True)
    p.add_argument('--lake-segments', required=True, help='Record 1 lake_segments.csv')
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cross = build_crosswalk(args.lake_segments)
    # one IS2_id can match two lake_ids where a granule was processed twice; keep the
    # non-duplicate member so the mapping used here is single-valued
    lookup = cross.drop_duplicates(subset='IS2_id', keep='first').set_index('IS2_id').lake_id

    cross[['lake_id', 'IS2_id', 'label']].to_csv(
        os.path.join(args.out_dir, 'segment_crosswalk.csv'), index=False)
    print(f'segment_crosswalk.csv: {len(cross):,} rows')

    for name, (rel, idx_is_id) in TABLES.items():
        df = pd.read_parquet(os.path.join(args.ml_dir, rel))
        if idx_is_id:
            df = df.reset_index()
        if 'IS2_id' not in df.columns:
            raise SystemExit(f'{name}: no IS2_id column or index')
        df.insert(0, 'lake_id', df.IS2_id.map(lookup))
        df['lake_id'] = df.lake_id.astype('category')
        df['IS2_id'] = df.IS2_id.astype('category')
        if '__index_level_0__' in df.columns:
            df = df.drop(columns='__index_level_0__')
        out = os.path.join(args.out_dir, f'{name}.parquet')
        df.to_parquet(out, compression='zstd', index=False)
        unmatched = df.lake_id.isna().sum()
        print(f'{name}.parquet: {len(df):>9,} rows x {len(df.columns):>3} cols   '
              f'{os.path.getsize(out)/1e6:7.1f} MB   unmatched lake_id: {unmatched}')

    with open(os.path.join(args.out_dir, 'model_config.json'), 'w') as f:
        json.dump(MODEL, f, indent=2)
    print('model_config.json written')


if __name__ == '__main__':
    main()
