"""Extract along-track depth measurements from FLUID/SuRRF lake HDF5 files into one Parquet table.

One row per 5 m along-track depth measurement, keyed on lake_id so it joins to the
lake-segment summary table. Lets users work with the depth data without opening
thousands of HDF5 files.
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd

# depth_data fields written by melt_lake.write_to_hdf5(), in output order
DEPTH_FIELDS = ['xatc', 'lon', 'lat', 'depth', 'conf', 'h_fit_surf', 'h_fit_bed', 'std_surf', 'std_bed']

# float16 in the HDF5 has no Parquet equivalent; float64 lat/lon is kept (float32 costs ~1 m)
DTYPES = {
    'xatc': 'float32', 'lon': 'float64', 'lat': 'float64',
    'depth': 'float32', 'conf': 'float32',
    'h_fit_surf': 'float32', 'h_fit_bed': 'float32',
    'std_surf': 'float32', 'std_bed': 'float32',
}


def read_one(path, lake_id):
    with h5py.File(path, 'r') as f:
        g = f['depth_data']
        df = pd.DataFrame({k: g[k][()] for k in DEPTH_FIELDS})
    df.insert(0, 'lake_id', lake_id)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', required=True, help='lakestats CSV with lake_id and label columns')
    p.add_argument('--data-dir', required=True, help='directory holding the <label>/<lake_id>.h5 tree')
    p.add_argument('--labels', nargs='+', default=['good_lake'], help='labels to include')
    p.add_argument('--out', required=True, help='output Parquet path')
    args = p.parse_args()

    summary = pd.read_csv(args.summary, usecols=['lake_id', 'label'])
    summary = summary[summary.label.isin(args.labels)]
    print(f'{len(summary)} segments with labels {args.labels}')

    frames, failed = [], []
    for i, r in enumerate(summary.itertuples(index=False), 1):
        fn = os.path.join(args.data_dir, r.label, f'{r.lake_id}.h5')
        try:
            frames.append(read_one(fn, r.lake_id))
        except Exception as e:
            failed.append((r.lake_id, repr(e)[:100]))
        if i % 1000 == 0:
            print(f'  {i}/{len(summary)}', flush=True)

    df = pd.concat(frames, ignore_index=True)
    df = df.astype(DTYPES)
    df['lake_id'] = df.lake_id.astype('category')
    df = df.sort_values(['lake_id', 'xatc'], ignore_index=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_parquet(args.out, compression='zstd', index=False)

    print(f'\nwrote {args.out}: {len(df):,} rows, {os.path.getsize(args.out)/1e6:.1f} MB')
    print(f'segments: {df.lake_id.nunique():,}   failed: {len(failed)}')
    for x in failed[:10]:
        print('   FAILED', x)


if __name__ == '__main__':
    main()
