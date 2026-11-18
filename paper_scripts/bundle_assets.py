"""Bundle per-segment HDF5 files and plots into one ZIP per melt season.

Zenodo caps a record at 100 files, so the ~22k per-segment files have to be bundled.
Contents are already compressed (gzip inside HDF5, JPEG), so the archives are stored
uncompressed - this is much faster and within a few percent of the same size.
"""
import argparse
import os
import zipfile

import pandas as pd

ASSETS = {
    #  key          directory              filename suffix
    'segments_h5':    ('detection_out_data', '.h5'),
    'quicklook_plots': ('detection_out_plot', '_quicklook.jpg'),
    'detail_plots':    ('detection_out_plot', '_details.jpg'),
    'context_plots':   ('detection_context',  '_context.jpg'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', required=True)
    p.add_argument('--base-dir', required=True)
    p.add_argument('--labels', nargs='+', default=['good_lake'])
    p.add_argument('--assets', nargs='+', default=list(ASSETS))
    p.add_argument('--out-dir', required=True)
    p.add_argument('--region-tag', default='GrIS')
    args = p.parse_args()

    s = pd.read_csv(args.summary, usecols=['lake_id', 'label', 'melt_season'])
    s = s[s.label.isin(args.labels)]
    os.makedirs(args.out_dir, exist_ok=True)

    manifest, missing = [], []
    for asset in args.assets:
        subdir, suffix = ASSETS[asset]
        for season, grp in s.groupby('melt_season'):
            out = os.path.join(args.out_dir, f'{asset}_{args.region_tag}_{season}.zip')
            n = 0
            with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
                for r in grp.itertuples(index=False):
                    src = os.path.join(args.base_dir, subdir, r.label, f'{r.lake_id}{suffix}')
                    if os.path.isfile(src):
                        z.write(src, arcname=f'{asset}_{args.region_tag}_{season}/{os.path.basename(src)}')
                        n += 1
                    else:
                        missing.append((asset, r.lake_id))
            mb = os.path.getsize(out) / 1e6
            manifest.append(dict(file=os.path.basename(out), n_files=n, size_mb=round(mb, 1)))
            print(f'  {os.path.basename(out):42s} {n:6,} files  {mb:8,.1f} MB', flush=True)

    mf = pd.DataFrame(manifest)
    mf.to_csv(os.path.join(args.out_dir, '_bundle_manifest.csv'), index=False)
    print(f'\n{len(mf)} archives, {mf.n_files.sum():,} files, {mf.size_mb.sum()/1000:.2f} GB')
    if missing:
        print(f'{len(missing)} expected files absent:')
        for a, lid in missing[:5]:
            print(f'   {a}: {lid}')
        pd.DataFrame(missing, columns=['asset', 'lake_id']).to_csv(
            os.path.join(args.out_dir, '_missing_assets.csv'), index=False)


if __name__ == '__main__':
    main()
