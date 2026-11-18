"""Add data-dictionary entries for the three Record 1 files that had none.

`segment_extents.csv`, `bundles/_bundle_manifest.csv` and `bundles/_missing_assets.csv` are
in the staged record but were documented in neither README.md nor data_dictionary.csv. Column
meanings below are taken from segment_extents.py and bundle_assets.py, not inferred from the
data. Re-running is safe: files already present in the dictionary are skipped.

    python zenodo_build/scripts/document_r1_extras.py [--dry-run]
"""
import argparse
import csv
import os

# (file, variable, dtype, units, description)
ROWS = [
    # --- segment_extents.csv ------------------------------------------------
    ('segment_extents.csv', 'lake_id', 'string', '',
     'ICESat-2 lake segment identifier. Joins to lake_segments.csv. good_lake segments only.'),
    ('segment_extents.csv', 'has_depth_data', 'bool', '',
     'Whether the segment HDF5 contains a depth_data group. False leaves every column below NaN.'),
    ('segment_extents.csv', 'prop_max_depth', 'float', 'm',
     "FLUID's detection-stage maximum depth estimate, read from properties/max_depth in the "
     'HDF5. This is NOT the SuRRF retrieval and reads lower; use maxdepth_c03 for the retrieval.'),
    ('segment_extents.csv', 'n_any', 'int', '',
     'Number of 5 m along-track depth bins with depth > 0, with no confidence gate.'),
    ('segment_extents.csv', 'ext_any', 'float', 'm',
     'Along-track EXTENT at depth > 0: distance from the first to the last such bin, '
     'max(xatc) - min(xatc). Dry ice between them is included, so this is not wet length. '
     'Wet length is n_any x 5 m. This is the quantity the quicklook plots annotate as '
     '"x.x km of lake surface".'),
    ('segment_extents.csv', 'maxdepth_any', 'float', 'm',
     'Maximum retrieved depth over bins with depth > 0, no confidence gate. Unreliable: at '
     'low confidence the lakebed fit is unconstrained. See maxdepth_c03.'),
    ('segment_extents.csv', 'n_c01', 'int', '',
     'Number of 5 m depth bins with depth > 0 and conf > 0.1.'),
    ('segment_extents.csv', 'ext_c01', 'float', 'm',
     'Along-track extent at depth > 0 and conf > 0.1. Reproduces the total_length column of '
     'the earlier lakestats tables to within 1e-6 m, which is the control this file was '
     'built against.'),
    ('segment_extents.csv', 'maxdepth_c01', 'float', 'm',
     'Maximum retrieved depth over bins with depth > 0 and conf > 0.1.'),
    ('segment_extents.csv', 'n_c03', 'int', '',
     'Number of 5 m depth bins with depth > 0 and conf > 0.3. Multiply by 5 m for wet length.'),
    ('segment_extents.csv', 'ext_c03', 'float', 'm',
     'Along-track extent at depth > 0 and conf > 0.3. This is the threshold used throughout '
     'this archive and in the accompanying paper.'),
    ('segment_extents.csv', 'maxdepth_c03', 'float', 'm',
     'Maximum retrieved depth over bins with depth > 0 and conf > 0.3. This is the quantity '
     'behind the dataset maximum of 33.17 m.'),
    ('segment_extents.csv', 'spacing_median', 'float', 'm',
     'Median along-track spacing between depth bins in this segment. Nominally 5 m.'),
    # --- lake_segments.gpkg -------------------------------------------------
    ('lake_segments.gpkg', '(all)', '', '',
     'GeoPackage mirror of lake_segments.csv: the same rows and the same columns, as point '
     'geometries in EPSG:3413. See the lake_segments.csv entries for every column.'),
    # --- bundles/_bundle_manifest.csv ---------------------------------------
    ('_bundle_manifest.csv', 'file', 'string', '',
     'Name of a ZIP archive in this record.'),
    ('_bundle_manifest.csv', 'n_files', 'int', '',
     'Number of files inside that archive.'),
    ('_bundle_manifest.csv', 'size_mb', 'float', 'MB',
     'Uncompressed size of the archive. Archives are stored uncompressed because their '
     'contents (HDF5, JPEG) are already compressed.'),
    # --- bundles/_missing_assets.csv ----------------------------------------
    ('_missing_assets.csv', 'asset', 'string', '',
     'Which per-segment asset is absent: quicklook_plots or detail_plots.'),
    ('_missing_assets.csv', 'lake_id', 'string', '',
     'The segment it is missing for. All 12 affected segments come from granule '
     'ATL03_20190814035453_07190403_006_02, whose plot output was lost in transfer from the '
     'compute cluster. Their HDF5 data and context plots are unaffected.'),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--record-dir', default='zenodo_build/greenland')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    dd = os.path.join(a.record_dir, 'data_dictionary.csv')
    existing = list(csv.DictReader(open(dd)))
    fields = list(existing[0].keys())
    have = {r['file'] for r in existing}

    new = [dict(zip(fields, r)) for r in ROWS if r[0] not in have]
    skipped = sorted({r[0] for r in ROWS if r[0] in have})

    print(f'data_dictionary.csv: {len(existing)} rows, documenting {sorted(have)}')
    print(f'adding {len(new)} rows for {sorted({r["file"] for r in new})}')
    if skipped:
        print(f'skipping (already documented): {skipped}')

    # Every file in the record should now be documented or deliberately not.
    on_disk = set()
    for root, _, names in os.walk(a.record_dir):
        on_disk.update(n for n in names if not n.startswith('.'))
    documented = have | {r['file'] for r in new}
    undocumented = sorted(f for f in on_disk - documented
                          if not f.endswith(('.md', '.zip')) and f != 'data_dictionary.csv')
    if undocumented:
        print(f'\nWARNING still undocumented: {undocumented}')

    if a.dry_run:
        print('\n--dry-run: nothing written')
        return
    if new:
        with open(dd, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(existing + new)
        print(f'\nwrote {dd}: {len(existing) + len(new)} rows')


if __name__ == '__main__':
    main()
