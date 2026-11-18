"""Along-track extent and maximum depth of every Greenland lake segment, from the HDF5 archive.

The manuscript quotes the length of a lake depth segment as the distance between the first and
the last along-track point at which SuRRF retrieved a depth greater than zero. Bare ice or ice
cover in between is included, so this is the segment's wet *extent*, not its wet length. It is
the same quantity the quicklook plots annotate as "x.x km of lake surface": `lakeanalysis/
S2lakes.py` sets `isdepth = dfd.depth > 0` and takes `max(xatc[isdepth]) - min(xatc[isdepth])`,
with no confidence gate.

Three masks are reported so the threshold choice is explicit, since the archive and the paper do
not use the same one:
    any   depth > 0                 the definition above, and what the plots show
    c01   depth > 0 and conf > 0.1  what lakestats `total_length` used
    c03   depth > 0 and conf > 0.3  the threshold the paper quotes elsewhere
True wet length (number of 5 m bins x 5 m) is reported alongside, because the archived
`length_water` column was computed as `xatc.diff().sum()`, which telescopes to max - min and is
therefore identical to `total_length` in every row rather than measuring wet length at all.

Maximum retrieved depth is computed at the same time, to check the paper's "up to 33 m" against
the depth profiles rather than against `properties/max_depth` (which is FLUID's detection-stage
estimate, not the SuRRF retrieval, and tops out lower).

    python zenodo_build/scripts/segment_extents.py
"""
import argparse, glob, os

import h5py
import numpy as np
import pandas as pd

# None means no confidence gate at all - the definition S2lakes.py plots and the
# manuscript quotes. 0.0 would NOT be the same thing: some samples have conf exactly 0.
MASKS = (('any', None), ('c01', 0.1), ('c03', 0.3))

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--h5dir',
               default='GLD3_complete/Greenland_ICESat2_FLUID_SuRRF_2019_2023/full_data_files_h5')
p.add_argument('--lakestats',
               default='greenland-supraglacial-lake-depths-figures/data/'
                       'FLUID-SuRRF_GrIS2019-23_lakestats_goodlakes.csv')
p.add_argument('--out', default='zenodo_build/greenland/segment_extents.csv')
a = p.parse_args()

files = sorted(glob.glob(os.path.join(a.h5dir, '*.h5')))
print(f'{len(files):,} segment files')

rows = []
for i, fn in enumerate(files):
    rec = {'lake_id': os.path.basename(fn)[:-3]}
    with h5py.File(fn, 'r') as f:
        rec['has_depth_data'] = 'depth_data' in f
        rec['prop_max_depth'] = float(f['properties']['max_depth'][()])
        if rec['has_depth_data']:
            xatc = np.asarray(f['depth_data']['xatc'][()], dtype=float)
            depth = np.asarray(f['depth_data']['depth'][()], dtype=float)
            conf = np.asarray(f['depth_data']['conf'][()], dtype=float)
            for tag, thr in MASKS:
                m = (depth > 0) if thr is None else ((depth > 0) & (conf > thr))
                rec[f'n_{tag}'] = int(m.sum())
                rec[f'ext_{tag}'] = (xatc[m].max() - xatc[m].min()) if m.any() else np.nan
                rec[f'maxdepth_{tag}'] = float(depth[m].max()) if m.any() else np.nan
            rec['spacing_median'] = float(np.median(np.diff(np.sort(xatc)))) if len(xatc) > 1 else np.nan
    rows.append(rec)
    if (i + 1) % 1000 == 0:
        print(f'  {i + 1:,}', flush=True)

df = pd.DataFrame(rows)

# the c01 extent must equal the archived total_length column, or the HDF5 read does not match
# the step that produced the published statistics
ls = pd.read_csv(a.lakestats, usecols=['lake_id', 'total_length', 'length_water'])
m = df.merge(ls, on='lake_id', validate='one_to_one')
both = m.ext_c01.notna() & m.total_length.notna()
err = (m.ext_c01[both] - m.total_length[both]).abs().max()
assert err < 1e-6, f'ext_c01 does not reproduce total_length (max diff {err:.3e} m)'
assert m.ext_c01.isna().sum() == m.total_length.isna().sum()
# length_water telescopes to max - min, so it equals total_length wherever the mask is
# non-empty; where it is empty, an empty Series sums to 0.0 while max - min is NaN
lw, tl = m.length_water.to_numpy(float), m.total_length.to_numpy(float)
nonempty = ~np.isnan(tl)
same = bool(np.isclose(lw[nonempty], tl[nonempty]).all())
n_phantom = int((np.isnan(tl) & (lw == 0.0)).sum())
print(f'control: ext_c01 reproduces total_length over {int(both.sum()):,} rows '
      f'(max difference {err:.3e} m)')
print(f'archived length_water == total_length wherever conf > 0.1 selects anything: {same} '
      f'(the telescoping xatc.diff().sum() bug); in the {n_phantom} segments it selects nothing, '
      'length_water reports 0.0 where total_length is NaN')
print(f'median along-track spacing: {df.spacing_median.median():.3f} m')

print(f'\n{"mask":>5}  {"n_seg":>6} {"min":>8} {"median":>8} {"mean":>8} {"max":>8}   '
      f'{"sum ext":>9} {"sum wet":>9}   {"max depth":>9}')
for tag, _ in MASKS:
    e = df[f'ext_{tag}'].dropna() / 1000
    print(f'{tag:>5}  {len(e):6d} {e.min():8.3f} {e.median():8.3f} {e.mean():8.3f} {e.max():8.3f}   '
          f'{e.sum():9.1f} {df[f"n_{tag}"].sum() * 5 / 1000:9.1f}   '
          f'{df[f"maxdepth_{tag}"].max():9.2f}')
print('                      (km)     (km)     (km)     (km)        (km)      (km)         (m)')
print(f'\nproperties/max_depth, for comparison: {df.prop_max_depth.max():.2f} m '
      '(FLUID detection stage, not the SuRRF retrieval)')

os.makedirs(os.path.dirname(a.out), exist_ok=True)
df.to_csv(a.out, index=False)
print(f'\nwrote {a.out}')
