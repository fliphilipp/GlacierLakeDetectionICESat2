"""Re-run the sonar validation with the Sentinel-2 acquisition times corrected.

THE BUG
-------
`IS2ML_utils.get_timediff_sonar` (and its ICESat-2 twin `get_timediff`) build the Sentinel-2
acquisition time as

    s2datetime = datetime.fromtimestamp(ms_since_epoch / 1e3)   # LOCAL time, naive
    t_s2       = s2datetime.replace(tzinfo=timezone.utc)        # RELABELLED as UTC

`fromtimestamp` without a `tz` returns the machine's local wall-clock time; `.replace` then
relabels it UTC instead of converting it. The result is the true UTC time minus the local
offset. The features were computed in US/Pacific during summer, so every `tdiff_sec` is
25200 s (7 h, PDT) too small.

Two independent confirmations:
  * a lake HDF5 stores both the product id and the derived time:
    S2A_MSIL2A_20210720T142941_... -> sensing start 14:29:41 UTC, but `time_imagery` was
    written as 07:30:29Z. Exactly 6:59:12 early.
  * reconstructing the sonar scenes' acquisition times from `tdiff_sec` puts all 35 of them
    between 07:40 and 09:00 UTC, which no Sentinel-2 overpass at 22 W can be; adding 7 h puts
    them at 14:40-16:00 UTC, where the verified NE Greenland product sits.

The correction is not a fixed 7 h: `fromtimestamp` uses the Pacific offset in force at the
converted instant, which is PDT for these July surveys but PST for austral-summer scenes.
`timefix.true_tdiff_sec` inverts through the zone rules rather than hard-coding either.

WHAT IT CHANGES
---------------
Each sonar point's ensemble estimate is the intercept of a linear fit of the per-scene
predicted depth against `tdiff_sec` - i.e. the model's depth extrapolated to dt = 0. With
every `tdiff_sec` 7 h too small, "dt = 0" is really 7 h *after* the sonar survey, so the
intercept is
            a_published = a_true + 25200 * slope        (25200 s = 7 h, PDT)
and a filling lake is reported deeper than it was when the sonar ran. The slope itself, and
every individual scene's prediction, are untouched.

    python zenodo_build/scripts/sonar_validation_rerun.py
"""
import argparse, os
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np, pandas as pd
from timefix import true_tdiff_sec
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data', default='greenland-supraglacial-lake-depths-figures/data/sonar')
p.add_argument('--out',  default='zenodo_build/sonar_validation_timefix')
a = p.parse_args()

# The image selection the published figure was made with. Recovered by reproducing the saved
# ensemble file exactly: lakes 522 and 610b match bit-for-bit with the selection left
# uncommented in sonar-validation-avg.ipynb, but 610a only matches the *commented* variant,
# which additionally drops scenes 4, 5, 10 and 11 - the ones taken after that lake flooded.
SELECTION = {
    'lutz_Sonar_2022-07-09_lake522':  set(range(15)) - {3, 4},
    'lutz_Sonar_2022-07-04_lake610a': set(range(15)) - {2, 3, 4, 5, 8, 9, 10, 11},
    'lutz_Sonar_2022-07-09_lake610b': set(range(15)) - {0, 2, 6, 7},
}
# What the saved individual-estimates CSV actually contains for 610a (post-flood scenes kept).
SELECTION_AS_SAVED = dict(SELECTION, **{
    'lutz_Sonar_2022-07-04_lake610a': set(range(15)) - {2, 3, 8, 9}})

ens = pd.read_csv(f'{a.data}/along_track_validation_ensemble_mean_7d_predictions.csv')
ind = pd.read_csv(f'{a.data}/along_track_validation_individual_estimates.csv')

# the sonar side of the subtraction was handled correctly; read the survey times back so the
# Sentinel-2 side can be reinterpreted in the zone it was accidentally written in
TFMT = '%Y-%m-%d %H:%M:%SZ'
REF_EPOCH = {}
for sid in ['Sonar_2022-07-09_lake522', 'Sonar_2022-07-04_lake610a', 'Sonar_2022-07-09_lake610b']:
    t = gpd.read_file(f'{a.data}/lutz_2024/{sid}.shp').Time.apply(
        lambda v: datetime.strptime(v, TFMT).replace(tzinfo=timezone.utc).timestamp())
    REF_EPOCH['lutz_' + sid] = t.median()


def subset(df, selection):
    keep = {f'{lake}_{n:02d}' for lake, nums in selection.items() for n in nums}
    return df[df.IS2_id.isin(keep)].copy()


def ensemble(df, tcol='tdiff_sec'):
    """Per sonar point, the intercept of predicted depth regressed on time difference."""
    rows = []
    for (lake, pid), x in df.groupby(['id_lake', 'plot_id'], sort=False):
        lm = LinearRegression().fit(np.asarray(x[tcol]).reshape(-1, 1), x.depth_pred)
        rows.append({'id_lake': lake, 'plot_id': pid, 'xatc': x.xatc.mean(),
                     'lon': x.lon.mean(), 'lat': x.lat.mean(),
                     'depth': x.depth.mean(), 'depth_smooth': x.depth_smooth.mean(),
                     'depth_pred_mean': lm.intercept_,
                     'slope_m_per_day': lm.coef_[0] * 86400, 'n_scenes': len(x)})
    return pd.DataFrame(rows)


def stats(df):
    d = df.depth_smooth - df.depth_pred_mean
    return {'n': len(df), 'MAE': np.mean(np.abs(d)), 'bias': np.mean(d),
            'RMSE': np.sqrt(np.mean(d ** 2)),
            'r': pearsonr(df.depth_pred_mean, df.depth_smooth).statistic,
            'pct_under': np.sum(d) / np.sum(df.depth_smooth) * 100,
            'mean_sonar': df.depth_smooth.mean(), 'mean_pred': df.depth_pred_mean.mean()}


def report(name, df):
    print(f'\n{name}')
    rows = {lake: stats(g) for lake, g in df.groupby('id_lake')}
    rows['ALL'] = stats(df)
    t = pd.DataFrame(rows).T
    t.index = [i.replace('lutz_Sonar_', '') for i in t.index]
    print(t.to_string(float_format=lambda v: f'{v:8.4f}'))
    return t


# ---------------------------------------------------------------- control: reproduce as published
pub = ensemble(subset(ind, SELECTION))
chk = ens.merge(pub[['id_lake', 'plot_id', 'depth_pred_mean']], on=['id_lake', 'plot_id'],
                suffixes=('_file', '_repro'))
err = (chk.depth_pred_mean_file - chk.depth_pred_mean_repro).abs().max()
assert len(chk) == len(ens), f'merge lost rows: {len(chk)} of {len(ens)}'
assert err < 1e-9, f'failed to reproduce the published ensemble (max diff {err:.3e} m)'
print(f'control: reproduced all {len(chk):,} published ensemble values '
      f'(max difference {err:.2e} m)')

t_pub = report('AS PUBLISHED  (dt = 0 is really 7 h after the sonar survey)', pub)

# ---------------------------------------------------------------- corrected acquisition times
ind_fix = subset(ind, SELECTION)
ind_fix['tdiff_true'] = true_tdiff_sec(ind_fix.id_lake.map(REF_EPOCH), ind_fix.tdiff_sec)
applied = (ind_fix.tdiff_true - ind_fix.tdiff_sec).unique()
print(f'\ncorrection applied: {np.round(applied / 3600, 3)} h '
      f'(all three surveys are July, so all PDT)')
fix = ensemble(ind_fix, tcol='tdiff_true')
t_fix = report('TIME-CORRECTED', fix)

# ---------------------------------------------------------------- what the shift actually did
j = pub.merge(fix, on=['id_lake', 'plot_id'], suffixes=('_pub', '_fix'))
j['delta'] = j.depth_pred_mean_fix - j.depth_pred_mean_pub
print('\nchange in the ensemble estimate, by lake (fixed minus published):')
g = j.groupby('id_lake').agg(
    mean_delta=('delta', 'mean'), min_delta=('delta', 'min'), max_delta=('delta', 'max'),
    mean_slope=('slope_m_per_day_pub', 'mean'))
g.index = [i.replace('lutz_Sonar_', '') for i in g.index]
print(g.to_string(float_format=lambda v: f'{v:8.4f}'))
print(f'\n  slope x 7 h is the whole effect: mean |delta| = '
      f'{j.delta.abs().mean():.4f} m, max = {j.delta.abs().max():.4f} m')

# ---------------------------------------------------------------- sensitivity: 610a scene choice
print('\n' + '=' * 78)
print('SENSITIVITY: the saved individual-estimates CSV keeps the four post-flood scenes for')
print('610a that the published figure dropped. Both variants, with times corrected:')
alt_in = subset(ind, SELECTION_AS_SAVED)
alt_in['tdiff_true'] = true_tdiff_sec(alt_in.id_lake.map(REF_EPOCH), alt_in.tdiff_sec)
alt = ensemble(alt_in, tcol='tdiff_true')
report('TIME-CORRECTED, 610a with all 11 scenes', alt)

os.makedirs(a.out, exist_ok=True)
pub.to_csv(f'{a.out}/ensemble_as_published.csv', index=False)
fix.to_csv(f'{a.out}/ensemble_time_corrected.csv', index=False)
alt.to_csv(f'{a.out}/ensemble_time_corrected_610a_all_scenes.csv', index=False)
pd.concat({'as_published': t_pub, 'time_corrected': t_fix}).to_csv(f'{a.out}/stats.csv')
print(f'\nwrote {a.out}/')
