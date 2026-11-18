"""Test re-run of the two independent validations with the 16-predictor model.

Both validations are rebuilt from their raw per-scene feature tables, so the only thing that
changes between variants is the model (and, optionally, the acquisition-time fix). The
original notebooks are not touched.

  sonar    3 lakes surveyed by boat in NE Greenland (Lutz et al., 2024), 6,082 points.
           Reference = the Gaussian-smoothed in-situ sonar depth.
  is2comp  9 lakes with manually digitised ICESat-2 depths - 5 in Greenland (Melling et al.,
           2024) and 4 on the Amery Ice Shelf (Fricker et al., 2021) - against which the
           published radiative-transfer (RTE) retrievals are also scored. This is the
           cross-continent generalisation test, since no Antarctic lake is in the training set.

In both, each point's estimate is the intercept of a linear fit of per-scene predicted depth
against the scene's time difference, i.e. the depth extrapolated to dt = 0.

    python zenodo_build/scripts/validate_16feature_model.py
"""
import argparse, json, os, re
from datetime import datetime, timezone

import geopandas as gpd, h5py, joblib, numpy as np, pandas as pd
from timefix import true_tdiff_sec
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--ml',   default='GlacierLakeDetectionICESat2/GlacierLakeIS2ML')
p.add_argument('--figs', default='greenland-supraglacial-lake-depths-figures')
p.add_argument('--config', default='zenodo_build/greenland_training/model_config.json')
p.add_argument('--out',  default='zenodo_build/sonar_validation_timefix')
a = p.parse_args()

MODELS = {'v2 (62 feat, msl=1)':   'zenodo_build/models/rf_depth_v2.joblib',
          'rfe16 (16 feat, msl=5)': 'zenodo_build/models/rf_depth_rfe_noangle_msl5.joblib'}

cfg = json.load(open(a.config))
ANGLE = re.compile(r'(SZA|VZA_|RAA_|PhA_)')
FEATS16 = [f for f in cfg['rfe_selected_features_exploratory'] if not ANGLE.match(f)]

# The image selections the published figures were made with (see sonar_validation_rerun.py
# for how the sonar ones were recovered; the is2comp ones are live code in
# is2comp-validation-avg.ipynb cell 6).
SEL_SONAR = {'lutz_Sonar_2022-07-09_lake522':  set(range(15)) - {3, 4},
             'lutz_Sonar_2022-07-04_lake610a': set(range(15)) - {2, 3, 4, 5, 8, 9, 10, 11},
             'lutz_Sonar_2022-07-09_lake610b': set(range(15)) - {0, 2, 6, 7}}
SEL_IS2COMP = {'lake_greenland_melling_1': {0, 1, 4},
               'lake_greenland_melling_2': {0, 1, 3},
               'lake_greenland_melling_3': {4},
               'lake_greenland_melling_4': {0, 3},
               'lake_greenland_melling_5': {2, 3, 4, 5, 6, 7}}
NDWI_GATE = 0.1444        # is2comp pipeline only; the sonar notebook has this line commented out


def load_models():
    out = {}
    for name, path in MODELS.items():
        m = joblib.load(path)
        out[name] = (m, list(m.feature_names_in_))
        print(f'  {name}: {len(m.feature_names_in_)} predictors, {os.path.getsize(path)/1e6:.0f} MB')
    return out


def ensemble(df, tcol):
    """Per point, the intercept of predicted depth regressed on time difference."""
    rows = []
    for (lake, pid), x in df.groupby(['id_lake', 'plot_id'], sort=False):
        lm = LinearRegression().fit(np.asarray(x[tcol]).reshape(-1, 1), x.depth_pred)
        rows.append({'id_lake': lake, 'plot_id': pid, 'lat': x.lat.mean(),
                     'xatc': x.xatc.mean(), 'depth_smooth': x.depth_smooth.mean(),
                     'depth_pred_mean': lm.intercept_})
    return pd.DataFrame(rows)


def scores(ref, est):
    ok = np.isfinite(ref) & np.isfinite(est)
    d = ref[ok] - est[ok]
    return {'n': int(ok.sum()), 'MAE': np.mean(np.abs(d)), 'bias': np.mean(d),
            'RMSE': np.sqrt(np.mean(d ** 2)),
            'r': pearsonr(ref[ok], est[ok]).statistic,
            # same convention as figscode.getstats_comparison: negative = underestimate
            'pct_est_ref': (est[ok].sum() / ref[ok].sum() - 1) * 100}


def show(title, table):
    print(f'\n{title}')
    print(pd.DataFrame(table).T.to_string(float_format=lambda v: f'{v:8.4f}'))


os.makedirs(a.out, exist_ok=True)
print('models:')
models = load_models()

# ==================================================================== 1. SONAR
print('\n' + '=' * 78 + '\n1. SONAR VALIDATION (NE Greenland, Lutz et al. 2024)\n' + '=' * 78)
feat = pd.read_parquet(f'{a.ml}/sonar/sonar_segments_depth_features_avg.parquet').reset_index(drop=True)
assert list(feat.columns[7:]) == cfg['features'], 'parquet feature order differs from model_config'
feat['id_lake'] = feat.IS2_id.str[:-3]
TFMT = '%Y-%m-%d %H:%M:%SZ'
REF_SONAR = {}
for sid in ['Sonar_2022-07-09_lake522', 'Sonar_2022-07-04_lake610a', 'Sonar_2022-07-09_lake610b']:
    t = gpd.read_file(f'{a.figs}/data/sonar/lutz_2024/{sid}.shp').Time.apply(
        lambda v: datetime.strptime(v, TFMT).replace(tzinfo=timezone.utc).timestamp())
    REF_SONAR['lutz_' + sid] = t.median()
keep = {f'{k}_{n:02d}' for k, v in SEL_SONAR.items() for n in v}
feat = feat[feat.IS2_id.isin(keep)].copy()

# the sonar reference is the in-situ depth smoothed with a 15 m Gaussian (S2 pixel scale),
# reproduced here exactly as in sonar-validation-avg.ipynb cell 1
STEP, SMOOTH_STD = 0.1, 15.0
wdw = int(np.ceil(SMOOTH_STD / STEP))
feat['depth_smooth'] = np.nan
for iid, d in feat.groupby('IS2_id'):
    xi = np.arange(0, d.xatc.max() + STEP, STEP)
    di = pd.Series(np.interp(xi, d.xatc, d.depth)).rolling(
        window=wdw * 3, center=True, min_periods=1, win_type='gaussian').mean(std=wdw)
    feat.loc[d.index, 'depth_smooth'] = np.interp(d.xatc, xi, di)

# control: the smoothing must match the saved individual-estimates CSV
saved = pd.read_csv(f'{a.figs}/data/sonar/along_track_validation_individual_estimates.csv')
chk = feat.merge(saved[['IS2_id', 'plot_id', 'depth_smooth', 'depth_pred']],
                 on=['IS2_id', 'plot_id'], suffixes=('', '_saved'))
assert len(chk) == len(feat), f'merge lost rows: {len(chk)} of {len(feat)}'
sm_err = (chk.depth_smooth - chk.depth_smooth_saved).abs().max()
print(f'control: sonar smoothing reproduced, max difference {sm_err:.2e} m')
assert sm_err < 1e-6

feat['tdiff_true'] = true_tdiff_sec(feat.id_lake.map(REF_SONAR), feat.tdiff_sec)
print('  sonar correction: '
      f'{np.unique(np.round((feat.tdiff_true - feat.tdiff_sec) / 3600, 3))} h')
variants, per_lake = {}, {}
for name, (m, mf) in models.items():
    feat['depth_pred'] = m.predict(feat[mf])
    for tname, tcol in [('published times', 'tdiff_sec'), ('time-corrected', 'tdiff_true')]:
        e = ensemble(feat, tcol)
        variants[f'{name} | {tname}'] = scores(e.depth_smooth.to_numpy(),
                                               e.depth_pred_mean.to_numpy())
        if tname == 'time-corrected':
            per_lake[name] = {lk.replace('lutz_Sonar_', ''):
                              scores(g.depth_smooth.to_numpy(), g.depth_pred_mean.to_numpy())
                              for lk, g in e.groupby('id_lake')}
        e.to_csv(f'{a.out}/sonar_{name.split()[0]}_{tcol}.csv', index=False)

# the archived model's own per-scene predictions, pushed through the same pipeline
arch = chk.rename(columns={'depth_pred': 'depth_pred'}).copy()
arch['id_lake'] = arch.IS2_id.str[:-3]
arch['tdiff_true'] = true_tdiff_sec(arch.id_lake.map(REF_SONAR), arch.tdiff_sec)
for tname, tcol in [('published times', 'tdiff_sec'), ('time-corrected', 'tdiff_true')]:
    e = ensemble(arch, tcol)
    variants[f'archived pickle | {tname}'] = scores(e.depth_smooth.to_numpy(),
                                                    e.depth_pred_mean.to_numpy())
show('sonar, all 6,082 points', variants)
for name, t in per_lake.items():
    show(f'sonar by lake, time-corrected - {name}', t)

# ==================================================================== 2. IS2COMP / RTE
print('\n' + '=' * 78 + '\n2. RTE COMPARISON (Melling et al. 2024 + Fricker et al. 2021)\n' + '=' * 78)
ic = pd.read_parquet(f'{a.ml}/is2comp/is2_segments_depth_features_avg.parquet').reset_index(drop=True)
assert list(ic.columns[7:]) == cfg['features'], 'is2comp feature order differs'
ic['id_lake'] = ic.IS2_id.str[:-3]
ic['depth_smooth'] = ic.depth
sel = {**{k: set(range(ic[ic.id_lake == k].IS2_id.nunique())) for k in ic.id_lake.unique()},
       **SEL_IS2COMP}
ic = ic[ic.IS2_id.isin({f'{k}_{n:02d}' for k, v in sel.items() for n in v})].copy()
REF_IC = {}
for lake in ic.id_lake.unique():
    with h5py.File(f'{a.figs}/data/is2comp/h5/{lake}.h5', 'r') as h:
        t = h['properties']['date_time'][()].decode()
    REF_IC[lake] = datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ').replace(
        tzinfo=timezone.utc).timestamp()
ic['tdiff_true'] = true_tdiff_sec(ic.id_lake.map(REF_IC), ic.tdiff_sec)
shifts = (ic.tdiff_true - ic.tdiff_sec) / 3600
print('  is2comp correction by lake (h):')
print(shifts.groupby(ic.id_lake).first().to_string())
print(f'{len(ic):,} rows, {ic.id_lake.nunique()} lakes, '
      f'{ic.IS2_id.nunique()} lake-scene combinations after selection')

cmp = pd.read_csv(f'{a.figs}/data/is2comp/comparison_melling_fricker.csv')
cmp['is_greenland'] = cmp.id_lake.str.contains('greenland')

def onto_manual_grid(ens):
    """Interpolate an ensemble estimate onto the manually digitised reference latitudes."""
    out = np.full(len(cmp), np.nan)
    for lake, c in cmp.groupby('id_lake'):
        e = ens[ens.id_lake == lake].sort_values('lat')
        out[c.index] = np.interp(c.lat, e.lat, e.depth_pred_mean)
    return out

rte_tbl = {}
for name, (m, mf) in models.items():
    pred = m.predict(ic[mf])
    pred[ic.ndwi.to_numpy() < NDWI_GATE] = 0.0      # the is2comp pipeline's NDWI gate
    ic['depth_pred'] = pred
    for tname, tcol in [('published times', 'tdiff_sec'), ('time-corrected', 'tdiff_true')]:
        cmp[f'pred_{name.split()[0]}_{tcol}'] = onto_manual_grid(ensemble(ic, tcol))

for region, mask in [('Greenland (Melling)', cmp.is_greenland),
                     ('Amery (Fricker)', ~cmp.is_greenland)]:
    d = cmp[mask]
    cols = ['predicted_depth'] + [c for c in cmp.columns if c.startswith('pred_')]
    cols += (['Red_z_lit', 'Green_z_lit'] if region.startswith('Greenland')
             else ['RTE_Landsat8_red_pan', 'RTE_Sentinel2_red'])
    tbl = {c: scores(d.manual.to_numpy(), d[c].to_numpy()) for c in cols if c in d}
    show(f'{region}: estimates vs manually digitised ICESat-2 depth', tbl)

cmp.to_csv(f'{a.out}/is2comp_with_16feature_predictions.csv', index=False)
pd.DataFrame(variants).T.to_csv(f'{a.out}/sonar_model_comparison.csv')
print(f'\nwrote {a.out}/')
