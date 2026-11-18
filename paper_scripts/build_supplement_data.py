"""Stage the inputs for the machine-learning supplementary figures (Figs S1-S3).

Produces, under the figures repo's data/sup/:

  rfe_stats.csv              the recursive-feature-elimination curve (62 -> 1 predictors),
                             copied from the RFE run in RF-varimp.ipynb
  ml_supplement_samples.parquet   one row per training sample, carrying the raw blue/green/red
                             reflectances the band-ratio method needs, plus OUT-OF-FOLD
                             predictions from the random forest and from the band ratio on the
                             identical grouped 10-fold split
  ml_supplement_segments.parquet  per-segment metadata used to pick the example segments and
                             scenes in Fig. S3
  ml_supplement_meta.json    band-ratio coefficients, feature lists and headline statistics

The band ratio is fitted the way the literature applies it (Sneed & Hamilton 2007; Moussavi
et al. 2016; Pope et al. 2016): a quadratic in X = ln(R_blue / R_green), with depth forced to
zero where NDWI falls below the water threshold.  It is fitted twice:
  * per fold, on that fold's training segments only  -> honest out-of-sample comparison
  * once on every sample                             -> the best case the method can achieve,
                                                        quoted as a reference
Folds are grouped by ICESat-2 segment, as for the random forest, because samples 5 m apart are
strongly autocorrelated and a random row split would leak them between folds.

    python zenodo_build/scripts/build_supplement_data.py
"""
import argparse, json, os, shutil, time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ML = 'GlacierLakeDetectionICESat2/GlacierLakeIS2ML'
NDWI_THRESH = 0.23   # the water threshold the thesis band-ratio fit used
MIN_REFL = 1.0       # reflectance floor; ln() of a zero-valued band is undefined

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--features', default='zenodo_build/greenland_training/training_features.parquet')
p.add_argument('--config',   default='zenodo_build/greenland_training/model_config.json')
p.add_argument('--reflectance', default=f'{ML}/data_paper/lake_segments_depth_reflectance_data.parquet')
p.add_argument('--info',     default=f'{ML}/data_paper/lake_segments_info.parquet')
p.add_argument('--rfe',      default=f'{ML}/rf_recursive_feature_elimination_stats.csv')
p.add_argument('--out',      default='greenland-supraglacial-lake-depths-figures/data/sup')
p.add_argument('--folds', type=int, default=10)
p.add_argument('--seed',  type=int, default=42)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

cfg = json.load(open(a.config))
hp = cfg['hyperparameters']
FEATS_ALL = cfg['features']
FEATS_16 = [f for f in cfg['rfe_selected_features_exploratory']
            if not f.startswith(('SZA', 'VZA_', 'RAA_', 'PhA_'))]
assert len(FEATS_ALL) == 62 and len(FEATS_16) == 16

df = pd.read_parquet(a.features)
ref = pd.read_parquet(a.reflectance, columns=['IS2_id', 'S2_id', 'xatc', 'depth', 'conf',
                                              'ndwi', 'lat', 'B2', 'B3', 'B4'])
# the two tables are two views of the same rows, in the same order - check, don't assume
assert len(df) == len(ref), f'{len(df)} vs {len(ref)} rows'
assert (df.IS2_id.values == ref.IS2_id.values).all(), 'IS2_id not aligned row-for-row'
assert np.array_equal(df.xatc.values, ref.xatc.values), 'xatc not aligned row-for-row'
assert np.allclose(df.depth.values, ref.depth.values), 'depth not aligned row-for-row'
for c in ['S2_id', 'B2', 'B3', 'B4']:
    df[c] = ref[c].values
print(f'{len(df):,} samples / {df.IS2_id.nunique():,} segments')

# ---------------------------------------------------------------- grouped folds (as cross_validate_model.py)
segs = df.lake_id.astype(str).unique()
rng = np.random.default_rng(a.seed)
df['fold'] = df.lake_id.astype(str).map(dict(zip(segs, rng.permutation(np.arange(len(segs)) % a.folds))))
print('fold sizes:', df.fold.value_counts().sort_index().to_list())

# ---------------------------------------------------------------- band ratio
X = np.log(df.B2 / df.B3).to_numpy()
X[(df.B3 <= MIN_REFL) | (df.B2 <= MIN_REFL) | ~np.isfinite(X)] = np.nan
df['bandratio_X'] = X
is_water = df.ndwi.to_numpy() > NDWI_THRESH
fittable = np.isfinite(X) & is_water & (df.depth.to_numpy() > 0)
print(f'band-ratio fit uses {fittable.sum():,} samples '
      f'(finite X, NDWI > {NDWI_THRESH}, depth > 0)')


def br_predict(coef, x, ndwi):
    """Quadratic in X, clipped at zero, and zero wherever NDWI says the pixel is not water."""
    d = np.polyval(coef, x)
    d[~np.isfinite(d)] = 0.0
    return np.where(ndwi > NDWI_THRESH, np.maximum(d, 0.0), 0.0)


coef_all = np.polyfit(X[fittable], df.depth.to_numpy()[fittable], 2)
df['br_pred_insample'] = br_predict(coef_all, X.copy(), df.ndwi.to_numpy())
print('band ratio, fitted on all samples:  D = %.4f X^2 + %.4f X + %.4f' % tuple(coef_all))

br_oof = np.full(len(df), np.nan)
coef_folds = []
for k in range(a.folds):
    te = (df.fold == k).to_numpy()
    m = fittable & ~te
    c = np.polyfit(X[m], df.depth.to_numpy()[m], 2)
    coef_folds.append(c.tolist())
    br_oof[te] = br_predict(c, X[te].copy(), df.ndwi.to_numpy()[te])
df['br_pred'] = br_oof
assert np.isfinite(df.br_pred).all(), 'band-ratio left some samples unpredicted'

# ---------------------------------------------------------------- random forest, same folds
preds = {}
for tag, feats in (('rf_pred', FEATS_ALL), ('rf16_pred', FEATS_16)):
    out = np.full(len(df), np.nan)
    print(f'\n{tag}: {len(feats)} predictors')
    for k in range(a.folds):
        te = (df.fold == k).to_numpy()
        rf = RandomForestRegressor(
            n_estimators=hp['n_estimators'], criterion=hp['criterion'],
            max_features=hp['max_features'], max_samples=hp['max_samples'],
            min_samples_leaf=hp['min_samples_leaf'], random_state=a.seed + k, n_jobs=-1)
        t0 = time.time()
        rf.fit(df.loc[~te, feats], df.loc[~te, 'depth'])
        out[te] = rf.predict(df.loc[te, feats])
        print(f'  fold {k}: MAE {np.abs(out[te] - df.depth.to_numpy()[te]).mean():.4f} '
              f'({time.time() - t0:.0f}s)', flush=True)
    preds[tag] = out
    df[tag] = out


# ---------------------------------------------------------------- statistics
def stats(obs, pred, mask):
    o, q = np.asarray(obs)[mask], np.asarray(pred)[mask]
    e = q - o
    return {'n': int(mask.sum()), 'MAE': float(np.abs(e).mean()),
            'RMSE': float(np.sqrt((e ** 2).mean())), 'bias': float(e.mean()),
            'r': float(np.corrcoef(o, q)[0, 1])}


obs = df.depth.to_numpy()
# the comparison set: samples ICESat-2 saw water in.  Dry ice is excluded because the band
# ratio is defined as zero there by construction, which would flatter it.
wet = obs > 0
# the thesis reported its MAE over wet samples that also pass an NDWI gate; keep both
wet_ndwi = wet & (df.ndwi.to_numpy() > 0.14)
summary = {}
for name, pred in [('random_forest_62', df.rf_pred), ('random_forest_16', df.rf16_pred),
                   ('band_ratio_oof', df.br_pred), ('band_ratio_insample', df.br_pred_insample)]:
    summary[name] = {'all_samples': stats(obs, pred, np.ones(len(df), bool)),
                     'water_only': stats(obs, pred, wet),
                     'water_ndwi_gated': stats(obs, pred, wet_ndwi)}
    s = summary[name]['water_only']
    print(f'  {name:22s} water-only  MAE {s["MAE"]:.3f}  RMSE {s["RMSE"]:.3f}  r {s["r"]:.3f}')

# ---------------------------------------------------------------- write
keep = ['IS2_id', 'S2_id', 'lat', 'xatc', 'depth', 'conf', 'ndwi', 'B2', 'B3', 'B4',
        'bandratio_X', 'fold', 'rf_pred', 'rf16_pred', 'br_pred', 'br_pred_insample']
out = df[keep].copy()
for c in out.columns:
    if out[c].dtype == np.float64:
        out[c] = out[c].astype(np.float32)
out['fold'] = out.fold.astype(np.int8)
out.to_parquet(f'{a.out}/ml_supplement_samples.parquet', index=False)

info = pd.read_parquet(a.info)
info[['conf_mean', 'n_scenes', 'basin', 'melt_season', 'max_depth', 'ndwi_thresh',
      'S2_id_first', 'lat', 'lon']].to_parquet(f'{a.out}/ml_supplement_segments.parquet')

shutil.copy(a.rfe, f'{a.out}/rfe_stats.csv')

json.dump({
    'ndwi_threshold': NDWI_THRESH,
    'band_ratio_definition': 'X = ln(B2 / B3); D = aX^2 + bX + c, clipped at 0, '
                             'and set to 0 where NDWI <= threshold',
    'band_ratio_coefficients_all_samples': coef_all.tolist(),
    'band_ratio_coefficients_by_fold': coef_folds,
    'features_62': FEATS_ALL,
    'features_16': FEATS_16,
    'features_19_rfe': cfg['rfe_selected_features_exploratory'],
    'hyperparameters': hp,
    'n_samples': int(len(df)), 'n_segments': int(df.IS2_id.nunique()),
    'folds': a.folds, 'seed': a.seed,
    'statistics': summary,
}, open(f'{a.out}/ml_supplement_meta.json', 'w'), indent=2)
print(f'\nwrote {a.out}/')
