"""10-fold cross-validation of the depth model, grouped by ICESat-2 lake segment.

Grouping by segment rather than by row is essential: measurements 5 m apart along a track are
strongly autocorrelated, and a random split leaks them between folds. This reproduces the
protocol described in the paper, with the fold assignment and the forests both seeded.
"""
import argparse, json, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--features', default='zenodo_build/greenland_training/training_features.parquet')
p.add_argument('--config',   default='zenodo_build/greenland_training/model_config.json')
p.add_argument('--out',      default='zenodo_build/models/cv_stats_v2.csv')
p.add_argument('--folds', type=int, default=10)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--min-samples-leaf', type=int, default=1)
p.add_argument('--feature-set', default='all', choices=['all', 'rfe', 'rfe_noangle'],
               help='all = 62 predictors (published); rfe = the 19 selected by recursive '
                    'feature elimination; rfe_noangle = those 19 minus the three angle terms, '
                    'which removes the dependency on Sentinel-2 L1C metadata entirely')
a = p.parse_args()

cfg = json.load(open(a.config)); hp = cfg['hyperparameters']
import re as _re
_ANGLE = _re.compile(r'(SZA|VZA_|RAA_|PhA_)')
if a.feature_set == 'all':
    feats = cfg['features']
elif a.feature_set == 'rfe':
    feats = cfg['rfe_selected_features_exploratory']
else:
    feats = [f for f in cfg['rfe_selected_features_exploratory'] if not _ANGLE.match(f)]
print(f'feature set: {a.feature_set} ({len(feats)} predictors)')
df = pd.read_parquet(a.features)
segs = df.lake_id.astype(str).unique()
rng = np.random.default_rng(a.seed)
fold_of = dict(zip(segs, rng.permutation(np.arange(len(segs)) % a.folds)))
df['_fold'] = df.lake_id.astype(str).map(fold_of)
print(f'{len(df):,} rows / {len(segs):,} segments / {a.folds} folds, seed {a.seed}')

rows = []
for k in range(a.folds):
    te, tr = df._fold == k, df._fold != k
    rf = RandomForestRegressor(
        n_estimators=hp['n_estimators'], criterion=hp['criterion'],
        max_features=hp['max_features'], max_samples=hp['max_samples'],
        min_samples_leaf=a.min_samples_leaf, random_state=a.seed + k, n_jobs=-1)
    t0 = time.time(); rf.fit(df.loc[tr, feats], df.loc[tr, 'depth'])
    pred = rf.predict(df.loc[te, feats]); obs = df.loc[te, 'depth'].values
    err = pred - obs
    nz = obs > 0
    rows.append(dict(fold=k, n_test=int(te.sum()), n_segments=int(df.loc[te, 'lake_id'].nunique()),
                     MAE=np.abs(err).mean(), RMSE=np.sqrt((err**2).mean()), bias=err.mean(),
                     MAE_water=np.abs(err[nz]).mean(), bias_water=err[nz].mean(),
                     r=np.corrcoef(obs, pred)[0, 1], secs=time.time() - t0))
    print(f'  fold {k}: MAE {rows[-1]["MAE"]:.4f}  bias {rows[-1]["bias"]:+.4f}  '
          f'r {rows[-1]["r"]:.4f}  ({rows[-1]["secs"]:.0f}s)')

cv = pd.DataFrame(rows); cv.to_csv(a.out, index=False)
print(f'\nwrote {a.out}')
for c in ['MAE', 'RMSE', 'bias', 'MAE_water', 'r']:
    m, s = cv[c].mean(), cv[c].std()
    print(f'  {c:10s} {m:7.4f} +/- {s:.4f}   (95% CI on the mean: +/- {1.96*s/np.sqrt(len(cv)):.4f})')
print('\npublished for comparison: MAE 0.33 +/- 0.01 m, bias 0.03 +/- 0.01 m')
