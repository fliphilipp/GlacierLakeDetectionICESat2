"""Feature importance for the 16-predictor (RFE, no-angle) random forest.

Two measures are written, because they answer different questions and the honest one is the
more expensive one:

  imp_impurity    mean decrease in impurity, read straight off the archived model. Free, but
                  computed on the training data and biased towards predictors that take many
                  distinct values - exactly the situation here, where every predictor is a
                  continuous reflectance.

  imp_perm        permutation importance measured on lakes the model never saw. A fresh model
                  is fitted on 90% of the lakes (split by lake_id, so no lake straddles the
                  split), then each predictor is shuffled in the held-out 10% and the increase
                  in MAE is recorded. This is the number to quote in a paper.

Both are reported so the two can be compared; `--which` picks the one the figure uses.

    python zenodo_build/scripts/feature_importance.py
"""
import argparse, json, os, re, time

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--features', default='zenodo_build/greenland_training/training_features.parquet')
p.add_argument('--config',   default='zenodo_build/greenland_training/model_config.json')
p.add_argument('--model',    default='zenodo_build/models/rf_depth_rfe_noangle_msl5.joblib')
p.add_argument('--out',      default='zenodo_build/models/feature_importance_rfe16.csv')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--min-samples-leaf', type=int, default=5)
p.add_argument('--n-repeats', type=int, default=5)
p.add_argument('--holdout-frac', type=float, default=0.1)
a = p.parse_args()

cfg = json.load(open(a.config))
ANGLE = re.compile(r'(SZA|VZA_|RAA_|PhA_)')
feats = [f for f in cfg['rfe_selected_features_exploratory'] if not ANGLE.match(f)]
rfe_rank = {f: i + 1 for i, f in enumerate(cfg['rfe_selected_features_exploratory'])}

df = pd.read_parquet(a.features, columns=['lake_id', cfg['target']] + feats)
target = cfg['target']
print(f'{len(df):,} rows x {len(feats)} predictors, {df.lake_id.nunique():,} lakes')

# ---------------------------------------------------------------- mean decrease in impurity
rf_archived = joblib.load(a.model)
assert list(rf_archived.feature_names_in_) == feats, 'archived model has a different feature set'
imp_impurity = pd.Series(rf_archived.feature_importances_, index=feats)
del rf_archived

# ------------------------------------------------- permutation importance on unseen lakes
rng = np.random.default_rng(a.seed)
lakes = df.lake_id.unique()
held = set(rng.choice(lakes, size=int(round(len(lakes) * a.holdout_frac)), replace=False))
is_held = df.lake_id.isin(held).to_numpy()
Xtr, ytr = df.loc[~is_held, feats], df.loc[~is_held, target]
Xte, yte = df.loc[is_held, feats].to_numpy(), df.loc[is_held, target].to_numpy()
print(f'  train {len(Xtr):,} rows / {len(lakes) - len(held):,} lakes   '
      f'held out {len(Xte):,} rows / {len(held):,} lakes')

hp = cfg['hyperparameters']
t0 = time.time()
rf = RandomForestRegressor(
    n_estimators=hp['n_estimators'], criterion=hp['criterion'],
    max_features=hp['max_features'], max_samples=hp['max_samples'],
    min_samples_leaf=a.min_samples_leaf, random_state=a.seed, n_jobs=-1).fit(Xtr, ytr)
base = np.abs(rf.predict(Xte) - yte).mean()
print(f'  fitted in {time.time()-t0:.0f} s; held-out MAE {base:.4f} m')

# Permute one column at a time in place, so no copy of the held-out matrix is kept around.
perm = np.zeros((len(feats), a.n_repeats))
for j, f in enumerate(feats):
    keep = Xte[:, j].copy()
    for k in range(a.n_repeats):
        Xte[:, j] = rng.permutation(keep)
        perm[j, k] = np.abs(rf.predict(Xte) - yte).mean() - base
    Xte[:, j] = keep
    print(f'  {f:<20s} dMAE {perm[j].mean():+.4f} +/- {perm[j].std(ddof=1):.4f} m')

out = pd.DataFrame({
    'feature': feats,
    'rfe_rank': [rfe_rank[f] for f in feats],
    'imp_impurity': imp_impurity[feats].to_numpy(),
    'imp_perm_dmae': perm.mean(axis=1),
    'imp_perm_sd': perm.std(axis=1, ddof=1),
}).sort_values('imp_perm_dmae', ascending=False)
out['imp_perm_frac'] = out.imp_perm_dmae / out.imp_perm_dmae.sum()

os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
out.to_csv(a.out, index=False)
print(f'\nheld-out baseline MAE {base:.4f} m; wrote {a.out}')
print(out.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
