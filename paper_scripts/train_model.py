"""Train a seeded, persistable replacement for the archived random-forest model.

Same hyperparameters as the published model (RF-dev.ipynb / Arndt & Fricker), with two
differences: random_state is set so the result is reproducible, and the model is persisted
with joblib compression instead of a bare pickle.
"""
import argparse, json, os, time
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--features', default='zenodo_build/greenland_training/training_features.parquet')
p.add_argument('--config',   default='zenodo_build/greenland_training/model_config.json')
p.add_argument('--out',      default='zenodo_build/models/rf_depth_v2.joblib')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--min-samples-leaf', type=int, default=1)
p.add_argument('--compress', type=int, default=3)
p.add_argument('--feature-set', default='all', choices=['all','rfe','rfe_noangle'])
a = p.parse_args()

cfg = json.load(open(a.config)); hp = cfg['hyperparameters']
import re as _re
_ANGLE = _re.compile(r'(SZA|VZA_|RAA_|PhA_)')
feats = (cfg['features'] if a.feature_set == 'all'
         else cfg['rfe_selected_features_exploratory'] if a.feature_set == 'rfe'
         else [f for f in cfg['rfe_selected_features_exploratory'] if not _ANGLE.match(f)])
df = pd.read_parquet(a.features)
X, y = df[feats], df[cfg['target']]
print(f'training on {len(df):,} rows x {len(feats)} features')
print(f'  n_estimators={hp["n_estimators"]} max_features={hp["max_features"]} '
      f'max_samples={hp["max_samples"]} min_samples_leaf={a.min_samples_leaf} seed={a.seed}')

t0 = time.time()
rf = RandomForestRegressor(
    n_estimators=hp['n_estimators'], criterion=hp['criterion'],
    max_features=hp['max_features'], max_samples=hp['max_samples'],
    min_samples_leaf=a.min_samples_leaf, random_state=a.seed, n_jobs=-1)
rf.fit(X, y)
print(f'  fitted in {time.time()-t0:.0f} s')

os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
joblib.dump(rf, a.out, compress=a.compress)
size = os.path.getsize(a.out) / 1e6
nodes = sum(e.tree_.node_count for e in rf.estimators_)
print(f'\nwrote {a.out}: {size:,.1f} MB  ({nodes:,} nodes across {len(rf.estimators_)} trees)')

pred = rf.predict(X)
print(f'  in-sample MAE {np.abs(pred - y).mean():.4f} m  (optimistic; see CV for an honest figure)')
