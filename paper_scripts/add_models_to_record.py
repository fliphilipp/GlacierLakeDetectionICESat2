"""Add the two trained random-forest models to the Zenodo training-data record.

The models and the training matrix belong in one record: a model cannot be used without
the exact feature list and column order that `training_features.parquet` defines, and
`model_config.json` is the document that ties them together.

What this copies in (flat, no subdirectories - Zenodo presents files as a flat list):

  rf_depth_v2.joblib            62-predictor production model; every depth prediction
                                in the paper comes from this one
  rf_depth_rfe_noangle.joblib   16-predictor model; the reduced set that needs no
                                Sentinel-2 L1C angle metadata, used in Fig. 2d
  cv_stats_*.csv                10-fold grouped cross-validation, one file per model
                                variant tried, including the three not shipped
  feature_importance_rfe16.csv  impurity and permutation importance for the 16 predictors

Before copying, each model is loaded with a bare `joblib.load` and its `feature_names_in_`
is checked against the feature list this record ships. A mismatch aborts the build: a model
whose feature order disagrees with the training matrix is worse than no model at all.

    python zenodo_build/scripts/add_models_to_record.py --dry-run
    python zenodo_build/scripts/add_models_to_record.py
"""
import argparse
import csv
import json
import os
import shutil
import sys

# ---------------------------------------------------------------- what to ship

# model file -> (key in model_config.json holding its feature list, one-line description)
MODELS = {
    'rf_depth_v2.joblib': (
        'features',
        'Production model. RandomForestRegressor on all 62 engineered predictors. '
        'Used for every random-forest depth prediction reported in the paper.'),
    'rf_depth_rfe_noangle.joblib': (
        'rfe_noangle_features',
        'Reduced model. RandomForestRegressor on the 16 predictors that survive recursive '
        'feature elimination once the four angle-term groups are excluded, so it needs no '
        'Sentinel-2 Level-1C metadata. Used for the predictor panel in Fig. 2d.'),
}

# cross-validation tables: filename -> (shipped model or None, description)
CV_STATS = {
    'cv_stats_v2.csv': (
        'rf_depth_v2.joblib', 'all 62 predictors'),
    'cv_stats_rfe.csv': (
        None, '19 predictors, the full recursive-feature-elimination set (model not shipped)'),
    'cv_stats_rfe_noangle.csv': (
        'rf_depth_rfe_noangle.joblib', '16 predictors, RFE set minus the angle terms'),
    'cv_stats_rfe_noangle_msl5.csv': (
        None, '16 predictors, min_samples_leaf=5 (model not shipped)'),
    'cv_stats_rfe_noangle_msl20.csv': (
        None, '16 predictors, min_samples_leaf=20 (model not shipped)'),
}

IMPORTANCE = 'feature_importance_rfe16.csv'

ANGLE_PREFIXES = ('SZA', 'VZA_', 'RAA_', 'PhA_')

# ------------------------------------------------------- data-dictionary rows

CV_COLUMNS = [
    ('fold', 'int', '', 'Cross-validation fold index, 0-9.'),
    ('n_test', 'int', '', 'Number of held-out samples in this fold.'),
    ('n_segments', 'int', '', 'Number of held-out ICESat-2 segments in this fold. Folds are '
                              'grouped by segment (IS2_id), never split within one.'),
    ('MAE', 'float', 'm', 'Mean absolute error over all held-out samples, water and non-water.'),
    ('RMSE', 'float', 'm', 'Root-mean-square error over all held-out samples.'),
    ('bias', 'float', 'm', 'Mean signed error (prediction minus ICESat-2 label).'),
    ('MAE_water', 'float', 'm', 'Mean absolute error over held-out samples with depth > 0 only.'),
    ('bias_water', 'float', 'm', 'Mean signed error over held-out samples with depth > 0 only.'),
    ('r', 'float', '', 'Pearson correlation between prediction and label on the held-out fold.'),
    ('secs', 'float', 's', 'Wall-clock fit time for this fold.'),
]

IMPORTANCE_COLUMNS = [
    ('feature', 'string', '', 'Predictor name, matching a column of training_features.parquet.'),
    ('rfe_rank', 'int', '', 'Elimination rank within the 16-predictor set: 1 is the last '
                            'predictor standing under recursive feature elimination.'),
    ('imp_impurity', 'float', '', "Mean decrease in impurity (scikit-learn's "
                                  'feature_importances_). Biased toward high-cardinality '
                                  'predictors; reported for completeness, not for ranking.'),
    ('imp_perm_dmae', 'float', 'm', 'Permutation importance: the increase in held-out MAE when '
                                    'this predictor is shuffled. This is the honest ranking.'),
    ('imp_perm_sd', 'float', 'm', 'Standard deviation of imp_perm_dmae across permutation repeats.'),
    ('imp_perm_frac', 'float', '', 'imp_perm_dmae as a fraction of the summed permutation '
                                   'importance of all 16 predictors.'),
]


def load_and_check(path, expected_features):
    """Load a model with a bare joblib.load and verify it matches the shipped feature list."""
    import joblib
    model = joblib.load(path)
    kind = f'{type(model).__module__}.{type(model).__name__}'
    got = list(getattr(model, 'feature_names_in_', []))
    if not kind.startswith('sklearn.'):
        raise SystemExit(
            f'ABORT {os.path.basename(path)}: top-level object is {kind}, not a scikit-learn '
            f'estimator. A model that needs a custom class to unpickle cannot be archived.')
    if got != expected_features:
        only_model = sorted(set(got) - set(expected_features))
        only_cfg = sorted(set(expected_features) - set(got))
        raise SystemExit(
            f'ABORT {os.path.basename(path)}: feature_names_in_ does not match the shipped '
            f'feature list ({len(got)} vs {len(expected_features)}).\n'
            f'  only in model: {only_model}\n  only in config: {only_cfg}\n'
            f'  (order matters: a reordered list fails this check too)')
    return kind, len(got), len(getattr(model, 'estimators_', []))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models-dir', default='zenodo_build/models')
    p.add_argument('--record-dir', default='zenodo_build/greenland_training')
    p.add_argument('--dry-run', action='store_true', help='check and report, write nothing')
    a = p.parse_args()

    cfg_path = os.path.join(a.record_dir, 'model_config.json')
    cfg = json.load(open(cfg_path))

    # Derive the 16-predictor list the same way train_model.py does, so the two cannot drift.
    cfg['rfe_noangle_features'] = [f for f in cfg['rfe_selected_features_exploratory']
                                   if not f.startswith(ANGLE_PREFIXES)]

    print(f'feature lists: all={len(cfg["features"])} '
          f'rfe={len(cfg["rfe_selected_features_exploratory"])} '
          f'rfe_noangle={len(cfg["rfe_noangle_features"])}\n')

    # --- verify every model before copying anything -------------------------
    checked = {}
    for fname, (feat_key, _) in MODELS.items():
        src = os.path.join(a.models_dir, fname)
        kind, n_feat, n_trees = load_and_check(src, cfg[feat_key])
        size_mb = os.path.getsize(src) / 1e6
        checked[fname] = dict(kind=kind, n_features=n_feat, n_estimators=n_trees,
                              size_mb=round(size_mb, 1))
        print(f'  OK {fname}: {kind}, {n_feat} features, {n_trees} trees, {size_mb:,.1f} MB')

    import numpy, pandas, sklearn, joblib
    env = {'python': sys.version.split()[0], 'scikit-learn': sklearn.__version__,
           'numpy': numpy.__version__, 'pandas': pandas.__version__, 'joblib': joblib.__version__}
    print(f'\n  written under: {env}')

    # --- rebuild the parts of model_config.json that describe the models ----
    cfg['models'] = {
        fname: {'description': desc,
                'feature_list_key': feat_key,
                'n_features': checked[fname]['n_features'],
                'n_estimators': checked[fname]['n_estimators'],
                'size_mb': checked[fname]['size_mb'],
                'cv_stats': next((cv for cv, (m, _) in CV_STATS.items() if m == fname), None)}
        for fname, (feat_key, desc) in MODELS.items()
    }
    cfg['persistence'] = {
        'format': 'joblib.dump(..., compress=3)',
        'load': "joblib.load('rf_depth_v2.joblib') - a plain sklearn estimator, no custom "
                'class and no shim required',
        'written_with': env,
        'compatibility': 'scikit-learn does not guarantee that a pickled estimator loads across '
                         'minor versions. These files were written under scikit-learn '
                         f'{env["scikit-learn"]}; use that series to load them. On a newer '
                         'scikit-learn, refit instead: train_model.py reproduces either model '
                         'from training_features.parquet in about 50 s.',
    }
    cfg['feature_selection'] = (
        'The production model (rf_depth_v2.joblib) uses all 62 engineered predictors, in the '
        'column order of training_features.parquet, verified against feature_names_in_. '
        'Recursive feature elimination was run as an exploration and selected 19 predictors '
        '(rfe_selected_features_exploratory); that reduced set was NOT used for the published '
        'depth predictions. Dropping the four angle-term groups from it leaves the 16 '
        'predictors in rfe_noangle_features, which is what rf_depth_rfe_noangle.joblib uses.')
    cfg.pop('model_class', None)   # described the superseded 13.5 GB notebook pickle
    cfg['provenance'] = (
        'Both models were fitted by zenodo_build/scripts/train_model.py with random_state=42. '
        'They replace an earlier 13.5 GB pickle whose top-level object was a notebook-local '
        'wrapper class; that file is not part of this record and is not needed to reproduce '
        'any result in the paper. See cv_stats_v2.csv for the published-performance comparison.')

    # --- data-dictionary rows for the new tables ----------------------------
    dd_path = os.path.join(a.record_dir, 'data_dictionary.csv')
    existing = list(csv.DictReader(open(dd_path)))
    fields = list(existing[0].keys())
    new_rows = []
    for cv_name, (model, what) in CV_STATS.items():
        tag = f'{what}; scored model: {model}' if model else what
        for var, dtype, units, desc in CV_COLUMNS:
            new_rows.append({'file': cv_name, 'variable': var, 'dtype': dtype,
                             'units': units, 'description': f'{desc} [{tag}]'})
    for var, dtype, units, desc in IMPORTANCE_COLUMNS:
        new_rows.append({'file': IMPORTANCE, 'variable': var, 'dtype': dtype,
                         'units': units, 'description': desc})

    already = {r['file'] for r in existing}
    new_rows = [r for r in new_rows if r['file'] not in already]

    copies = ([(os.path.join(a.models_dir, f), os.path.join(a.record_dir, f)) for f in MODELS]
              + [(os.path.join(a.models_dir, f), os.path.join(a.record_dir, f)) for f in CV_STATS]
              + [(os.path.join(a.models_dir, IMPORTANCE),
                  os.path.join(a.record_dir, IMPORTANCE))])

    print(f'\nwould copy {len(copies)} files, add {len(new_rows)} data-dictionary rows, '
          f'rewrite model_config.json')
    if a.dry_run:
        print('\n--dry-run: nothing written')
        return

    for src, dst in copies:
        shutil.copy2(src, dst)
    with open(dd_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing + new_rows)
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')

    total = sum(os.path.getsize(os.path.join(a.record_dir, f))
                for f in os.listdir(a.record_dir)
                if os.path.isfile(os.path.join(a.record_dir, f)))
    print(f'\nrecord now: {len(os.listdir(a.record_dir))} files, {total/1e6:,.0f} MB')


if __name__ == '__main__':
    main()
