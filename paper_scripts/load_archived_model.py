"""Load the archived 13.5 GB random-forest pickle and re-persist it compactly.

The archived model is a `__main__.MeanEnsembleRegressor` - a hand-written wrapper that averages
several RandomForestRegressor instances. That class only ever existed in a notebook, so the
pickle cannot be opened by anyone who does not recreate it first; the shim below does that.

Memory: the pickle is ~13.5 GB and unpickles to at least that much. An address-space limit is
set so that an overrun fails cleanly instead of driving the machine into swap. Raise --max-gb
if you have headroom, lower it if you want to fail faster.

    python zenodo_build/scripts/load_archived_model.py --max-gb 40
"""
import argparse
import os
import resource
import sys
import time

import numpy as np


class MeanEnsembleRegressor:
    """Reconstruction of the class the archived model was pickled against.

    Must match the original's module (`__main__`) and name for unpickling to succeed.
    Only the attribute layout matters for loading; `predict` is reproduced from RF-dev.ipynb.
    """

    def __init__(self, regressors=None):
        self.regressors = regressors or []

    def fit(self, X, y):
        for r in self.regressors:
            r.fit(X, y)
        return self

    def predict(self, X):
        return np.mean(np.column_stack([r.predict(X) for r in self.regressors]), axis=1)


def describe(model):
    regs = getattr(model, 'regressors', [model])
    print(f'  sub-regressors : {len(regs)}')
    total_trees = total_nodes = 0
    for i, r in enumerate(regs):
        trees = getattr(r, 'estimators_', [])
        nodes = sum(t.tree_.node_count for t in trees)
        total_trees += len(trees)
        total_nodes += nodes
        if i < 3:
            print(f'    [{i}] {type(r).__name__}: {len(trees)} trees, {nodes:,} nodes, '
                  f'max_features={getattr(r, "max_features", "?")}, '
                  f'max_samples={getattr(r, "max_samples", "?")}')
    print(f'  totals         : {total_trees:,} trees, {total_nodes:,} nodes')
    feats = getattr(regs[0], 'feature_names_in_', None)
    if feats is not None:
        print(f'  features       : {len(feats)}')
    return regs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pickle',
                   default='GlacierLakeDetectionICESat2/GlacierLakeIS2ML/models/ensemble_regressor_test.pkl')
    p.add_argument('--out-dir', default='zenodo_build/models')
    p.add_argument('--max-gb', type=float, default=40.0)
    p.add_argument('--compress', type=int, default=3)
    p.add_argument('--keep', type=int, default=0,
                   help='re-save only the first N sub-regressors (0 = all)')
    args = p.parse_args()

    limit = int(args.max_gb * 1e9)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    print(f'address-space limit: {args.max_gb:.0f} GB')
    print(f'loading {args.pickle} ({os.path.getsize(args.pickle)/1e9:.2f} GB)...')

    import pickle
    sys.modules['__main__'].MeanEnsembleRegressor = MeanEnsembleRegressor
    t0 = time.time()
    try:
        with open(args.pickle, 'rb') as f:
            model = pickle.load(f)
    except MemoryError:
        sys.exit(f'MemoryError after {time.time()-t0:.0f}s - raise --max-gb or free memory')
    print(f'loaded in {time.time()-t0:.0f} s\n')

    regs = describe(model)

    import joblib
    os.makedirs(args.out_dir, exist_ok=True)
    keep = regs[:args.keep] if args.keep else regs
    out = os.path.join(args.out_dir, f'rf_depth_archived_{len(keep)}x.joblib')
    joblib.dump(MeanEnsembleRegressor(list(keep)), out, compress=args.compress)
    print(f'\nwrote {out}: {os.path.getsize(out)/1e9:.2f} GB '
          f'(from {os.path.getsize(args.pickle)/1e9:.2f} GB, '
          f'{len(keep)}/{len(regs)} sub-regressors, joblib compress={args.compress})')


if __name__ == '__main__':
    main()
