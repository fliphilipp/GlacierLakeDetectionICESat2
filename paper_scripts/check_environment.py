"""Smoke-test the `icelakes` environment against every code path in this project.

Checks that the imports resolve, that the version pins hold, and that the operations the
project actually depends on still behave - rather than just that packages are installed.

    conda activate icelakes
    python zenodo_build/scripts/check_environment.py
"""
import importlib
import sys
import traceback
import warnings

RESULTS = []


def check(name):
    """Decorator turning a function into a reported pass/fail check."""
    def wrap(fn):
        try:
            detail = fn()
            RESULTS.append((True, name, detail or ''))
        except Exception as e:
            RESULTS.append((False, name, f'{type(e).__name__}: {e}'))
        return fn
    return wrap


# --- imports -----------------------------------------------------------------
MODULES = [
    'numpy', 'pandas', 'scipy', 'sklearn', 'geopandas', 'shapely', 'pyproj', 'fiona',
    'rasterio', 'contextily', 'geopy', 'h5py', 'pyarrow', 'lxml', 'duckdb', 'requests',
    'ee', 'boto3', 'rsa', 'matplotlib', 'cmcrameri', 'PIL', 'statsmodels', 'tabulate',
    'IPython', 'psutil', 'streamlit',
]
for _m in MODULES:
    def _f(m=_m):
        mod = importlib.import_module(m)
        return getattr(mod, '__version__', 'ok')
    check(f'import {_m}')(_f)

for _m in ['earthaccess', 'harmony']:
    def _f(m=_m):
        mod = importlib.import_module(m)
        return getattr(mod, '__version__', 'ok')
    check(f'import {_m} (forward-looking)')(_f)


# --- version pins that matter ------------------------------------------------
@check('numpy < 2 (NEP 50 promotion)')
def _():
    import numpy as np
    assert int(np.__version__.split('.')[0]) < 2, f'numpy {np.__version__}'
    return np.__version__


@check('pandas < 3 (fillna(method=) still present)')
def _():
    import pandas as pd
    assert int(pd.__version__.split('.')[0]) < 3, f'pandas {pd.__version__}'
    return pd.__version__


@check('scikit-learn 1.3.x (archived pickle compat)')
def _():
    import sklearn
    assert sklearn.__version__.startswith('1.3.'), f'sklearn {sklearn.__version__}'
    return sklearn.__version__


# --- behaviours the project relies on ----------------------------------------
@check('detection.py:183 ffill pattern still works')
def _():
    import numpy as np
    import pandas as pd
    s = pd.Series([1.0, np.nan, np.nan, 4.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        out = s.fillna(method='ffill')
    assert out.tolist() == [1.0, 1.0, 1.0, 4.0]
    return 'ffill ok (FutureWarning expected)'


@check('float16 HDF5 round-trip (photon_data dtypes)')
def _():
    import io
    import h5py
    import numpy as np
    buf = io.BytesIO()
    a = np.array([1.5, 2.5, np.nan], dtype=np.float16)
    with h5py.File(buf, 'w') as f:
        f.create_dataset('x', data=a, compression='gzip')
    with h5py.File(buf, 'r') as f:
        b = f['x'][()]
    assert b.dtype == np.float16
    return 'float16 preserved'


@check('Parquet round-trip with categorical + zstd')
def _():
    import io
    import pandas as pd
    df = pd.DataFrame({'lake_id': pd.Categorical(['a', 'a', 'b']), 'depth': [1.0, 2.0, 3.0]})
    buf = io.BytesIO()
    df.to_parquet(buf, compression='zstd', index=False)
    back = pd.read_parquet(buf)
    assert len(back) == 3 and back.depth.sum() == 6.0
    return 'zstd + dictionary encoding ok'


@check('geopandas reprojection 4326 -> 3413')
def _():
    import geopandas as gpd
    g = gpd.GeoDataFrame(geometry=gpd.points_from_xy([-47.77], [69.33]), crs='EPSG:4326')
    p = g.to_crs('EPSG:3413')
    x, y = p.geometry.x.iloc[0], p.geometry.y.iloc[0]
    assert -200000 < x < 0 and -2400000 < y < -2100000, f'({x:.0f}, {y:.0f})'
    return f'({x:.0f}, {y:.0f}) m'


@check('scipy.signal.find_peaks with prominence (FLUID core)')
def _():
    import numpy as np
    from scipy.signal import find_peaks
    y = np.array([0, 1, 0, 3, 0, 1, 0], dtype=float)
    pk, props = find_peaks(y, prominence=0.5)
    assert list(pk) == [1, 3, 5]
    return f'{len(pk)} peaks'


@check('sklearn KDTree kNN (FLUID photon density)')
def _():
    import numpy as np
    from sklearn.neighbors import KDTree
    rng = np.random.default_rng(0)
    pts = rng.random((200, 2))
    d, i = KDTree(pts).query(pts[:5], k=4)
    assert d.shape == (5, 4)
    return 'exact kNN ok'


@check('rasterio read of a project chip')
def _():
    import glob
    import rasterio
    hits = glob.glob('GLD3_complete/GrIS/detection_imagery/*/*_imagery.tif')
    if not hits:
        return 'skipped (no chip found)'
    with rasterio.open(hits[0]) as r:
        a = r.read(1)
    return f'{a.shape[0]}x{a.shape[1]} {r.crs}'


@check('earthengine-api importable (auth checked separately)')
def _():
    import ee
    assert hasattr(ee, 'Initialize')
    return f'{ee.__version__} - run `earthengine authenticate` and set EE_PROJECT'


if __name__ == '__main__':
    print(f'python {sys.version.split()[0]}\n')
    width = max(len(n) for _, n, _ in RESULTS)
    for ok, name, detail in RESULTS:
        print(f'  {"PASS" if ok else "FAIL"}  {name:<{width}}  {detail}')
    n_fail = sum(1 for ok, _, _ in RESULTS if not ok)
    print(f'\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed')
    sys.exit(1 if n_fail else 0)
