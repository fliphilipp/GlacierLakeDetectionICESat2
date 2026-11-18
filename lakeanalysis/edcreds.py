"""Earthdata Login credentials for the NSIDC download endpoints.

`lakeanalysis.nsidc` and `lakeanalysis.S2` used to import this from a top-level `ed` package,
which is gitignored and therefore absent from every clone and from the archived release. Because
the import sat at module level, `import lakeanalysis.nsidc` raised ModuleNotFoundError before any
function ran, so neither module could be imported at all outside the original working copy.

Resolution order, first hit wins:

  1. the local, gitignored ``ed/edcreds.py``, if it is importable, so an existing working copy
     behaves exactly as before;
  2. the environment: ``EARTHDATA_USERNAME``, ``EARTHDATA_PASSWORD``, ``EARTHDATA_EMAIL``.

Register for an Earthdata Login at https://urs.earthdata.nasa.gov/users/new. No credential is
stored in this repository, and none is read from a file other than the gitignored one above.
"""
import os

__all__ = ['getedcreds']

_ENV_VARS = ('EARTHDATA_USERNAME', 'EARTHDATA_PASSWORD', 'EARTHDATA_EMAIL')


def getedcreds():
    """Return ``(username, password, email)`` for Earthdata Login.

    Raises
    ------
    RuntimeError
        If no local ``ed.edcreds`` module is importable and the three environment
        variables are not all set. The message names what to set and where to register.
    """
    try:
        from ed.edcreds import getedcreds as _local  # noqa: PLC0415 - optional local override
    except ImportError:
        pass
    else:
        return _local()

    uid, pwd, email = (os.environ.get(v) for v in _ENV_VARS)
    if not (uid and pwd and email):
        missing = ', '.join(v for v in _ENV_VARS if not os.environ.get(v))
        raise RuntimeError(
            'Earthdata Login credentials are not available: {} not set. Register at '
            'https://urs.earthdata.nasa.gov/users/new, then export {}.'.format(
                missing, ', '.join(_ENV_VARS)))
    return uid, pwd, email
