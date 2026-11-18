"""Recover true UTC Sentinel-2 acquisition times from the naive-local-time bug.

`IS2ML_utils.get_timediff` and `get_timediff_sonar` both do

    s2datetime = datetime.fromtimestamp(ms / 1e3)          # machine-local wall clock, naive
    t_s2       = s2datetime.replace(tzinfo=timezone.utc)    # relabelled UTC, not converted

so every stored time difference is the true one minus the machine's UTC offset at that
instant. The machine was on US/Pacific time, and `fromtimestamp` applies the offset in force
*at the timestamp being converted*, not at run time. The correction is therefore NOT a
constant:

    melt-season Greenland scenes (Jun-Sep)   -> PDT, UTC-7  -> add 25200 s
    Amery scenes (Dec-Feb, austral summer)   -> PST, UTC-8  -> add 28800 s

Verified against Sentinel-2 product ids, which carry the true sensing time:
  * NE Greenland lake HDF5: product S2A_MSIL2A_20210720T142941_... but `time_imagery` stored
    as 07:30:29Z - 6:59:12 early (PDT).
  * Amery: the stored differences imply acquisitions on 2019-01-02T04:19 and 2019-01-03T03:49
    once 8 h is added; the cached metadata for UTM zones 41/42 holds exactly
    S2B_..._20190102T041719_... and S2B_..._20190103T034629_... (+7 h misses both by an hour).

Inverting through the Pacific zone rules handles both cases without hard-coding either.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

PACIFIC = ZoneInfo('America/Los_Angeles')


def true_tdiff_sec(ref_epoch, tdiff_buggy_sec):
    """True time difference, given the reference epoch (correct UTC) and the buggy difference.

    `ref_epoch` is the ICESat-2 or sonar acquisition time as seconds since the Unix epoch;
    that side of the subtraction was always handled correctly. `ref_epoch + tdiff_buggy_sec`
    is therefore the Sentinel-2 acquisition as a Pacific wall-clock reading, which we
    reinterpret in its own zone and convert back to UTC.
    """
    ref_epoch = np.asarray(ref_epoch, dtype=float)
    tdiff = np.asarray(tdiff_buggy_sec, dtype=float)
    ref_b, tdiff_b = np.broadcast_arrays(ref_epoch, tdiff)
    out = np.empty(tdiff_b.shape, dtype=float)
    for i, (r, t) in enumerate(zip(ref_b.ravel(), tdiff_b.ravel())):
        wall = datetime.fromtimestamp(r + t, tz=timezone.utc).replace(tzinfo=None)
        out.ravel()[i] = wall.replace(tzinfo=PACIFIC).astimezone(timezone.utc).timestamp() - r
    return out.reshape(tdiff_b.shape)


def shift_sec(ref_epoch, tdiff_buggy_sec):
    """Just the correction that gets added, in seconds (25200 under PDT, 28800 under PST)."""
    return true_tdiff_sec(ref_epoch, tdiff_buggy_sec) - np.asarray(tdiff_buggy_sec, dtype=float)
