import numpy as np
import pandas as pd

from .config import (
    FEATURES, WINDOW, BASELINE, ALARM_FROM, ALARM_LEVEL, PERSIST, VIB_LIMIT, BAND_EDGES,
)


def health_index(df):
    """Mahalanobis distance of the smoothed sensors from the motor's own healthy baseline.

    The covariance comes from the raw baseline readings divided by the smoothing window,
    which is far more stable than estimating it from the heavily overlapping smoothed readings.
    Assumes time_step equals the row number.
    """
    raw = df[FEATURES].to_numpy()
    smooth = df[FEATURES].rolling(WINDOW, min_periods=1).mean().to_numpy()
    t = df["time_step"].to_numpy()
    base = raw[(t >= BASELINE[0]) & (t < BASELINE[1])]
    mu = base.mean(axis=0)
    inv_cov = np.linalg.inv(np.cov(base, rowvar=False) / WINDOW)
    diff = smooth - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))


def detect_fault(index, time_step):
    """First reading where the index has stayed above the alarm level for PERSIST readings."""
    above = pd.Series(index > ALARM_LEVEL).astype(int)
    alarm = (above.rolling(PERSIST).min() == 1).to_numpy()
    hits = time_step[(time_step >= ALARM_FROM) & alarm]
    return int(hits[0]) if len(hits) else None


def failure_time(df):
    """First reading where smoothed vibration reaches the failure limit."""
    vib = df["vibration_mm_s"].rolling(WINDOW, min_periods=1).mean().to_numpy()
    t = df["time_step"].to_numpy()
    hits = t[(vib >= VIB_LIMIT) & (t >= ALARM_FROM)]
    return int(hits[0]) if len(hits) else None


def health_score(index, d_fail):
    """100 while the index is below the alarm level, falling linearly to 0 at the failure level."""
    fraction = np.clip((index - ALARM_LEVEL) / (d_fail - ALARM_LEVEL), 0, 1)
    return 100 * (1 - fraction)


def health_status(score):
    healthy, warning, degraded = BAND_EDGES
    return np.select(
        [score > healthy, score > warning, score > degraded],
        ["Healthy", "Warning", "Degraded"],
        default="Critical",
    )