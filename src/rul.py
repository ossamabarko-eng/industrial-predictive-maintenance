import numpy as np
import pandas as pd

from .config import HISTORY, STEP, MAX_RUL


def predict_rul(index, now, d_fail, degree):
    """Fit the recent index trend, extrapolate to the failure level, return readings left."""
    x = np.arange(-HISTORY, 1)
    y = index[now - HISTORY: now + 1]
    coef = np.polyfit(x, y, degree)
    coef[-1] -= d_fail
    roots = np.roots(coef)
    roots = roots[np.isreal(roots)].real
    future = roots[roots > 0]
    return min(future.min(), MAX_RUL) if len(future) else np.nan


def rul_series(index, first, last, d_fail):
    """RUL estimates every STEP readings from `first` up to (not including) `last`.

    Three estimators: a straight-line fit, a parabola fit, and the earlier (more
    conservative) of the two. When there is no usable trend, the previous estimate
    is kept, one step closer to failure.
    """
    rows, previous = [], {}
    for now in range(first, last, STEP):
        p1 = predict_rul(index, now, d_fail, 1)
        p2 = predict_rul(index, now, d_fail, 2)
        preds = {"linear": p1, "quadratic": p2, "conservative": np.minimum(p1, p2)}
        row = {"time_step": now}
        for method, pred in preds.items():
            if np.isnan(pred) and method in previous:
                pred = max(previous[method] - STEP, 0)
            previous[method] = pred
            row[method] = pred
        rows.append(row)
    return pd.DataFrame(rows)