import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import (
    RUNS_DIR, RESULTS_DIR, FEATURES, WINDOW, BASELINE, PERSIST, ALARM_FROM,
    MIN_AFTER_DETECTION, VIB_LIMIT,
)
from src.health import health_index, detect_fault, failure_time
from src.rul import rul_series

# =====================================================================
# 1. Fault detection: how early does each detector raise an alarm?
# =====================================================================


def detector_scores(df):
    """Higher score = more abnormal, for three different detectors."""
    t = df["time_step"].to_numpy()
    smooth = df[FEATURES].rolling(WINDOW, min_periods=1).mean()
    base = (t >= BASELINE[0]) & (t < BASELINE[1])

    scaler = StandardScaler().fit(smooth[base])
    forest = IsolationForest(n_estimators=200, random_state=42)
    forest.fit(scaler.transform(smooth[base]))

    return {
        "Vibration threshold": smooth["vibration_mm_s"].to_numpy(),
        "Isolation Forest": -forest.score_samples(scaler.transform(smooth)),
        "Health index (Mahalanobis)": health_index(df),
    }


rows = []
for path in sorted(RUNS_DIR.glob("run_*.csv")):
    df = pd.read_csv(path)
    t = df["time_step"].to_numpy()
    start = int(t[(df["status"] == "Degrading").to_numpy()][0])
    healthy = (t >= ALARM_FROM) & (t < start)
    after = t >= start

    scores = detector_scores(df)
    for method, score in scores.items():
        # An alarm needs PERSIST readings in a row above the threshold (rolling minimum above it).
        # Best case: the lowest threshold that gives zero false alarms on this motor's healthy readings.
        rolling_min = pd.Series(score).rolling(PERSIST).min().to_numpy()
        threshold = np.nanmax(rolling_min[healthy])
        hits = t[after & (rolling_min > threshold)]
        alarm = int(hits[0]) if len(hits) else None
        rows.append({"run": path.stem, "method": method, "degradation_start": start, "alarm": alarm,
                     "level_at_alarm": df.loc[alarm, "degradation_level"] if alarm is not None else np.nan})

    # The rule the pipeline really uses: one fixed alarm level for every motor
    alarm = detect_fault(scores["Health index (Mahalanobis)"], t)
    rows.append({"run": path.stem, "method": "Health index, fixed alarm level (deployed)",
                 "degradation_start": start, "alarm": alarm,
                 "level_at_alarm": df.loc[alarm, "degradation_level"] if alarm is not None else np.nan})

det = pd.DataFrame(rows)
det["false_alarm"] = det["alarm"].notna() & (det["alarm"] < det["degradation_start"])
det["missed"] = det["alarm"].isna()
det["delay"] = det["alarm"] - det["degradation_start"]
det.to_csv(RESULTS_DIR / "detector_comparison.csv", index=False)

table = []
for method, g in det.groupby("method", sort=False):
    ok = g[~g["false_alarm"] & ~g["missed"]]
    table.append({
        "method": method,
        "false_alarms": f"{int(g['false_alarm'].sum())}/{len(g)}",
        "missed": f"{int(g['missed'].sum())}/{len(g)}",
        "median_delay": ok["delay"].median(),
        "worst_delay": ok["delay"].max(),
        "mean_level_at_alarm": ok["level_at_alarm"].mean(),
    })

print("FAULT DETECTION")
print("Delay = readings from the start of degradation to the first alarm (smaller is better).")
print("Level at alarm = true degradation level (0 to 1) when the alarm fired.")
print("The first three rows use each motor's best-case threshold (zero false alarms);")
print("the last row is the fixed alarm level the pipeline really uses.\n")
print(pd.DataFrame(table).round(2).to_string(index=False))

# =====================================================================
# 2. Remaining useful life: four sensors combined vs vibration alone
# =====================================================================

runs = {}
for path in sorted(RUNS_DIR.glob("run_*.csv")):
    df = pd.read_csv(path)
    t = df["time_step"].to_numpy()
    index = health_index(df)
    t_fail = failure_time(df)
    runs[path.stem] = {
        "index": index,
        "vibration": df["vibration_mm_s"].rolling(WINDOW, min_periods=1).mean().to_numpy(),
        "t_det": detect_fault(index, t),
        "t_fail": t_fail,
        "d_at_fail": index[t_fail - 10: t_fail + 11].mean(),
    }


def rul_errors(signal, failure_level):
    """RUL error of the three estimators when extrapolating `signal` to its failure level."""
    parts = []
    for name, r in runs.items():
        est = rul_series(r[signal], r["t_det"] + MIN_AFTER_DETECTION, r["t_fail"], failure_level(name))
        est["true_rul"] = r["t_fail"] - est["time_step"]
        parts.append(est)
    est = pd.concat(parts)
    out = []
    for method in ["linear", "quadratic", "conservative"]:
        error = (est[method] - est["true_rul"]).dropna()
        near = est[est["true_rul"] <= 200]
        near_error = (near[method] - near["true_rul"]).dropna()
        out.append({"method": method, "MAE": error.abs().mean(),
                    "MAE_last_200": near_error.abs().mean(), "bias": error.mean()})
    return pd.DataFrame(out)


# Health index: failure level learned from the other motors. Vibration: the limit itself, known exactly.
from_others = lambda name: float(np.median([r["d_at_fail"] for k, r in runs.items() if k != name]))

health = rul_errors("index", from_others)
health.insert(0, "signal", "Health index (4 sensors)")
vibration = rul_errors("vibration", lambda name: VIB_LIMIT)
vibration.insert(0, "signal", "Vibration only")

rul_comparison = pd.concat([health, vibration])
rul_comparison.to_csv(RESULTS_DIR / "rul_signal_comparison.csv", index=False)

print("\n\nREMAINING USEFUL LIFE (errors in readings)")
print("Failure is defined by vibration, so the vibration-only baseline gets the exact failure")
print("level for free, while the health index must learn it from other motors.\n")
print(rul_comparison.round(1).to_string(index=False))