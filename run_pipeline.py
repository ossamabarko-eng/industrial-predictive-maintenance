import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import RUNS_DIR, RESULTS_DIR, FIGURES_DIR, MIN_AFTER_DETECTION, STEP
from src.simulate import simulate_all
from src.health import health_index, detect_fault, failure_time, health_score, health_status
from src.rul import rul_series

METHODS = ["linear", "quadratic", "conservative"]

# 1. Data: simulate 12 motors if none exist yet
if not any(RUNS_DIR.glob("run_*.csv")):
    print("No runs found, simulating 12 motors...")
    simulate_all()

# 2. Health index, detection time and failure time for every run
runs = {}
for path in sorted(RUNS_DIR.glob("run_*.csv")):
    df = pd.read_csv(path)
    t = df["time_step"].to_numpy()
    idx = health_index(df)
    t_fail = failure_time(df)
    runs[path.stem] = {
        "df": df,
        "index": idx,
        "t_det": detect_fault(idx, t),
        "t_fail": t_fail,
        "d_at_fail": idx[t_fail - 10: t_fail + 11].mean(),
    }

# 3. Leave-one-run-out: each run's failure level comes from the other runs only
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

estimates, summary = [], []
for name, r in runs.items():
    d_fail = float(np.median([v["d_at_fail"] for k, v in runs.items() if k != name]))
    score = health_score(r["index"], d_fail)

    est = rul_series(r["index"], r["t_det"] + MIN_AFTER_DETECTION, r["t_fail"], d_fail)
    est["true_rul"] = r["t_fail"] - est["time_step"]
    est["run"] = name
    estimates.append(est)

    # One result file per motor, ready for the dashboard
    out = pd.DataFrame({
        "time_step": r["df"]["time_step"],
        "health_index": r["index"],
        "health_score": score,
        "health_status": health_status(score),
        "degradation_level": r["df"]["degradation_level"],
    })
    out = out.merge(est[["time_step"] + METHODS], on="time_step", how="left")
    out[METHODS] = out[METHODS].ffill(limit=STEP - 1)   # latest estimate stays valid until the next one
    out["true_rul"] = (r["t_fail"] - out["time_step"]).clip(lower=0)
    out.to_csv(RESULTS_DIR / f"{name}.csv", index=False)

    summary.append({
        "run": name,
        "detected": r["t_det"],
        "failure": r["t_fail"],
        "warning_lead_time": r["t_fail"] - r["t_det"],
        "failure_level": round(d_fail, 2),
    })

summary = pd.DataFrame(summary)
summary.to_csv(RESULTS_DIR / "run_summary.csv", index=False)
print(summary.to_string(index=False))

# 4. RUL accuracy
long = pd.concat(estimates).melt(
    id_vars=["run", "time_step", "true_rul"],
    value_vars=METHODS,
    var_name="method",
    value_name="pred_rul",
)
long["error"] = long["pred_rul"] - long["true_rul"]

rows = []
for method in METHODS:
    m = long[long["method"] == method]
    valid = m.dropna(subset=["pred_rul"])
    near = valid[valid["true_rul"] <= 200]
    rows.append({
        "method": method,
        "MAE": valid["error"].abs().mean(),
        "MAE_last_200": near["error"].abs().mean(),
        "bias": valid["error"].mean(),
        "too_late": (valid["error"] > 0).mean(),
        "no_prediction": m["pred_rul"].isna().mean(),
    })
metrics = pd.DataFrame(rows)
metrics.to_csv(RESULTS_DIR / "rul_metrics.csv", index=False)

print("\nRUL error in readings (positive = predicted failure later than reality)")
print(metrics.round({"MAE": 1, "MAE_last_200": 1, "bias": 1, "too_late": 2, "no_prediction": 2}).to_string(index=False))

# 5. Plot predicted vs true RUL
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
for ax, method in zip(axes, METHODS):
    m = long[long["method"] == method]
    for name, g in m.groupby("run"):
        ax.plot(g["true_rul"], g["pred_rul"], linewidth=0.8, alpha=0.7)
    lim = m["true_rul"].max()
    ax.plot([0, lim], [0, lim], "k--", label="Perfect prediction")
    ax.set_title(method.capitalize())
    ax.set_xlabel("True remaining useful life (readings)")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("Predicted RUL (readings)")
axes[0].set_xlim(lim, 0)
axes[0].set_ylim(-20, 600)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rul_prediction.png", dpi=150)

print(f"\nSaved results to {RESULTS_DIR}/ and figures/rul_prediction.png")