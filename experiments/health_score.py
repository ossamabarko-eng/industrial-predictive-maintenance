import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

features = ["temperature_C", "vibration_mm_s", "current_A", "rpm"]
window = 50
VIB_LIMIT = 3.5   # placeholder failure rule: replace with an engineering limit

data = pd.read_csv("data/motor_degradation_data.csv")
Path("figures").mkdir(exist_ok=True)

X = pd.DataFrame(index=data.index)
for col in features:
    X[col + "_avg"] = data[col].rolling(window, min_periods=1).mean()

t = data["time_step"]
start = t[data["status"] == "Degrading"].min()
train_mask = (t >= window) & (t < 0.7 * start)
calib_mask = (t >= 0.7 * start) & (t < start)

# Mahalanobis distance from the healthy baseline
scaler = StandardScaler().fit(X[train_mask])
Z = scaler.transform(X)
mu = Z[train_mask].mean(axis=0)
inv_cov = np.linalg.pinv(np.cov(Z[train_mask], rowvar=False))
diff = Z - mu
data["health_index"] = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))

# Healthy level: 99th percentile on unseen healthy readings
d_ok = np.percentile(data.loc[calib_mask, "health_index"], 99)

# Failure level: index value when smoothed vibration first reaches the limit
crossed = t[(X["vibration_mm_s_avg"] >= VIB_LIMIT) & (t >= start)]
if len(crossed) > 0:
    d_fail = data.loc[t == crossed.min(), "health_index"].iloc[0]
else:
    d_fail = data["health_index"].max()
    print("Vibration limit never reached; using the maximum index value.")

# Health score: 100 at d_ok, 0 at d_fail
frac = ((data["health_index"] - d_ok) / (d_fail - d_ok)).clip(0, 1)
data["health_score"] = 100 * (1 - frac)
data["health_status"] = pd.cut(
    data["health_score"],
    bins=[-1, 10, 40, 70, 100.1],
    labels=["Critical", "Degraded", "Warning", "Healthy"],
)

data = data[t >= window]
t = data["time_step"]
deg = t >= start

print(f"Degradation starts at reading {start}")
print(f"d_ok = {d_ok:.2f}, d_fail = {d_fail:.2f}")
print(f"Spearman correlation of the index with time during degradation: "
      f"{data.loc[deg, ['health_index', 'time_step']].corr(method='spearman').iloc[0, 1]:.3f}")
print("\nFirst reading in each status:")
print(data.groupby("health_status", observed=True)["time_step"].min().sort_values())
print("\nHealthy-period readings below 70:",
      int(((t < start) & (data["health_score"] <= 70)).sum()))

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

axes[0].plot(t, data["health_index"], linewidth=0.8)
axes[0].axhline(d_ok, color="green", linestyle="--", label="Healthy level")
axes[0].axhline(d_fail, color="red", linestyle="--", label="Failure level")
axes[0].set_ylabel("Health index (Mahalanobis)")
axes[0].legend()

axes[1].plot(t, data["health_score"], linewidth=1.2, color="black")
axes[1].axhspan(70, 100, color="green", alpha=0.12, label="Healthy")
axes[1].axhspan(40, 70, color="gold", alpha=0.18, label="Warning")
axes[1].axhspan(10, 40, color="orange", alpha=0.18, label="Degraded")
axes[1].axhspan(0, 10, color="red", alpha=0.15, label="Critical")
axes[1].set_ylabel("Health score")
axes[1].set_xlabel("Time step")
axes[1].set_ylim(0, 100)
axes[1].legend(loc="lower left")

for ax in axes:
    ax.axvline(start, color="gray", linestyle="--")
    ax.grid(alpha=0.3)

axes[0].set_title("Machine health index and health score")
plt.tight_layout()
plt.savefig("figures/health_score.png", dpi=150)
plt.show()

data[["time_step", "health_index", "health_score", "health_status",
      "degradation_level"]].to_csv("data/health_score.csv", index=False)