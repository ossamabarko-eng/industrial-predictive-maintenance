import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

features = ["temperature_C", "vibration_mm_s", "current_A", "rpm"]
window = 50      # smoothing window
persist = 20     # consecutive flagged readings needed for an alarm

data = pd.read_csv("data/motor_degradation_data.csv")
Path("figures").mkdir(exist_ok=True)

# Smoothed features
X = pd.DataFrame(index=data.index)
for col in features:
    X[col + "_avg"] = data[col].rolling(window, min_periods=1).mean()

t = data["time_step"]
start = t[data["status"] == "Degrading"].min()

# Healthy period split: train on 70%, set the threshold on the rest
train_mask = (t >= window) & (t < 0.7 * start)
calib_mask = (t >= 0.7 * start) & (t < start)

scaler = StandardScaler().fit(X[train_mask])
model = IsolationForest(n_estimators=200, random_state=42)
model.fit(scaler.transform(X[train_mask]))

# Higher score = more abnormal
data["anomaly_score"] = -model.score_samples(scaler.transform(X))

threshold = np.percentile(data.loc[calib_mask, "anomaly_score"], 99)

flag = (data["anomaly_score"] > threshold).astype(int)
data["alarm"] = flag.rolling(persist).min() == 1

# Evaluation
healthy_eval = (t >= window) & (t < start)
false_alarm_rate = data.loc[healthy_eval, "alarm"].mean()
print(f"Degradation starts at reading {start}")
print(f"Threshold (99th percentile of unseen healthy data): {threshold:.3f}")
print(f"False alarm rate during healthy period: {false_alarm_rate:.1%}")

alarms = data.loc[(t >= start) & data["alarm"]]
if len(alarms) > 0:
    first = alarms.iloc[0]
    print(f"First alarm at reading {int(first['time_step'])} "
          f"({int(first['time_step'] - start)} readings after degradation began)")
    print(f"Degradation level at first alarm: {first['degradation_level']:.2f}")
    print(f"Share of degrading period flagged: {alarms.shape[0] / (t >= start).sum():.1%}")
else:
    print("No alarm raised during degradation")

# Plot
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t, data["anomaly_score"], linewidth=0.8, label="Anomaly score")
ax.axhline(threshold, color="red", linestyle="--", label="Alarm threshold")
ax.axvline(start, color="gray", linestyle="--", label="Degradation begins")
if len(alarms) > 0:
    ax.axvline(first["time_step"], color="orange", linestyle="-", label="First alarm")
ax.set_xlabel("Time step")
ax.set_ylabel("Anomaly score")
ax.set_title("Anomaly score trained on healthy operation")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/anomaly_score.png", dpi=150)
plt.show()

# Saved for the health score step
data[["time_step", "anomaly_score", "alarm", "degradation_level"]].to_csv(
    "data/anomaly_results_degradation.csv", index=False
)