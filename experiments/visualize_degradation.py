import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data = pd.read_csv("data/motor_degradation_data.csv")
Path("figures").mkdir(exist_ok=True)

signals = [
    ("temperature_C", "Temperature (°C)"),
    ("vibration_mm_s", "Vibration (mm/s)"),
    ("current_A", "Current (A)"),
    ("rpm", "Speed (RPM)"),
    ("degradation_level", "Degradation level"),
]

fig, axes = plt.subplots(len(signals), 1, figsize=(11, 12), sharex=True)

for ax, (col, label) in zip(axes, signals):
    ax.plot(data["time_step"], data[col], linewidth=0.8)
    if col != "degradation_level":
        ax.plot(
            data["time_step"],
            data[col].rolling(50, min_periods=1).mean(),
            linewidth=1.8,
            label="50-reading average",
        )
        ax.legend(loc="upper left")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)

# Mark where the degradation period begins
start = data.loc[data["status"] == "Degrading", "time_step"].min()
for ax in axes:
    ax.axvline(start, linestyle="--", color="gray")

axes[0].set_title("Motor sensor signals and degradation over time")
axes[-1].set_xlabel("Time step")

plt.tight_layout()
plt.savefig("figures/degradation_overview.png", dpi=150)
plt.show()