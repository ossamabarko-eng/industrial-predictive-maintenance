import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config import (
    RUNS_DIR, RESULTS_DIR, FIGURES_DIR, FEATURES, WINDOW, ALARM_LEVEL,
    MIN_AFTER_DETECTION, VIB_LIMIT, BASELINE,
)

SAFETY_MARGIN = 50   # readings subtracted from the predicted RUL when planning maintenance
LABELS = {
    "temperature_C": "Temperature (°C)",
    "vibration_mm_s": "Vibration (mm/s)",
    "current_A": "Current (A)",
    "rpm": "Speed (RPM)",
}

st.set_page_config(page_title="Industrial Predictive Maintenance", layout="wide")
st.title("Industrial Predictive Maintenance Dashboard")
st.caption("Simulated condition monitoring of industrial motors. One reading = one time step.")

# ---------- Load the results written by run_pipeline.py ----------
summary_path = RESULTS_DIR / "run_summary.csv"
if not summary_path.exists():
    st.error("No results found. Run `python run_pipeline.py` first.")
    st.stop()

summary = pd.read_csv(summary_path)


@st.cache_data
def load_run(name):
    sensors = pd.read_csv(RUNS_DIR / f"{name}.csv")
    results = pd.read_csv(RESULTS_DIR / f"{name}.csv")
    return sensors, results


def recommendation(status, rul):
    """Turn the health status and predicted remaining life into a maintenance action."""
    known = not np.isnan(rul)
    if status == "Critical" or (known and rul <= SAFETY_MARGIN):
        return "error", "Immediate maintenance required. Stop the motor or repair it now."
    if status == "Degraded":
        if known:
            window = max(rul - SAFETY_MARGIN, 0)
            return "error", f"Plan maintenance now and complete it within about {window:.0f} readings."
        return "error", "Plan maintenance now."
    if status == "Warning":
        return "warning", "Schedule an inspection for mechanical wear (bearings, alignment, lubrication)."
    return "success", "No action needed. Continue routine monitoring."


# ---------- Sidebar ----------
run = st.sidebar.selectbox("Motor", summary["run"])
info = summary[summary["run"] == run].iloc[0]
sensors, res = load_run(run)

detected = int(info["detected"])
failure = int(info["failure"])
now = st.sidebar.slider("Current reading", BASELINE[1], len(res) - 1, detected + 150)
st.sidebar.caption(
    "Move the slider to see what the system would show at that moment. "
    "Only data up to the selected reading is used."
)

# ---------- Current state ----------
cur = res.iloc[now]
score, status, index = cur["health_score"], cur["health_status"], cur["health_index"]
rul = cur["conservative"]

if now >= failure:
    rul, rul_text, rul_note = 0.0, "0", "Failure limit reached"
elif now < detected:
    rul_text, rul_note = "—", "No fault detected yet"
elif np.isnan(rul):
    rul_text, rul_note = "—", "Collecting trend"
else:
    rul_text, rul_note = f"{rul:.0f}", "Conservative estimate"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Health score", f"{score:.0f} / 100")
c2.metric("Status", status)
c3.metric("Health index", f"{index:.1f}", delta=f"alarm level {ALARM_LEVEL}", delta_color="off", delta_arrow="off")
c4.metric("Remaining life (readings)", rul_text, delta=rul_note, delta_color="off", delta_arrow="off")
level, message = recommendation(status, rul)
getattr(st, level)(f"**Maintenance recommendation:** {message}")

st.divider()

# ---------- Health score, health index and RUL ----------
hist = res.iloc[: now + 1]
left, middle, right = st.columns(3)

with left:
    st.subheader("Health score")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.axhspan(70, 100, color="green", alpha=0.12)
    ax.axhspan(40, 70, color="gold", alpha=0.18)
    ax.axhspan(10, 40, color="orange", alpha=0.18)
    ax.axhspan(0, 10, color="red", alpha=0.15)
    ax.plot(hist["time_step"], hist["health_score"], color="black", linewidth=1.2)
    ax.set_xlim(detected + MIN_AFTER_DETECTION, len(res))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Reading")
    ax.set_ylabel("Health score")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with middle:
    st.subheader("Anomaly (health index)")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(hist["time_step"], hist["health_index"], linewidth=1.2)
    ax.axhline(ALARM_LEVEL, color="red", linestyle="--", label="Alarm level")
    ax.set_xlim(0, len(res))
    ax.set_ylim(0, res["health_index"].max() * 1.05)
    ax.set_xlabel("Reading")
    ax.set_ylabel("Distance from healthy baseline")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Remaining useful life")
    est = hist.dropna(subset=["conservative"])
    if est.empty:
        st.info(f"Estimates start {MIN_AFTER_DETECTION} readings after a fault is detected.")
    else:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot(est["time_step"], est["conservative"], label="Predicted (conservative)", linewidth=1.5)
        ax.plot(est["time_step"], est["true_rul"], "k--", label="True (known only in simulation)", linewidth=1)
        ax.set_xlim(0, len(res))
        ax.set_xlabel("Reading")
        ax.set_ylabel("Readings left")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

# ---------- Sensor trends ----------
st.subheader("Sensor trends")
sens = sensors.iloc[: now + 1]
fig, axes = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
for ax, col in zip(axes.flat, FEATURES):
    ax.plot(sens["time_step"], sens[col], linewidth=0.5, alpha=0.5)
    ax.plot(sens["time_step"], sens[col].rolling(WINDOW, min_periods=1).mean(),
            linewidth=1.8, label=f"{WINDOW}-reading average")
    if now >= detected:
        ax.axvline(detected, color="red", linestyle="--", linewidth=1, label="Fault detected")
    ax.set_title(LABELS[col], fontsize=10)
    ax.set_xlim(0, len(res))
    ax.grid(alpha=0.3)
axes[0, 0].legend(loc="upper left", fontsize=8)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------- Alert history ----------
st.subheader("Alert history")
events = []
if now >= detected:
    events.append({"reading": detected, "event": "Fault detected",
                   "health score": round(float(res.loc[detected, "health_score"]))})
for label in ["Warning", "Degraded", "Critical"]:
    reached = hist.loc[hist["health_status"] == label, "time_step"]
    if len(reached):
        first = int(reached.min())
        events.append({"reading": first, "event": f"Status first reached {label}",
                       "health score": round(float(res.loc[first, "health_score"]))})
if now >= failure:
    events.append({"reading": failure, "event": f"Failure limit reached (vibration {VIB_LIMIT} mm/s)",
                   "health score": round(float(res.loc[failure, "health_score"]))})

if events:
    st.dataframe(pd.DataFrame(events).sort_values("reading"), hide_index=True)
else:
    st.info("No alerts so far. The motor is operating normally.")

# ---------- Model information ----------
with st.expander("How accurate are the RUL predictions?"):
    metrics = pd.read_csv(RESULTS_DIR / "rul_metrics.csv")
    st.dataframe(metrics.round(2), hide_index=True)
    st.caption(
        "Leave-one-run-out test on simulated motors. Errors are in readings; "
        "a positive bias means failure was predicted later than it happened."
    )
    image = FIGURES_DIR / "rul_prediction.png"
    if image.exists():
        st.image(str(image))

with st.expander("Simulation ground truth (not available on a real machine)"):
    st.write(f"True degradation level: {cur['degradation_level']:.2f}")
    st.write(f"True remaining useful life: {cur['true_rul']:.0f} readings")

st.caption(
    f"Simulated data. Failure is defined as smoothed vibration reaching {VIB_LIMIT} mm/s, "
    "a placeholder for a real engineering limit."
)