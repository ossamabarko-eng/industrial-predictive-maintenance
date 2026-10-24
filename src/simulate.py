import numpy as np
import pandas as pd

from .config import RUNS_DIR


def simulate_run(run_id, n_readings=2000):
    """Simulate one motor: a healthy period followed by gradual degradation."""
    rng = np.random.default_rng(1000 + run_id)
    t = np.arange(n_readings)

    # Each motor degrades differently
    start = int(rng.integers(750, 1201))     # reading where degradation begins
    duration = int(rng.integers(500, 701))   # readings until degradation_level reaches 1
    shape = rng.uniform(1.0, 1.8)            # 1 = linear wear, above 1 = accelerating wear

    level = np.clip((t - start) / duration, 0, 1) ** shape

    # Small motor-to-motor differences in the healthy operating point
    temp0 = 65 + rng.normal(0, 1.0)
    vib0 = 2.0 + rng.normal(0, 0.08)
    cur0 = 4.5 + rng.normal(0, 0.1)
    rpm0 = 1500 + rng.normal(0, 5)

    noise = 1 + 0.5 * level  # sensors get noisier as the motor wears

    data = pd.DataFrame({
        "run_id": run_id,
        "time_step": t,
        "temperature_C": temp0 + 2.5 * level + rng.normal(0, 2.0, n_readings) * noise,
        "vibration_mm_s": vib0 + 1.8 * level + rng.normal(0, 0.25, n_readings) * noise,
        "current_A": cur0 + 0.7 * level + rng.normal(0, 0.25, n_readings) * noise,
        "rpm": rpm0 - 100 * level + rng.normal(0, 10, n_readings) * noise,
        "degradation_level": level,
    })
    data["status"] = np.where(t >= start, "Degrading", "Normal")
    return data


def simulate_all(n_runs=12, n_readings=2000):
    """Simulate several motors and save each one to data/runs/."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for run_id in range(1, n_runs + 1):
        simulate_run(run_id, n_readings).to_csv(RUNS_DIR / f"run_{run_id:02d}.csv", index=False)