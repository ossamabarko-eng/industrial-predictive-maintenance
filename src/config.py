from pathlib import Path

FEATURES = ["temperature_C", "vibration_mm_s", "current_A", "rpm"]

# Signal processing
WINDOW = 50                  # smoothing window (readings)
BASELINE = (50, 450)         # each motor's own early readings = its healthy baseline

# Fault detection
ALARM_FROM = 450             # start looking for faults after the baseline period
ALARM_LEVEL = 5.5            # health index above this = abnormal (same for every motor)
PERSIST = 20                 # consecutive readings above ALARM_LEVEL = fault detected

# Failure definition (placeholder: replace with a real limit, e.g. from ISO 20816)
VIB_LIMIT = 3.5              # smoothed vibration (mm/s) at which the motor counts as failed

# Remaining useful life
HISTORY = 150                # readings of recent trend used for each prediction
STEP = 5                     # make a prediction every 5 readings
MIN_AFTER_DETECTION = 100    # wait this long after detection before predicting
MAX_RUL = 1500               # cap on predicted RUL (readings)

# Health score status bands: above 70 Healthy, 40-70 Warning, 10-40 Degraded, below 10 Critical
BAND_EDGES = (70, 40, 10)

RUNS_DIR = Path("data/runs")
RESULTS_DIR = Path("data/results")
FIGURES_DIR = Path("figures")