import numpy as np
import pandas as pd
import pytest

from src.config import ALARM_LEVEL, VIB_LIMIT, WINDOW
from src.simulate import simulate_run
from src.health import health_index, detect_fault, failure_time, health_score, health_status
from src.rul import predict_rul, rul_series


@pytest.fixture(scope="module")
def motor():
    return simulate_run(run_id=1)


def degradation_start(df):
    return int(df.loc[df["status"] == "Degrading", "time_step"].min())


def test_simulation_is_reproducible():
    pd.testing.assert_frame_equal(simulate_run(run_id=3), simulate_run(run_id=3))


def test_health_index_is_low_when_healthy_and_high_when_degraded(motor):
    index = health_index(motor)
    t = motor["time_step"].to_numpy()
    healthy = index[(t >= 450) & (t < degradation_start(motor))]
    assert healthy.max() < ALARM_LEVEL
    assert index[-1] > 10 * ALARM_LEVEL


def test_no_fault_is_detected_on_healthy_data(motor):
    healthy = motor[motor["status"] == "Normal"].reset_index(drop=True)
    assert detect_fault(health_index(healthy), healthy["time_step"].to_numpy()) is None


def test_fault_is_detected_soon_after_degradation_starts(motor):
    detected = detect_fault(health_index(motor), motor["time_step"].to_numpy())
    start = degradation_start(motor)
    assert detected is not None
    assert start <= detected < start + 200


def test_failure_time_is_when_smoothed_vibration_reaches_the_limit(motor):
    t_fail = failure_time(motor)
    vibration = motor["vibration_mm_s"].rolling(WINDOW, min_periods=1).mean()
    assert vibration.iloc[t_fail] >= VIB_LIMIT
    assert vibration.iloc[t_fail - 1] < VIB_LIMIT


def test_health_score_and_status_bands():
    score = health_score(np.array([0.0, ALARM_LEVEL, 40.0, 100.0]), d_fail=40.0)
    assert list(score) == [100, 100, 0, 0]
    status = health_status(np.array([100, 71, 55, 20, 5]))
    assert list(status) == ["Healthy", "Healthy", "Warning", "Degraded", "Critical"]


def test_rul_recovers_a_linear_trend():
    index = 0.5 * np.arange(1000)   # at reading 300 the index is 150 and reaches 300 at reading 600
    assert predict_rul(index, now=300, d_fail=300.0, degree=1) == pytest.approx(300, abs=1)


def test_rul_is_nan_when_the_index_is_falling():
    index = 100 - 0.1 * np.arange(1000)
    assert np.isnan(predict_rul(index, now=500, d_fail=80.0, degree=1))


def test_rul_series_counts_down_on_a_clean_ramp():
    index = 0.05 * np.arange(2000)   # reaches the failure level 80 at reading 1600
    est = rul_series(index, first=300, last=1000, d_fail=80.0)
    assert est["linear"].iloc[0] == pytest.approx(1300, abs=1)
    assert est["linear"].is_monotonic_decreasing