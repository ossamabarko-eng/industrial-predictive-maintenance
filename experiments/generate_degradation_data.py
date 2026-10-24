import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible results

np.random.seed(42)

# Number of simulated time steps

n = 2000

time = np.arange(n)

# Simulated gradual degradation begins at time step 1200

degradation_start = 1200

degradation = np.zeros(n)

degradation[degradation_start:] = np.linspace(
0,
1,
n - degradation_start
)

# Simulate sensor readings

temperature = (
65
+ 3 * degradation
+ np.random.normal(0, 1.5, n)
)

vibration = (
2.0
+ 1.8 * degradation
+ np.random.normal(0, 0.2, n)
)

current = (
4.5
+ 0.7 * degradation
+ np.random.normal(0, 0.25, n)
)

rpm = (
1500
- 100 * degradation
+ np.random.normal(0, 15, n)
)

# Create a dataset

data = pd.DataFrame({
"time_step": time,
"temperature_C": temperature,
"vibration_mm_s": vibration,
"current_A": current,
"rpm": rpm,
"degradation_level": degradation
})

# Label the simulated degradation period

data["status"] = "Normal"

data.loc[
data["time_step"] >= degradation_start,
"status"
] = "Degrading"

# Save the dataset

output_folder = Path("data")
output_folder.mkdir(exist_ok=True)

output_path = output_folder / "motor_degradation_data.csv"

data.to_csv(output_path, index=False)

print("Degradation dataset generated successfully.")
print("Total readings:", len(data))
print("\nStatus distribution:")
print(data["status"].value_counts())

print("\nFirst five readings:")
print(data.head())

print("\nLast five readings:")
print(data.tail())

print("\nSaved to:", output_path)
