import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

features = ["temperature_C", "vibration_mm_s", "current_A", "rpm"]
data = pd.read_csv("data/motor_degradation_data.csv")

# Rolling averages smooth the noise so the slow drift shows up
window = 50
model_features = []
for col in features:
    name = col + "_avg"
    data[name] = data[col].rolling(window, min_periods=1).mean()
    model_features.append(name)

# Hold out every 4th block of 100 readings as the test set
block = data["time_step"] // 100
is_test = block % 4 == 3
train = data[~is_test]
test = data[is_test]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(train[model_features], train["degradation_level"])

predictions = model.predict(test[model_features])

baseline = abs(test["degradation_level"] - train["degradation_level"].mean()).mean()
print(f"Baseline MAE (always predict the mean): {baseline:.3f}")
print(f"Model MAE: {mean_absolute_error(test['degradation_level'], predictions):.3f}")
print(f"Model R2: {r2_score(test['degradation_level'], predictions):.3f}")

# Save for the dashboard
results = test[["time_step", "degradation_level"]].copy()
results["predicted_level"] = predictions
results.to_csv("data/degradation_predictions.csv", index=False)
print("Saved to data/degradation_predictions.csv")
# Test on a completely separate run
run2 = pd.read_csv("data/motor_degradation_data_run2.csv")
for col in features:
    run2[col + "_avg"] = run2[col].rolling(window, min_periods=1).mean()

pred2 = model.predict(run2[model_features])
print(f"Run 2 MAE: {mean_absolute_error(run2['degradation_level'], pred2):.3f}")
print(f"Run 2 R2: {r2_score(run2['degradation_level'], pred2):.3f}")