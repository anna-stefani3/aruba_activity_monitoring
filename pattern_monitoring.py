import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
from pprint import pprint
from generalised_monitoring import GeneralisedMonitoring

# Load Data
file_path = "[ARUBA]-activities_fixed_interval_data.csv"
df = pd.read_csv(file_path)
df["Time"] = pd.to_datetime(df["Time"])
df.set_index("Time", inplace=True)  # Set 'Time' as the index for easier resampling
df["Date"] = df.index.date  # Extract Date

# Step 1: Print summary of data after loading
print("Step 1: Data Loaded")
print(df.head())
input("Press Enter to continue...")


# Convert time to decimal representation
def time_to_decimal(t):
    return round(t.hour + t.minute / 60, 2)


# Identify night-time activities dynamically
def classify_night_activities(df, start_hour=18, end_hour=4):
    df["Hour"] = df.index.hour
    # Classify night activities based on time boundaries
    df["Is_Night"] = (df["Hour"] >= start_hour) | (df["Hour"] <= end_hour)
    return df


# Apply night-time activity classification
df = classify_night_activities(df)


# Define function to compute daily stats
def compute_daily_stats(x):
    activity_shifted = x["activity"].shift(1)

    # Count transitions from Sleeping to Bed_to_Toilet
    sleep_to_bed_to_toilet = ((x["activity"] == "Bed_to_Toilet") & (activity_shifted == "Sleeping")).sum()
    eating_count = ((x["activity"] == "Eating") & (activity_shifted != "Eating")).sum()
    cooking_count = ((x["activity"] == "Meal_Preparation") & (activity_shifted != "Meal_Preparation")).sum()

    # Determine the sleep start time using vectorized operations
    sleep_start_indices = (x["activity"] == "Sleeping") & ~activity_shifted.isin(["Sleeping", "Bed_to_Toilet"])
    sleep_start_time = None
    if sleep_start_indices.any():
        sleep_start_time = time_to_decimal(x.index[sleep_start_indices][0])  # Get first occurrence

    # Determine the wake-up time using vectorized operations
    wake_up_indices = ~x["activity"].isin(["Sleeping", "Bed_to_Toilet"]) & (activity_shifted == "Sleeping")
    wake_up_time = None
    if wake_up_indices.any():
        wake_up_time = time_to_decimal(x.index[wake_up_indices][0])  # Get first occurrence

    return pd.Series(
        {
            "sleep_duration": (x["activity"] == "Sleeping").sum() / 60,
            "sleep_disturbances": sleep_to_bed_to_toilet,
            "sleep_start_time": sleep_start_time,
            "wake_up_time": wake_up_time,
            "eating_count": eating_count,
            "cooking_count": cooking_count,
            "active_duration": (x["activity"].isin(["Meal_Preparation", "Wash_Dishes", "Housekeeping"])).sum() / 60,
        }
    )


# Step 2: Compute daily stats for all activities
df_daily_stats = df.groupby(df.index.date).apply(compute_daily_stats)

# Print summary of daily aggregated data
print("Step 2: Date-wise aggregated data computed:")
print(df_daily_stats.head())
input("Press Enter to continue...")

# Handle missing data by filling with zeros
df_daily_stats = df_daily_stats.fillna(0)


# Normalcy Test on Each Column
def perform_normalcy_test(data, column):
    stat, p = stats.shapiro(data.dropna())
    print(f"Normalcy test for {column}:")
    print(f"  Statistics = {stat}, p-value = {p}")
    if p > 0.05:
        print("  Data looks normal (fail to reject H0)\n\n")
    else:
        print("  Data does not look normal (reject H0)\n\n")


# Step 3: Perform normalcy test and show outputs for each column and combination
print("Step 3: Performing normalcy tests on each column and combination...")

for col in df_daily_stats.columns:
    perform_normalcy_test(df_daily_stats[col], col)

# Test combinations of columns
combo_cols = ["sleep_duration", "sleep_disturbances", "eating_count", "cooking_count"]
combo_data = df_daily_stats[combo_cols].dropna()
perform_normalcy_test(combo_data.sum(axis=1), "combined_columns_sum")

input("Press Enter to continue...")

# Split Data into 20%-80% ratio based on days
total_days = df_daily_stats.shape[0]
split_index = int(total_days * 0.2)

train_data = df_daily_stats.iloc[:split_index]
test_data = df_daily_stats.iloc[split_index:]

# Step 4: Data split completed
print(f"Step 4: Data split completed.")
print(f"\nTraining Data Summary:")
print(train_data.head())
print(f"\nTesting Data Summary:")
print(test_data.head())
input("Press Enter to continue...")

# Train GMM model on train data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(train_scaled)

# Step 5: GMM model training completed
print("Step 5: GMM model training completed.")
print("GMM Converged:", gmm.converged_)
print("Means:", gmm.means_)
print("Covariances:", gmm.covariances_)
input("Press Enter to continue...")


# Function to detect anomaly
def detect_anomaly(data_point, gmm_model, scaler, feature_names):
    data_point_df = pd.DataFrame([data_point], columns=feature_names)
    scaled_point = scaler.transform(data_point_df)
    score = gmm_model.score_samples(scaled_point)
    return score[0]


recent_scores = []
alert_triggered = False
window_size = 5
feature_names = train_data.columns


# Dynamically adjust anomaly threshold based on window statistics
def dynamic_threshold(recent_scores):
    mean_score = np.mean(recent_scores)
    std_dev = np.std(recent_scores)
    return mean_score - std_dev


print("\n\n")
generalised_monitoring = GeneralisedMonitoring()
# Step 6: Simulate real-time anomaly detection
for i in range(test_data.shape[0]):
    day_data = test_data.iloc[i]
    anomaly_score = detect_anomaly(day_data, gmm, scaler, feature_names)

    if len(recent_scores) >= window_size:
        recent_scores.pop(0)
    recent_scores.append(anomaly_score)

    if len(recent_scores) == window_size:
        threshold = dynamic_threshold(recent_scores)

        if anomaly_score < threshold:  # Use dynamic threshold here
            print(f"Day {str(i).rjust(3)} : {test_data.index[i]} - Abnormal")
            risks_analysis = generalised_monitoring.highlight_risks(day_data)
            scores = generalised_monitoring.get_scores(day_data)
            alerts = []
            for feature in scores.keys():
                if feature in risks_analysis.keys() and scores[feature] <= 0.5:
                    alerts.append(
                        f"----  {feature.ljust(20)} Value: {str(day_data[feature]).ljust(10)} Score : {str(int(scores[feature] * 10)).ljust(10)} {risks_analysis[feature]}"
                    )
            if len(alerts) == 0:
                print("----  [NORMAL] Generalised Scores are Normal")
            else:
                for alert in alerts:
                    print(alert)
            print("\n\n")
            alert_triggered = True

if not alert_triggered:
    print("No abnormal patterns detected over the simulated period.")
