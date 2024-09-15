import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np

# Load Data
file_path = "[ARUBA]-activities_fixed_interval_data.csv"
df = pd.read_csv(file_path)
df["Time"] = pd.to_datetime(df["Time"])
df.set_index("Time", inplace=True)  # Set 'Time' as the index for easier resampling
df["Date"] = df.index.date  # Extract Date

# 1. Print summary of data after loading
print("Step 1: Data Loaded")
print(df.head())
input("Press Enter to continue...")

# Filter relevant activities only
relevant_activities = ["Sleeping", "Bed_to_Toilet", "Eating", "Meal_Preparation"]
df = df[df["activity"].isin(relevant_activities)]


# Convert time to decimal representation
def time_to_decimal(t):
    return t.hour + t.minute / 60


# Define function to compute daily stats
def compute_daily_stats(x):
    activity_shifted = x["activity"].shift(1)

    # Count transitions from Sleeping to Bed_to_Toilet
    sleep_to_bed_to_toilet = ((x["activity"] == "Bed_to_Toilet") & (activity_shifted == "Sleeping")).sum()

    # Time when the person went to sleep
    sleep_start_times = x[(x["activity"] == "Sleeping") & (~activity_shifted.isin(["Sleeping", "Bed_to_Toilet"]))].index
    if not sleep_start_times.empty:
        sleep_start = time_to_decimal(sleep_start_times[0])
    else:
        sleep_start = None

    # Time when the person woke up
    wake_up_times = x[~x["activity"].isin(["Sleeping", "Bed_to_Toilet"])].index
    if not wake_up_times.empty:
        wake_up = time_to_decimal(wake_up_times[0])
    else:
        wake_up = None

    return pd.Series(
        {
            "sleep_count": (x["activity"] == "Sleeping").sum(),
            "sleep_disturbances": sleep_to_bed_to_toilet,
            "sleep_start_time": sleep_start,
            "wake_up_time": wake_up,
            "eating_count": (x["activity"] == "Eating").sum(),
            "meal_preparation_count": (x["activity"] == "Meal_Preparation").sum(),
        }
    )


# Compute daily stats for all activities
df_daily_stats = df.groupby(df.index.date).apply(compute_daily_stats)

# 2. Print summary of daily aggregated data
print("Step 2: Date-wise aggregated data computed:")
print(df_daily_stats.head())
input("Press Enter to continue...")

# Handle missing data by filling with zeros
df_daily_stats = df_daily_stats.fillna(0)


# Normalcy Test on Each Column
def perform_normalcy_test(data, column):
    # Shapiro-Wilk Test for normality
    stat, p = stats.shapiro(data.dropna())
    print(f"Normalcy test for {column}:")
    print(f"  Statistics = {stat}, p-value = {p}")
    if p > 0.05:
        print("  Data looks normal (fail to reject H0)")
    else:
        print("  Data does not look normal (reject H0)")


# 3. Perform normalcy test and show outputs for each column and combination
print("Step 3: Performing normalcy tests on each column and combination...")

# Test each column individually
for col in df_daily_stats.columns:
    perform_normalcy_test(df_daily_stats[col], col)

# Test combinations of columns
combo_cols = ["sleep_count", "sleep_disturbances", "eating_count", "meal_preparation_count"]
combo_data = df_daily_stats[combo_cols].dropna()

# Test for combination
print("Normalcy test for combined columns (sum):")
perform_normalcy_test(combo_data.sum(axis=1), "combined_columns_sum")

input("Press Enter to continue...")

# Split Data into 20%-80% ratio based on days
total_days = df_daily_stats.shape[0]
split_index = int(total_days * 0.2)

train_data = df_daily_stats.iloc[:split_index]
test_data = df_daily_stats.iloc[split_index:]

# 4. Show summary of train and test data
print(f"Step 4: Data split completed.")
print(f"\nTraining Data Summary:")
print(train_data.describe())
print(f"\nTesting Data Summary:")
print(test_data.describe())
input("Press Enter to continue...")

# Train GMM model on train data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)  # Use all columns for fitting

gmm = GaussianMixture(n_components=2, random_state=42)  # Adjust components if necessary
gmm.fit(train_scaled)

# 5. Print relevant GMM details
print("Step 5: GMM model training completed.")
print("GMM Converged:", gmm.converged_)
print("Means:", gmm.means_)
print("Covariances:", gmm.covariances_)
input("Press Enter to continue...")


def detect_anomaly(data_point, gmm_model, scaler, feature_names):
    # Convert the data_point into a DataFrame with appropriate column names
    data_point_df = pd.DataFrame([data_point], columns=feature_names)
    scaled_point = scaler.transform(data_point_df)
    score = gmm_model.score_samples(scaled_point)
    return score[0]


recent_scores = []

alert_triggered = False
window_size = 5

# Get feature names from training data
feature_names = train_data.columns

# Simulate real-time arrival of each day's data
for i in range(test_data.shape[0]):
    day_data = test_data.iloc[i]  # Simulate getting one day's data
    anomaly_score = detect_anomaly(day_data, gmm, scaler, feature_names)

    # Maintain the sliding window of the last 14 days' scores
    if len(recent_scores) >= window_size:
        recent_scores.pop(0)  # Remove the oldest score to keep only the last 14 days
    recent_scores.append(anomaly_score)

    # Only start anomaly detection after we have 14 days of data
    if len(recent_scores) == window_size:
        avg_recent_score = np.mean(recent_scores)

        # Check if today's anomaly score is significantly lower (indicating an anomaly)
        if anomaly_score < (avg_recent_score - 3):  # Adjust threshold as necessary
            print(f"Day {str(i).rjust(3)} : {test_data.index[i]} - Abnormal")
            alert_triggered = True

if not alert_triggered:
    print("No abnormal patterns detected over the simulated period.")
