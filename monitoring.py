import pandas as pd
import numpy as np
from scipy.stats import chi2

# Load the dataset
df = pd.read_csv("[ARUBA]-activities_fixed_interval_data.csv")

print("Data loaded successfully:")
print(df.head())
input("\nPress Enter to continue...")


print("\nColumn names after checking:")
print(df.columns)
input("\nPress Enter to continue...")

# Convert Time column to datetime format
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

# Drop any rows where the 'Time' or 'activity' is missing
df.dropna(subset=["Time", "activity"], inplace=True)

print("\nData after dropping missing values:")
print(df.head())
input("\nPress Enter to continue...")

# Set 'Time' as the index for easier resampling and time-based operations
df.set_index("Time", inplace=True)


# Define function to compute daily stats
def compute_daily_stats(x):
    return pd.Series(
        {
            "sleep_count": (x["activity"] == "Sleeping").sum(),
            "sleep_disturbances": (x["activity"] == "Bed_to_Toilet").sum(),
        }
    )


# Compute daily stats
df_daily_stats = df.groupby(df.index.date).apply(compute_daily_stats)

print("\nDaily stats calculated:")
print(df_daily_stats.head())
input("\nPress Enter to continue...")

# Determine the split date based on 80% of the total dates
unique_dates = df_daily_stats.index
total_dates = len(unique_dates)
split_point_index = int(total_dates * 0.1)
split_date = unique_dates[split_point_index]

print(f"\nDetermined split date: {split_date}")
input("\nPress Enter to continue...")

# Split data into training and monitoring
df_train = df_daily_stats[df_daily_stats.index < split_date]
df_monitor = df_daily_stats[df_daily_stats.index >= split_date]

print("\nTraining data sample:")
print(df_train.head())
print("\nMonitoring data sample:")
print(df_monitor.head())
input("\nPress Enter to continue...")

# Define importance weights for anomaly detection
importance_weights = {
    "sleep_count": 1,
    "sleep_disturbances": 0.8,
}


# Function to compute personalized profile
def compute_personal_profile(df):
    activity_matrix = df.to_numpy()
    mean_vector = activity_matrix.mean(axis=0)
    cov_matrix = np.cov(activity_matrix, rowvar=False)
    return mean_vector, cov_matrix


# Compute mean and covariance from training data
mean_vector_train, cov_matrix_train = compute_personal_profile(df_train)

print("\nPersonalized profile (mean and covariance) from training data:")
print(f"Mean vector: {mean_vector_train}")
print(f"Covariance matrix: {cov_matrix_train}")
input("\nPress Enter to continue...")


# Function to detect anomalies using weighted Mahalanobis distance
def detect_anomaly(activity_vector, mean_vector, cov_matrix, threshold=0.01):
    weights = np.array(list(importance_weights.values()))

    diff = activity_vector - mean_vector
    weighted_diff = diff * weights
    inv_cov = np.linalg.inv(cov_matrix)
    mahalanobis_dist = np.sqrt(weighted_diff.T @ inv_cov @ weighted_diff)

    dof = len(activity_vector)
    p_value = 1 - chi2.cdf(mahalanobis_dist**2, dof)

    if p_value < threshold:
        return "Anomaly Detected", p_value
    else:
        return "Normal", p_value


# Function to update Grist score
def update_grist_score(personal_status):
    if personal_status == "Anomaly Detected":
        return "Alert"
    else:
        return "Normal"


# Monitoring function that computes daily Grist Score
def monitor_activities_daily(df_monitor, mean_vector_train, cov_matrix_train, df_population):
    scores = []
    for day, stats in df_monitor.iterrows():
        activity_vector_monitor = stats.to_numpy()

        personal_status, _ = detect_anomaly(activity_vector_monitor, mean_vector_train, cov_matrix_train)

        grist_score = update_grist_score(personal_status)
        scores.append((day, grist_score, stats))
    return scores


# Run the monitoring
daily_grist_scores = monitor_activities_daily(df_monitor, mean_vector_train, cov_matrix_train, df_train)

print("\nDaily Grist Scores with p-values:")
for day, score, stats in daily_grist_scores:
    print(f"Date: {day}, Grist Score: {score}, Sleep data -> [{stats['sleep_count']}, {stats['sleep_disturbances']}]")
