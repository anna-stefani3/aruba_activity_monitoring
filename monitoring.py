import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

# Load the dataset
df = pd.read_csv("[ARUBA]-activities_fixed_interval_data.csv")

print("Data loaded successfully:")
print(df.head())  # Check if data is loaded correctly
input("\nPress Enter to continue...")

# Assuming the two columns are "Time" and "activity", if they are concatenated, try to split them
if len(df.columns) == 1:  # If there's only one column due to incorrect parsing
    df = df["Time"].str.split(",", expand=True)
    df.columns = ["Time", "activity"]  # Rename the columns after splitting

print("\nColumn names after checking:")
print(df.columns)
input("\nPress Enter to continue...")

# Convert Time column to datetime format
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")  # Use 'coerce' to handle any invalid datetime formats

# Drop any rows where the 'Time' or 'activity' is missing
df.dropna(subset=["Time", "activity"], inplace=True)

print("\nData after dropping missing values:")
print(df.head())  # Print a subset
input("\nPress Enter to continue...")

# Set 'Time' as the index for easier resampling and time-based operations
df.set_index("Time", inplace=True)

# Ensure the data is at a 1-minute interval (forward fill if there are any gaps)
df_resampled = df.resample("1T").ffill()

print("\nResampled data with 1-minute intervals:")
print(df_resampled.head())  # Show first few rows of resampled data
input("\nPress Enter to continue...")

# Aggregate data on a per-day basis
# Count occurrences of each activity per day
df_daily = df_resampled.groupby(df_resampled.index.date).apply(lambda x: x["activity"].value_counts())

print("\nDaily aggregated data (activity counts per day):")
print(df_daily.head())  # Show first few rows of aggregated daily data
input("\nPress Enter to continue...")


# Define specific metrics like sleep_count and sleep disturbances
def compute_daily_stats(x):
    return pd.Series(
        {
            "sleep_count": (x["activity"] == "Sleeping").sum(),
            "sleep_disturbances": (x["activity"] == "Bed_to_Toilet").sum(),
            "meal_preparation_count": (x["activity"] == "Meal_Preparation").sum(),
            "eating_count": (x["activity"] == "Eating").sum(),
        }
    )


df_daily_stats = df_resampled.groupby(df_resampled.index.date).apply(compute_daily_stats)

print("\nDaily stats calculated:")
print(df_daily_stats.head())  # Show first few rows of daily stats
input("\nPress Enter to continue...")

# Determine the split date based on 80% of the total dates
unique_dates = df_daily_stats.index
total_dates = len(unique_dates)
split_point_index = int(total_dates * 0.8)  # 80% for training, 20% for monitoring
split_date = unique_dates[split_point_index]

print(f"\nDetermined split date: {split_date}")
input("\nPress Enter to continue...")

# Split data into training and monitoring based on the split date
df_train = df_daily_stats[df_daily_stats.index < split_date]  # Training data before the split date
df_monitor = df_daily_stats[df_daily_stats.index >= split_date]  # Monitoring data from the split date onward

print("\nTraining data sample:")
print(df_train.head())
print("\nMonitoring data sample:")
print(df_monitor.head())
input("\nPress Enter to continue...")


# Define a function to compute personalized profile (mean and covariance) based on daily stats
def compute_personal_profile(df):
    # Convert the daily stats into a NumPy matrix
    activity_matrix = df.to_numpy()

    # Return mean and covariance for this person's activity profile
    mean_vector = activity_matrix.mean(axis=0)  # Mean of daily stats
    cov_matrix = np.cov(activity_matrix, rowvar=False)  # Covariance of daily stats
    return mean_vector, cov_matrix


# Compute mean and covariance from training data
mean_vector_train, cov_matrix_train = compute_personal_profile(df_train)

print("\nPersonalized profile (mean and covariance) from training data:")
print(f"Mean vector: {mean_vector_train}")
print(f"Covariance matrix: {cov_matrix_train}")
input("\nPress Enter to continue...")


# Function to compute probability based on multivariate Gaussian distribution
def compute_likelihood(activity_vector, mean_vector, cov_matrix):
    dist = multivariate_normal(mean=mean_vector, cov=cov_matrix)
    likelihood = dist.pdf(activity_vector)
    return likelihood


# Anomaly detection function
def detect_anomaly(activity_vector, mean_vector, cov_matrix, threshold=0.001):
    likelihood = compute_likelihood(activity_vector, mean_vector, cov_matrix)
    # If likelihood is an array, check if any value is below the threshold
    if np.isscalar(likelihood):  # If it's a single value
        if likelihood < threshold:
            return "Anomaly Detected", likelihood
        else:
            return "Normal", likelihood
    else:  # If it's an array, check if all values are above or below the threshold
        if np.all(likelihood < threshold):
            return "Anomaly Detected", likelihood
        else:
            return "Normal", likelihood


# Compute generalized mean and covariance for the entire population (using training data)
def compute_general_model(df_population):
    activity_matrix = df_population.to_numpy()
    mean_vector_gen = activity_matrix.mean(axis=0)
    cov_matrix_gen = np.cov(activity_matrix, rowvar=False)
    return mean_vector_gen, cov_matrix_gen


# Scoring system
def update_grist_score(personal_status, generalized_status):
    if personal_status == "Anomaly Detected" and generalized_status == "Anomaly Detected":
        return "Critical"
    elif personal_status == "Anomaly Detected":
        return "Alert"
    else:
        return "Normal"


# Monitoring function that computes daily Grist Score for each day in the monitoring period
def monitor_activities_daily(df_monitor, mean_vector_train, cov_matrix_train, df_population):
    # Compute generalized mean and covariance for population
    mean_vector_gen, cov_matrix_gen = compute_general_model(df_population)

    scores = []
    for day, stats in df_monitor.iterrows():
        # Extract daily stats for this day
        activity_vector_monitor = stats.to_numpy()

        # Detect anomalies based on personal profile
        personal_status, likelihood_personal = detect_anomaly(
            activity_vector_monitor, mean_vector_train, cov_matrix_train
        )

        # Detect anomalies based on generalized population profile
        generalized_status, likelihood_general = detect_anomaly(
            activity_vector_monitor, mean_vector_gen, cov_matrix_gen
        )

        # Update grist score based on the statuses
        grist_score = update_grist_score(personal_status, generalized_status)
        scores.append((day, grist_score))

    return scores


# Example usage
daily_grist_scores = monitor_activities_daily(df_monitor, mean_vector_train, cov_matrix_train, df_train)

print("\nDaily Grist Scores:")
for day, score in daily_grist_scores:
    print(f"Date: {day}, Grist Score: {score}")

input("\nPress Enter to finish...")
