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

# Filter only relevant activities for health analysis
activities_of_interest = ["Sleeping", "Meal_Preparation", "Bed_to_Toilet", "Eating"]
df_filtered = df[df["activity"].isin(activities_of_interest)]

print("\nFiltered dataset (only health-related activities):")
print(df_filtered.head())  # Show first few rows of filtered data
input("\nPress Enter to continue...")

# Set 'Time' as the index for easier resampling and time-based operations
df_filtered.set_index("Time", inplace=True)

# Resample data to ensure each activity is reported in consistent 1-minute intervals
df_resampled = df_filtered.resample("1T").ffill()

print("\nResampled data with 1-minute intervals:")
print(df_resampled.head())  # Show first few rows of resampled data
input("\nPress Enter to continue...")

# Split the dataset into training and monitoring sets
split_point = "2012-01-07"  # Example split point - adjust as necessary
df_train = df_resampled[:split_point]  # Training data before split
df_monitor = df_resampled[split_point:]  # Monitoring data after split

print("\nTraining data sample:")
print(df_train.head())
print("\nMonitoring data sample:")
print(df_monitor.head())
input("\nPress Enter to continue...")

# Compute total duration for each activity in both training and monitoring periods
activity_duration_train = df_train.groupby("activity").size()
activity_duration_monitor = df_monitor.groupby("activity").size()

print("\nActivity duration in training data:")
print(activity_duration_train)
input("\nPress Enter to continue...")

print("\nActivity duration in monitoring data:")
print(activity_duration_monitor)
input("\nPress Enter to continue...")


# Define a function to compute personalized profile (mean and covariance)
def compute_personal_profile(df):
    # Compute total duration for relevant activities
    activity_durations = df.groupby("activity").size().reindex(activities_of_interest, fill_value=0)

    # Create a vector of activity durations
    activity_vector = activity_durations.to_numpy()

    # Return mean and covariance for this person's activity profile
    mean_vector = activity_vector.mean()  # Mean of activity durations
    cov_matrix = np.cov(activity_vector, rowvar=False)  # Covariance of activity durations
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
def detect_anomaly(activity_vector, mean_vector, cov_matrix, threshold=0.01):
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


# Compute generalized mean and covariance for the entire population (could be other data)
def compute_general_model(df_population):
    mean_vector_gen = df_population.groupby("activity").size().reindex(activities_of_interest, fill_value=0).mean()
    cov_matrix_gen = np.cov(
        df_population.groupby("activity").size().reindex(activities_of_interest, fill_value=0), rowvar=False
    )
    return mean_vector_gen, cov_matrix_gen


# Scoring system
def update_grist_score(personal_status, generalized_status):
    if personal_status == "Anomaly Detected" and generalized_status == "Anomaly Detected":
        return "Critical"
    elif personal_status == "Anomaly Detected":
        return "Alert"
    else:
        return "Normal"


# Monitoring function that uses monitoring data and the personal profile to detect anomalies
def monitor_activities(df_monitor, mean_vector_train, cov_matrix_train, df_population):
    # Compute generalized mean and covariance for population
    mean_vector_gen, cov_matrix_gen = compute_general_model(df_population)

    # Extract activity durations for the monitoring period
    activity_durations_monitor = df_monitor.groupby("activity").size().reindex(activities_of_interest, fill_value=0)
    activity_vector_monitor = activity_durations_monitor.to_numpy()

    # Detect anomalies based on personal profile
    personal_status, likelihood_personal = detect_anomaly(activity_vector_monitor, mean_vector_train, cov_matrix_train)

    # Detect anomalies based on generalized population profile
    generalized_status, likelihood_general = detect_anomaly(activity_vector_monitor, mean_vector_gen, cov_matrix_gen)

    # Update grist score based on the statuses
    grist_score = update_grist_score(personal_status, generalized_status)

    return grist_score, likelihood_personal, likelihood_general


# Example usage
grist_score, likelihood_personal, likelihood_general = monitor_activities(
    df_monitor, mean_vector_train, cov_matrix_train, df_resampled
)
print(f"\nGrist Score: {grist_score}")
print(f"Likelihood (Personal): {likelihood_personal}")
print(f"Likelihood (General): {likelihood_general}")
