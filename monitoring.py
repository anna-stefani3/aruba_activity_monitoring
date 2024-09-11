import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

# Load the dataset
df = pd.read_csv("[ARUBA]-activities_fixed_interval_data.csv")

# Convert Time column to datetime format
df["Time"] = pd.to_datetime(df["Time"])

# Set 'Time' as the index for easier resampling and time-based operations
df.set_index("Time", inplace=True)

# Filter only relevant activities for health analysis
activities_of_interest = ["Sleeping", "Meal_Preparation", "Bed_to_Toilet", "Eating"]
df_filtered = df[df["activity"].isin(activities_of_interest)]

# Resample data to ensure each activity is reported in consistent 1-minute intervals
df_resampled = df_filtered.resample("1T").ffill()

# Compute total duration for each activity
activity_duration = df_resampled.groupby("activity").size()

# Calculate start and end times for relevant activities
activity_time_features = (
    df_filtered.groupby("activity")
    .apply(
        lambda group: {
            "start_time": group.index.min(),
            "end_time": group.index.max(),
            "total_duration_minutes": (group.index.max() - group.index.min()).total_seconds() / 60,
        }
    )
    .apply(pd.Series)
)


# Function to compute personalized profile (mean and covariance)
def compute_personal_profile(df):
    # Features of interest: duration of sleeping, eating, bed_to_toilet, and cooking
    feature_columns = ["Sleeping_duration", "Meal_Preparation_duration", "Bed_to_Toilet_duration", "Eating_duration"]

    # Calculate mean and covariance for the activities
    # Assume df_personal is a DataFrame that contains historical activity durations for a week (past 7 days)
    df_personal = df.copy()  # Placeholder for actual past data
    mean_vector = df_personal.mean()
    cov_matrix = df_personal.cov()

    return mean_vector, cov_matrix


# Function to compute probability based on multivariate Gaussian distribution
def compute_likelihood(activity_vector, mean_vector, cov_matrix):
    dist = multivariate_normal(mean=mean_vector, cov=cov_matrix)
    likelihood = dist.pdf(activity_vector)
    return likelihood


# Define a threshold for likelihood to mark as abnormal
THRESHOLD = 0.01  # Tune this based on cross-validation


# Anomaly detection function
def detect_anomaly(activity_vector, mean_vector, cov_matrix, threshold=THRESHOLD):
    likelihood = compute_likelihood(activity_vector, mean_vector, cov_matrix)
    if likelihood < threshold:
        return "Anomaly Detected", likelihood
    else:
        return "Normal", likelihood


# Compute generalized mean and covariance for the entire population
def compute_general_model(df_population):
    mean_vector_gen = df_population.mean()
    cov_matrix_gen = df_population.cov()
    return mean_vector_gen, cov_matrix_gen


# Scoring system
def update_grist_score(personal_status, generalized_status):
    if personal_status == "Anomaly Detected" and generalized_status == "Anomaly Detected":
        return "Critical"
    elif personal_status == "Anomaly Detected":
        return "Alert"
    else:
        return "Normal"


# Full monitoring function to track activities and detect anomalies
def monitor_activities(df, df_population):
    # Compute the personalized profile
    mean_vector, cov_matrix = compute_personal_profile(df)

    # Compute the generalized profile
    mean_vector_gen, cov_matrix_gen = compute_general_model(df_population)

    # Assume we have the current day's activity vector
    current_activity_vector = [450, 60, 5, 30]  # Example

    # Detect anomalies
    personal_status, likelihood_personal = detect_anomaly(current_activity_vector, mean_vector, cov_matrix)
    generalized_status, likelihood_general = detect_anomaly(current_activity_vector, mean_vector_gen, cov_matrix_gen)

    # Update the grist score
    grist_score = update_grist_score(personal_status, generalized_status)

    return grist_score, likelihood_personal, likelihood_general


# Example run
grist_score, likelihood_personal, likelihood_general = monitor_activities(df_filtered, df_filtered)
print(
    f"Grist Score: {grist_score}, Likelihood (Personal): {likelihood_personal}, Likelihood (General): {likelihood_general}"
)
