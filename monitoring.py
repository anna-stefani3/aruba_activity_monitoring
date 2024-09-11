import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

# Load the dataset
df = pd.read_csv("aruba.csv")

# Verify the column names and check for any issues in the data
print(df.head())  # Check if data is loaded correctly

# Assuming the two columns are "Time" and "activity", if they are concatenated, try to split them
if len(df.columns) == 1:  # If there's only one column due to incorrect parsing
    df = df['Time'].str.split(',', expand=True)
    df.columns = ['Time', 'activity']  # Rename the columns after splitting

# Convert Time column to datetime format
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')  # Use 'coerce' to handle any invalid datetime formats

# Drop any rows where the 'Time' or 'activity' is missing
df.dropna(subset=['Time', 'activity'], inplace=True)

# Filter only relevant activities for health analysis
activities_of_interest = ['Sleeping', 'Meal_Preparation', 'Bed_to_Toilet', 'Eating']
df_filtered = df[df['activity'].isin(activities_of_interest)]

# Set 'Time' as the index for easier resampling and time-based operations
df_filtered.set_index('Time', inplace=True)

# Resample data to ensure each activity is reported in consistent 1-minute intervals
df_resampled = df_filtered.resample('1T').ffill()

# Split the dataset into training and monitoring sets
split_point = '2012-01-07'  # Example split point - adjust as necessary
df_train = df_resampled[:split_point]  # Training data before split
df_monitor = df_resampled[split_point:]  # Monitoring data after split

# Compute total duration for each activity in both training and monitoring periods
activity_duration_train = df_train.groupby('activity').size()
activity_duration_monitor = df_monitor.groupby('activity').size()

# Define a function to compute personalized profile (mean and covariance)
def compute_personal_profile(df):
    # Compute total duration for relevant activities
    activity_durations = df.groupby('activity').size().reindex(activities_of_interest, fill_value=0)

    # Create a vector of activity durations
    activity_vector = activity_durations.to_numpy()

    # Return mean and covariance for this person's activity profile
    mean_vector = activity_vector.mean()  # Mean of activity durations
    cov_matrix = np.cov(activity_vector, rowvar=False)  # Covariance of activity durations
    return mean_vector, cov_matrix

# Compute mean and covariance from training data
mean_vector_train, cov_matrix_train = compute_personal_profile(df_train)

# Function to compute probability based on multivariate Gaussian distribution
def compute_likelihood(activity_vector, mean_vector, cov_matrix):
    dist = multivariate_normal(mean=mean_vector, cov=cov_matrix)
    likelihood = dist.pdf(activity_vector)
    return likelihood

# Anomaly detection function
def detect_anomaly(activity_vector, mean_vector, cov_matrix, threshold=0.01):
    likelihood = compute_likelihood(activity_vector, mean_vector, cov_matrix)
    if likelihood < threshold:
        return "Anomaly Detected", likelihood
    else:
        return "Normal", likelihood

# Compute generalized mean and covariance for the entire population (could be other data)
def compute_general_model(df_population):
    mean_vector_gen = df_population.groupby('activity').size().reindex(activities_of_interest, fill_value=0).mean()
    cov_matrix_gen = np.cov(df_population.groupby('activity').size().reindex(activities_of_interest, fill_value=0), rowvar=False)
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
    activity_durations_monitor = df_monitor.groupby('activity').size().reindex(activities_of_interest, fill_value=0)
    activity_vector_monitor = activity_durations_monitor.to_numpy()

    # Detect anomalies based on personal profile
    personal_status, likelihood_personal = detect_anomaly(activity_vector_monitor, mean_vector_train, cov_matrix_train)

    # Detect anomalies based on generalized population profile
    generalized_status, likelihood_general = detect_anomaly(activity_vector_monitor, mean_vector_gen, cov_matrix_gen)

    # Update grist score based on the statuses
    grist_score = update_grist_score(personal_status, generalized_status)

    return grist_score, likelihood_personal, likelihood_general

# Example usage
grist_score, likelihood_personal, likelihood_general = monitor_activities(df_monitor, mean_vector_train, cov_matrix_train, df_resampled)
print(f"Grist Score: {grist_score}, Likelihood (Personal): {likelihood_personal}, Likelihood (General): {likelihood_general}")
