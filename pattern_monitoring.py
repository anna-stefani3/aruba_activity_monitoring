# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.mixture import GaussianMixture
import numpy as np
from datetime import datetime

# Step 1: Load Data
print("Step 1: Loading the dataset")
file_path = "[ARUBA]-activities_fixed_interval_data.csv"
df = pd.read_csv(file_path)

# Overview of the dataset
print("Initial overview of dataset:")
print(df.head())
print(df.info())
input("Press Enter to continue to the next step...")

# Step 2: Preprocessing the Data
print("\nStep 2: Preprocessing Data")
# Convert 'Time' column to datetime format
df["Time"] = pd.to_datetime(df["Time"])

# Checking for missing values
missing_values = df.isnull().sum()
print(f"Missing values in each column:\n{missing_values}")
input("Press Enter to continue after reviewing missing values...")

# Fill missing values (if any)
if missing_values.any():
    df = df.fillna(method="ffill")  # Forward fill any missing values for simplicity
    print("Missing values have been filled using forward fill method.")
else:
    print("No missing values detected.")

input("Press Enter to continue after missing value handling...")

# Step 3: Activity Frequency Count
print("\nStep 3: Analyzing Activity Frequency")
activity_counts = df["activity"].value_counts()
print("Activity frequency counts:\n", activity_counts)

# Plot the activity frequency
plt.figure(figsize=(10, 6))
sns.barplot(x=activity_counts.index, y=activity_counts.values)
plt.title("Activity Frequency")
plt.xticks(rotation=45)
plt.ylabel("Count")
plt.xlabel("Activity")
plt.tight_layout()
plt.show()
input("Press Enter to continue after viewing the activity frequency plot...")

# Step 4: Time Segmentation by Day
print("\nStep 4: Time Segmentation by Day")
# Extract date and hour from 'Time' for day-based analysis
df["date"] = df["Time"].dt.date
df["hour"] = df["Time"].dt.hour
df["minute"] = df["Time"].dt.minute

# Aggregating data by day and checking activity per day
daywise_activity = df.groupby(["date", "activity"]).size().unstack(fill_value=0)
print("Day-wise distribution of activities:\n", daywise_activity.head())

# Plot daily activity distribution for top activities
plt.figure(figsize=(12, 8))
for activity in ["Sleeping", "Meal_Preparation", "Relax", "Work"]:
    sns.lineplot(data=daywise_activity[activity], label=activity)
plt.title("Daily Distribution of Key Activities")
plt.ylabel("Count")
plt.xlabel("Day")
plt.legend()
plt.tight_layout()
plt.show()

input("Press Enter to continue after reviewing the daily activity distribution plot...")

# Step 5: Day-wise Sleep Patterns
print("\nStep 5: Identifying Day-wise Sleep Patterns")

# Filtering out 'Sleeping' activity data
sleep_data = df[df["activity"] == "Sleeping"]

# Aggregating sleep duration per day
sleep_data["sleep_duration"] = sleep_data.groupby("date")["Time"].diff().dt.total_seconds().fillna(0)
daily_sleep_duration = sleep_data.groupby("date")["sleep_duration"].sum() / 3600  # in hours

# Extracting time of first sleep activity each day (sleep onset time)
sleep_onset_time = sleep_data.groupby("date")["Time"].first().dt.time

# Plot sleep duration per day
plt.figure(figsize=(10, 6))
sns.barplot(x=daily_sleep_duration.index, y=daily_sleep_duration.values)
plt.title("Daily Sleep Duration (Hours)")
plt.xticks(rotation=45)
plt.ylabel("Total Sleep Hours")
plt.xlabel("Day")
plt.tight_layout()
plt.show()

print(f"Daily sleep duration (in hours):\n{daily_sleep_duration}")
print(f"Time at which sleep starts each day:\n{sleep_onset_time}")
input("Press Enter to continue after reviewing daily sleep data...")

# Step 6: Sleep Disturbance Count per Day
print("\nStep 6: Identifying Day-wise Sleep Disturbances")

# Assuming 'Bed_to_Toilet' represents sleep disturbance
disturbance_data = df[df["activity"] == "Bed_to_Toilet"]

# Count the number of sleep disturbances per day
daily_disturbance_count = disturbance_data.groupby("date").size()

# Plot sleep disturbance count per day
plt.figure(figsize=(10, 6))
sns.barplot(x=daily_disturbance_count.index, y=daily_disturbance_count.values)
plt.title("Daily Sleep Disturbance Count")
plt.xticks(rotation=45)
plt.ylabel("Disturbance Count")
plt.xlabel("Day")
plt.tight_layout()
plt.show()

print(f"Daily sleep disturbance count:\n{daily_disturbance_count}")
input("Press Enter to continue after reviewing sleep disturbance data...")

# Step 7: Gaussian Mixture Model (for activity clustering)
print("\nStep 7: Applying Gaussian Mixture Model for activity clustering")
# Creating a feature matrix for Gaussian Mixture Model (based on hour and activity)
df["activity_encoded"] = pd.Categorical(df["activity"]).codes
X = df[["hour", "activity_encoded"]].to_numpy()

# Fitting a Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
gmm.fit(X)

# Predict clusters
df["cluster"] = gmm.predict(X)

# Display a sample of the clustering results
print("Sample of clustering results:")
print(df[["Time", "activity", "hour", "cluster"]].head())
input("Press Enter to continue after reviewing the clustering results...")

# Step 8: Visualizing Clusters
print("\nStep 8: Visualizing Clusters")
plt.figure(figsize=(10, 6))
sns.scatterplot(x="hour", y="activity_encoded", hue="cluster", data=df, palette="tab10")
plt.title("Activity Clusters based on Gaussian Mixture Model")
plt.xlabel("Hour of the Day")
plt.ylabel("Encoded Activity")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

input("Press Enter to continue after viewing the clustering visualization...")

# Step 9: Normality Testing
print("\nStep 9: Normality Testing")
# Perform normality test (Shapiro-Wilk) on sleep duration (or any other feature)
shapiro_test = stats.shapiro(daily_sleep_duration)
print(f"Shapiro-Wilk test statistic: {shapiro_test.statistic}, p-value: {shapiro_test.pvalue}")
if shapiro_test.pvalue > 0.05:
    print("The daily sleep duration follows a normal distribution.")
else:
    print("The daily sleep duration does not follow a normal distribution.")

input("Press Enter to continue after reviewing the normality test results...")

# Step 10: Applying Rule-based Detection for Sleep Anomalies
print("\nStep 10: Rule-based Anomaly Detection for Sleep Behavior")
# Define thresholds (these can be changed based on specific needs)
sleep_threshold = 7  # hours
disturbance_threshold = 3  # arbitrary disturbance count threshold

# Checking if sleep duration per day is below threshold
anomalies = daily_sleep_duration[daily_sleep_duration < sleep_threshold]
if not anomalies.empty:
    print(f"Warning: Sleep duration below {sleep_threshold} hours on these days:\n{anomalies}")
else:
    print(f"Sleep duration is within normal range on all days.")

# Checking for sleep disturbance anomalies (more than 3 disturbances in a day)
disturbance_anomalies = daily_disturbance_count[daily_disturbance_count > disturbance_threshold]
if not disturbance_anomalies.empty:
    print(f"Warning: More than {disturbance_threshold} disturbances detected on these days:\n{disturbance_anomalies}")
else:
    print("Disturbances are within acceptable range.")

input("Press Enter to complete the analysis...")

# Final Step: Wrap-up and Insights
print("\nStep 11: Analysis Complete")
print(
    "You have completed the analysis. Review the above steps and results to refine or rerun any specific parts of the process."
)
