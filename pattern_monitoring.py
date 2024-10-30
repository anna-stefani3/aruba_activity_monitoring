import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
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


# Define function to compute daily stats
def compute_daily_stats(current_row):
    previous_activity = current_row["activity"].shift(1)

    # Count transitions from Sleeping to Bed_to_Toilet
    sleep_to_bed_to_toilet = ((current_row["activity"] == "Bed_to_Toilet") & (previous_activity == "Sleeping")).sum()
    eating_count = ((current_row["activity"] == "Eating") & (previous_activity != "Eating")).sum()
    cooking_count = ((current_row["activity"] == "Meal_Preparation") & (previous_activity != "Meal_Preparation")).sum()

    # Determine the sleep start time using vectorized operations
    sleep_start_indices = (current_row["activity"] == "Sleeping") & ~previous_activity.isin(
        ["Sleeping", "Bed_to_Toilet"]
    )
    sleep_start_time = None
    if sleep_start_indices.any():
        sleep_start_time = time_to_decimal(current_row.index[sleep_start_indices][0])  # Get first occurrence

    # Determine the wake-up time using vectorized operations
    wake_up_indices = ~current_row["activity"].isin(["Sleeping", "Bed_to_Toilet"]) & (previous_activity == "Sleeping")
    wake_up_time = None
    if wake_up_indices.any():
        wake_up_time = time_to_decimal(current_row.index[wake_up_indices][0])  # Get first occurrence

    return pd.Series(
        {
            "sleep_duration": round((current_row["activity"] == "Sleeping").sum() / 60, 2),
            "sleep_disturbances": sleep_to_bed_to_toilet,
            "sleep_start_time": sleep_start_time,
            "wake_up_time": wake_up_time,
            "eating_count": eating_count,
            "cooking_count": cooking_count,
            "active_duration": round(
                (current_row["activity"].isin(["Meal_Preparation", "Wash_Dishes", "Housekeeping"])).sum() / 60, 2
            ),
        }
    )


# Step 2: Compute daily stats for all activities
df_daily_stats = df.groupby(df.index.date).apply(compute_daily_stats)
df_daily_stats.to_csv("daily_stats.csv", index=False)

# Print summary of daily aggregated data
print("Step 2: Date-wise aggregated data computed:")
print(df_daily_stats.head())
print("Rows: ", df_daily_stats.shape[0], " Columns: ", df_daily_stats.shape[1])
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
perform_normalcy_test(
    combo_data.sum(axis=1), """["sleep_duration", "sleep_disturbances", "eating_count", "cooking_count"]"""
)

# Test combinations of columns
combo_cols = ["sleep_duration", "sleep_disturbances"]
combo_data = df_daily_stats[combo_cols].dropna()
perform_normalcy_test(combo_data.sum(axis=1), """["sleep_duration", "sleep_disturbances"]""")


# Test combinations of columns
combo_cols = ["eating_count", "cooking_count"]
combo_data = df_daily_stats[combo_cols].dropna()
perform_normalcy_test(combo_data.sum(axis=1), """["eating_count", "cooking_count"]""")

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
sleeping_features = ["sleep_duration", "sleep_disturbances"]
eating_features = ["eating_count", "cooking_count"]

sleeping_scaler = StandardScaler()
sleeping_trained_scaled = sleeping_scaler.fit_transform(train_data[sleeping_features])

eating_scaler = StandardScaler()
eating_trained_scaled = eating_scaler.fit_transform(train_data[eating_features])

sleeping_gmm = GaussianMixture(n_components=2, random_state=42)
sleeping_gmm.fit(sleeping_trained_scaled)
eating_gmm = GaussianMixture(n_components=2, random_state=42)
eating_gmm.fit(eating_trained_scaled)

# Step 5: GMM model training completed
print("Step 5: GMM model training completed.")
input("Press Enter to continue...")


# Function to get_gmm_model_score
def get_gmm_model_score(data_point, gmm_model, scaler, feature_names):
    data_point_df = pd.DataFrame([data_point], columns=feature_names)
    scaled_point = scaler.transform(data_point_df)
    score = gmm_model.score_samples(scaled_point)
    return score[0]


recent_sleeping_anomaly_scores = []
recent_eating_anomaly_scores = []
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

    sleeping_anomaly_score = get_gmm_model_score(
        day_data.loc[sleeping_features], sleeping_gmm, sleeping_scaler, feature_names=sleeping_features
    )
    eating_anomaly_score = get_gmm_model_score(
        day_data.loc[eating_features], eating_gmm, eating_scaler, feature_names=eating_features
    )

    if len(recent_sleeping_anomaly_scores) >= window_size:
        recent_sleeping_anomaly_scores.pop(0)
    if len(recent_eating_anomaly_scores) >= window_size:
        recent_eating_anomaly_scores.pop(0)
    recent_sleeping_anomaly_scores.append(sleeping_anomaly_score)
    recent_eating_anomaly_scores.append(eating_anomaly_score)

    if len(recent_sleeping_anomaly_scores) == window_size:
        threshold = dynamic_threshold(recent_sleeping_anomaly_scores)

        if sleeping_anomaly_score < threshold:
            print(f"Day {str(i).rjust(3)} : {test_data.index[i]} - Abnormal")
            sleep_duration_score, quality_range = generalised_monitoring.get_sleep_duration_score(
                day_data[sleeping_features[0]]
            )
            sleep_disturbance_score, quality_range = generalised_monitoring.get_sleep_disturbance_score(
                day_data[sleeping_features[1]]
            )
            scores = {sleeping_features[0]: sleep_duration_score, sleeping_features[1]: sleep_disturbance_score}
            question_1_score = generalised_monitoring.get_egrist_score(generalised_monitoring.question_1, scores)
            question_2_score = generalised_monitoring.get_egrist_score(generalised_monitoring.question_2, scores)
            print("SLEEPING ANOMALY")
            print(f"----  Question : {generalised_monitoring.question_1}    Score: {question_1_score}")
            print(f"----  Question : {generalised_monitoring.question_2}    Score: {question_2_score}")

            print("\n\n")
            alert_triggered = True

    if len(recent_eating_anomaly_scores) == window_size:
        threshold = dynamic_threshold(recent_eating_anomaly_scores)

        if eating_anomaly_score < threshold:
            print(f"Day {str(i).rjust(3)} : {test_data.index[i]} - Abnormal")
            eating_score, quality_range = generalised_monitoring.get_eating_count_score(day_data[eating_features[0]])
            cooking_score, quality_range = generalised_monitoring.get_cooking_count_score(day_data[eating_features[1]])
            scores = {eating_features[0]: eating_score, eating_features[1]: cooking_score}
            question_5_score = generalised_monitoring.get_egrist_score(generalised_monitoring.question_5, scores)
            print("EATING ANOMALY")
            print(f"----  Question : {generalised_monitoring.question_5}    Score: {question_5_score}")

            print("\n\n")
            alert_triggered = True

if not alert_triggered:
    print("No abnormal patterns detected over the simulated period.")
