import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

# Load the dataset from a CSV file
df = pd.read_csv("[ARUBA]-activities_fixed_interval_data.csv")


def split_train_test(df, num_training_days):
    df["Time"] = pd.to_datetime(df["Time"])
    training_data = df[df["Time"] < df["Time"].min() + pd.Timedelta(days=num_training_days)]
    testing_data = df[df["Time"] >= df["Time"].min() + pd.Timedelta(days=num_training_days)]
    return training_data, testing_data


def encode_train_test(train_df, test_df):
    train_activities = train_df["activity"].values
    test_activities = test_df["activity"].values
    encoder = LabelEncoder()
    train_encoded = encoder.fit_transform(train_activities)
    test_encoded = encoder.transform(test_activities)
    return np.array(train_encoded), np.array(test_encoded), encoder


def remove_consecutive_duplicates(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")
    mask = df[column_name] != df[column_name].shift(1)
    result_df = df[mask].reset_index(drop=True)
    return result_df


def create_sequences_with_dates(df, encoded_data, window_size):
    sequences = []
    timestamps = []
    for i in range(len(encoded_data) - window_size + 1):
        sequence = encoded_data[i : i + window_size]
        sequence_dates = df["Time"].iloc[i : i + window_size].tolist()
        sequences.append(sequence)
        timestamps.append(sequence_dates)
    return np.array(sequences), timestamps


# Data preparation
df = remove_consecutive_duplicates(df, "activity")
training, testing = split_train_test(df, num_training_days=30)
train_encoded, test_encoded, encoder = encode_train_test(training, testing)

# Define the window size for sequences
activity_sequence_size = 4

# Create sequences with timestamps for both training and testing datasets
train_encoded_sequence, train_dates = create_sequences_with_dates(training, train_encoded, activity_sequence_size)
test_encoded_sequence, test_dates = create_sequences_with_dates(testing, test_encoded, activity_sequence_size)

# Initialize and train the HMM model
n_components = 10
model = hmm.MultinomialHMM(n_components=n_components, n_iter=1000, tol=0.01, random_state=42)
model.fit(train_encoded_sequence)


def calculate_dynamic_threshold(train_encoded_sequence, model):
    """
    Calculate a dynamic threshold based on log-likelihood scores.

    Parameters:
    - train_encoded_sequence: List of sequences where each sequence is an array of encoded data.
    - model: A trained model with a `score` method that returns log-likelihood.

    Returns:
    - threshold: The dynamically computed threshold value.
    """
    # Calculate log-likelihood scores for each sequence
    log_likelihoods = [model.score(seq.reshape(1, -1)) for seq in train_encoded_sequence]

    # Compute mean of the log-likelihood scores
    mean_ll = np.mean(log_likelihoods)

    # Compute standard deviation of the log-likelihood scores
    std_ll = np.std(log_likelihoods)

    # Set the multiplier for the standard deviation
    k = 2

    # Calculate the threshold as mean minus k times the standard deviation
    threshold = mean_ll - k * std_ll

    return threshold


def detect_anomaly(sequence, model, threshold):
    sequence = np.array(sequence).reshape(1, -1)
    log_likelihood = model.score(sequence)
    return log_likelihood < threshold


anomaly_threshold = calculate_dynamic_threshold(train_encoded_sequence, model)
print(f"Anomaly Threshold = {round(anomaly_threshold,2)}")

# Check test sequences for anomalies and print corresponding dates
for sequence, dates in zip(test_encoded_sequence, test_dates):
    if detect_anomaly(sequence, model, anomaly_threshold):
        formated_activites_list = [item.ljust(18) for item in encoder.inverse_transform(sequence.flatten())]
        formated_activites_string = "".join(formated_activites_list)
        print(f"Anomaly [{dates[-1]}]: {formated_activites_string}")
