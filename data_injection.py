import pandas as pd
import numpy as np
from hmmlearn import hmm


def generate_synthetic_data_using_hmm(original_data):
    """
    Use HMM to generate synthetic data and append it to the original data.

    Parameters:
    original_data (pandas.DataFrame): The original dataset.

    Returns:
    pandas.DataFrame: The original dataset with appended synthetic data.
    """
    # Prepare the data for HMM training
    activities = np.array(original_data)
    # Create a boolean mask for rows with NaN values
    activities = np.nan_to_num(activities, nan=0)

    # Train the HMM
    model = hmm.GaussianHMM(n_components=20, random_state=42)
    model.fit(activities.reshape(-1, 1))

    # Generate synthetic data using the trained HMM
    synthetic_length = len(original_data) * 10  # 10 Times the original data
    synthetic_X = model.sample(synthetic_length)[0].flatten()
    synthetic_activities = [activities[int(x)] for x in synthetic_X]
    synthetic_data = pd.DataFrame(synthetic_activities, columns=original_data.columns)

    # Combine the original and synthetic data
    return pd.concat([original_data, synthetic_data], ignore_index=True)


def inject_anomalies_day_wise(
    train_df,
    test_df,
    feature_list,
    injections_count,
    min_std=3,
    max_std=10,
):
    """
    Inject anomalies into test data based on statistics from training data.

    Parameters:
    - train_df: DataFrame with training data (used for calculating stats).
    - test_df: DataFrame with testing data (to inject anomalies).
    - feature_list: List of features to consider for anomaly injection.
    - injections_count: Number of anomalies to inject.
    - min_std: Minimum Multiplier for standard deviation to control anomaly intensity.
    - max_std: maximum Multiplier for standard deviation to control anomaly intensity.

    Returns:
    - DataFrame with injected anomalies.
    """
    # Calculate mean and std for each feature in training data
    stats = train_df[feature_list].agg(["mean", "std"]).T

    # Copy test data to avoid altering original
    test_injected = test_df.copy()

    for _ in range(injections_count):
        # Randomly choose a feature from the list
        feature = np.random.choice(feature_list)

        # Randomly choose a row index in the test data
        row_id = np.random.choice(test_injected.index)

        # randomly generating std_dev_multiplier (min_std represents the minimum accepted std)
        std_dev_multiplier = min_std + ((max_std - min_std) * np.random.rand())

        # Inject anomaly: using mean and std from training data
        anomaly_value = stats.loc[feature, "mean"] + std_dev_multiplier * stats.loc[feature, "std"] * np.random.choice(
            [-1, 1]
        )
        if feature != "sleep_disturbances" and "count" not in feature:
            test_injected.at[row_id, feature] = round(anomaly_value, 2)
        else:
            test_injected.at[row_id, feature] = round(anomaly_value, 0)
    return test_injected


def inject_anomalies_continuous_days(
    train_df,
    test_df,
    feature_list,
    injections_count,
    min_number_days,
    max_number_days,
    min_std=3,
    max_std=10,
):
    """
    Inject anomalies over continuous days in test data based on training stats.

    Parameters:
    - train_df: DataFrame with training data (used for calculating stats).
    - test_df: DataFrame with testing data (to inject anomalies).
    - feature_list: List of features to consider for anomaly injection.
    - injections_count: Number of times to inject anomalies.
    - min_number_days: Minimum number of continuous days for injection.
    - max_number_days: Maximum number of continuous days for injection.
    - min_std: Minimum Multiplier for standard deviation to control anomaly intensity.
    - max_std: maximum Multiplier for standard deviation to control anomaly intensity.

    Returns:
    - DataFrame with injected anomalies.
    """
    # Calculate mean and std for each feature in training data
    stats = train_df[feature_list].agg(["mean", "std"]).T

    # Copy test data to avoid altering original
    test_injected = test_df.copy()

    for _ in range(injections_count):
        # Randomly choose a feature from the list
        feature = np.random.choice(feature_list)

        # Randomly choose the number of days for this injection
        days_to_inject = np.random.randint(min_number_days, max_number_days + 1)

        # Randomly choose a starting row index in the test data, ensuring it fits within test data length
        start_row = np.random.choice(test_injected.index[:-days_to_inject])

        # randomly generating std_dev_multiplier (min_std represents the minimum accepted std)
        std_dev_multiplier = min_std + ((max_std - min_std) * np.random.rand())

        # Inject anomaly for a continuous range of days
        anomaly_value = stats.loc[feature, "mean"] + std_dev_multiplier * stats.loc[feature, "std"]
        percentage = 0.05  # 5%
        lower_bound = anomaly_value - (anomaly_value * percentage)
        upper_bound = anomaly_value + (anomaly_value * percentage)
        for i in range(days_to_inject):
            if feature != "sleep_disturbances" and "count" not in feature:
                test_injected.at[start_row + i, feature] = round(np.random.uniform(lower_bound, upper_bound), 2)
            else:
                test_injected.at[start_row + i, feature] = round(np.random.uniform(lower_bound, upper_bound), 0)

    return test_injected


if __name__ == "__main__":
    """
    TESTING INJECTION PROCESSES
    """
    # Part 1: Completed
    stats_df = pd.read_csv("daily_stats.csv")
    synthetic_stats_df = generate_synthetic_data_using_hmm(stats_df)
    synthetic_stats_df.to_csv("synthetic_stats_df.csv", index=False)

    # Part 2: Completed
    train_df = synthetic_stats_df[:730]
    test_df = synthetic_stats_df[730:]
    feature_list = ["sleep_duration", "sleep_disturbances"]
    injections_count = 300
    injected_synthetic_stats_df = inject_anomalies_day_wise(train_df, test_df, feature_list, injections_count)
    injected_synthetic_stats_df.to_csv("injected_synthetic_stats_df.csv", index=False)

    # Part 3: Completed
    min_number_days = 2
    max_number_days = 30
    injections_count = 50
    continous_injected_synthetic_stats_df = inject_anomalies_continuous_days(
        train_df, test_df, feature_list, injections_count, min_number_days, max_number_days
    )
    continous_injected_synthetic_stats_df.to_csv("continous_injected_synthetic_stats_df.csv", index=False)
