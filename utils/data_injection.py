import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import poisson


def generate_synthetic_data_using_gmm_and_poisson(original_data, n_components=3, n_samples_multiplier=10):
    """
    Generate synthetic data using Gaussian Mixture Model (GMM) for continuous features
    and Poisson distribution for discrete features.

    Parameters:
    original_data (pd.DataFrame): The original dataset with `sleep_duration` and `sleep_disturbances`.
    n_components (int): The number of components (clusters) for the GMM (for continuous features).
    n_samples_multiplier (int): The multiplier for the number of synthetic samples to generate.

    Returns:
    pd.DataFrame: The original dataset with appended synthetic data.
    """

    # Separate continuous and discrete columns
    continuous_columns = ["sleep_duration"]
    discrete_columns = ["sleep_disturbances"]

    # Fit a Gaussian Mixture Model (GMM) for continuous features
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(original_data[continuous_columns])

    # Generate synthetic data for continuous features (e.g., sleep_duration)
    n_samples = len(original_data) * n_samples_multiplier
    synthetic_continuous_data = gmm.sample(n_samples)[0]

    # Generate synthetic data for discrete features (e.g., sleep_disturbances) using Poisson distribution
    mean_disturbances = original_data["sleep_disturbances"].mean()  # Mean value of original disturbances
    synthetic_discrete_data = poisson.rvs(mu=mean_disturbances, size=n_samples)

    # Create a DataFrame for the synthetic data
    synthetic_df = pd.DataFrame(synthetic_continuous_data, columns=continuous_columns)
    synthetic_df["sleep_disturbances"] = synthetic_discrete_data  # Add the synthetic sleep_disturbances column

    # Combine original and synthetic data
    return pd.concat([original_data, synthetic_df], ignore_index=True)


def gaussian_based_inject_anomalies_continuous_days(
    test_df, feature_list, injections_count, min_number_days, max_number_days
):
    """
    Inject anomalies over continuous days in test data based on training stats.
    Introduces smaller, more gradual changes for a realistic behavioral shift.

    Parameters:
    - test_df: DataFrame with testing data (to inject anomalies).
    - feature_list: List of features to consider for anomaly injection.
    - injections_count: Number of times to inject anomalies.
    - min_number_days: Minimum number of continuous days for injection.
    - max_number_days: Maximum number of continuous days for injection.

    Returns:
    - DataFrame with injected anomalies.
    """
    # Calculate mean and std for each feature in training data
    stats = test_df[feature_list].agg(["mean", "std"]).T

    # Copy test data to avoid altering original
    test_injected = test_df.copy()

    for _ in range(injections_count):
        methods_list = ["increase_in_sleep_duration", "decrease_in_sleep_duration"]
        # Randomly choose a method from the methods_list
        method = np.random.choice(methods_list)

        # Randomly choose the number of days for this injection
        days_to_inject = np.random.randint(min_number_days, max_number_days + 1)

        # Randomly choose a starting row index in the test data, ensuring it fits within test data length
        start_row = np.random.choice(test_injected.index[:-days_to_inject])

        # Apply Gaussian noise with mean shift to simulate realistic anomaly
        if method == "increase_in_sleep_duration":
            # Use normal distribution to simulate realistic anomaly
            current_increament_size = 0

            for i in range(days_to_inject):
                increament_noise = np.random.uniform(0.1, 0.5)
                current_increament_size += increament_noise
                new_sleep_duration_value = (
                    stats.loc["sleep_duration", "mean"] + stats.loc["sleep_duration", "std"] + current_increament_size
                )

                # Ensure sleep duration stays within bounds [0, 24]
                test_injected.at[start_row + i, "sleep_duration"] = round(np.clip(new_sleep_duration_value, 0, 24), 2)
                test_injected.at[start_row + i, "label"] = 1  # Label as anomalous
        if method == "decrease_in_sleep_duration":
            # For sleep disturbances, increment probabilistically
            numbers = [0, 1, 2, 3]
            probabilities = np.array([0.75, 0.18, 0.05, 0.02])
            sleep_disturbance_threshold = 3

            disturbance_change = np.random.choice(numbers, size=days_to_inject, p=probabilities)
            current_decrement_size = 0
            for i in range(days_to_inject):
                decrement_noise = np.random.uniform(0.5, 0.25)
                current_decrement_size += decrement_noise
                new_sleep_duration_value = (
                    stats.loc["sleep_duration", "mean"] - stats.loc["sleep_duration", "std"] - current_decrement_size
                )
                new_sleep_disturbances_value = sleep_disturbance_threshold + disturbance_change[i]
                # Ensure a realistic range for sleep disturbances
                test_injected.at[start_row + i, "sleep_duration"] = round(np.clip(new_sleep_duration_value, 0, 24), 2)
                test_injected.at[start_row + i, "sleep_disturbances"] = round(
                    np.clip(new_sleep_disturbances_value, 0, 7), 0
                )
                test_injected.at[start_row + i, "label"] = 1  # Label as anomalous

    return test_injected
