from utils.preprocessing import perform_cleaning_resampling_splitting_and_data_injection
from utils.lagged_features import get_lagged_features_dataframe

from utils.personalised_model import execute_personalised_model


import numpy as np
import random

seed = 42
random.seed(seed)
np.random.seed(seed)


filename = "aruba_dataset.csv"
train_data, test_data = perform_cleaning_resampling_splitting_and_data_injection(filename=filename)

sleep_duration_stats = train_data["sleep_duration"].agg(["mean", "std", "min", "max"]).T
print(
    f"\n\n==== Average Sleep Duration is {round(sleep_duration_stats['mean'],2)} Hours"
    f" and std is {round(sleep_duration_stats['std'],2)} Hours ====\n\n"
)

sleep_features = ["sleep_duration", "sleep_disturbances"]

lagged_train_data = get_lagged_features_dataframe(train_data, sleep_features, lag_size=5)
lagged_test_data = get_lagged_features_dataframe(test_data, sleep_features, lag_size=5)


print("Accuracy Without Lagged Features")
# skipping first 4 items so that the number of number of data points are same

print(f"Number of rows in Without Lagged Feature Dataset : {train_data[5:].shape[0]}")
print(f"Number of rows in Lagged Feature Dataset : {lagged_train_data.shape[0]}")
execute_personalised_model(train_data[5:], test_data[5:])  # without Lagged Features

print("Accuracy With Lagged Features")
execute_personalised_model(lagged_train_data, lagged_test_data)  # with Lagged Features
