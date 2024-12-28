from utils.preprocessing import perform_cleaning_resampling_splitting_and_data_injection
from utils.lagged_features import get_lagged_features_dataframe
from utils.personalised_model import execute_personalised_model, execute_example_flow
from utils.dbscan import test_dbscan_clustering

import numpy as np
import pandas as pd
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
lag_size = 5
lagged_train_data = get_lagged_features_dataframe(train_data, sleep_features, lag_size=lag_size)
lagged_test_data = get_lagged_features_dataframe(test_data, sleep_features, lag_size=lag_size)


###### PERSONALISED MODEL ######
print("Accuracy Without Lagged Features")
# skipping first 4 items so that the number of number of data points are same

print(f"Number of rows in Without Lagged Feature Dataset : {train_data[lag_size - 1 :].shape[0]}")
print(f"Number of rows in Lagged Feature Dataset : {lagged_train_data.shape[0]}")
execute_personalised_model(train_data[lag_size - 1 :], test_data[lag_size - 1 :])  # without Lagged Features

print("Accuracy With Lagged Features")
execute_personalised_model(lagged_train_data, lagged_test_data)  # with Lagged Features


###### DBSCAN MODEL ######
print("\n\nWithout Lagged Features")
test_dbscan_clustering(test_data[sleep_features], test_data["label"])  # without Lagged Features
print("\n\nWith Lagged Features")
test_dbscan_clustering(lagged_test_data, lagged_test_data["label"])  # with Lagged Features


###### EXAMPLE EXECUTION ######

print("\n\nExample Output for Personalised and Generalised Models Together")
example_data = {
    "sleep_duration": [3, 3, 5, 5, 8, 8, 11, 11, 13, 13],
    "sleep_disturbances": [0, 8, 1, 6, 0, 4, 0, 5, 1, 4],
}
example_df = pd.DataFrame(example_data)


execute_example_flow(train_data[lag_size - 1 :], example_df)
