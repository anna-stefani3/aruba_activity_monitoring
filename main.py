from utils.preprocessing import perform_cleaning_resampling_splitting_and_data_injection
from utils.lagged_features import get_lagged_features_dataframe

from utils.personalised_model import execute_personalised_model

filename = "aruba_dataset.csv"
train_data, test_data = perform_cleaning_resampling_splitting_and_data_injection(filename=filename)

sleep_features = ["sleep_duration", "sleep_disturbances"]

lagged_train_data = get_lagged_features_dataframe(train_data, sleep_features)
lagged_test_data = get_lagged_features_dataframe(test_data, sleep_features)


execute_personalised_model(train_data, test_data)  # without Lagged Features
execute_personalised_model(lagged_train_data, lagged_test_data)  # with Lagged Features
