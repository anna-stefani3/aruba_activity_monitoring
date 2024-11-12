from utils.preprocessing import perform_cleaning_resampling_splitting_and_data_injection
from utils.lagged_features import get_lagged_features_dataframe
from utils.personalised_model import PersonalisedModel

filename = "aruba_dataset.csv"
train_data, test_data = perform_cleaning_resampling_splitting_and_data_injection(filename=filename)

sleep_duration_train_data = get_lagged_features_dataframe(train_data, "sleep_duration")
sleep_duration_test_data = get_lagged_features_dataframe(test_data, "sleep_duration")


print(sleep_duration_train_data.head())
print(sleep_duration_test_data.head())


model = PersonalisedModel(sleep_duration_train_data, sleep_duration_test_data, covariance_type="spherical")
model.train_gmm_model()

scaler = model.scaler
pdf_based_threshold = model.get_score_threshold()

for i in range(sleep_duration_test_data.shape[0]):
    day_data = sleep_duration_test_data.iloc[i]
    scaled_day_data = scaler.transform(day_data)
    score = model.get_score(scaled_day_data)
