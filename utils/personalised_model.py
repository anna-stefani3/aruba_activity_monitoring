from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np


class PersonalisedModel:
    def __init__(
        self,
        training_data: pd.DataFrame,
        covariance_type: str = "spherical",
    ):
        self.training_data = training_data.drop("label", axis=1)
        self.scaler = StandardScaler()
        self.model = GaussianMixture(n_components=2, covariance_type=covariance_type, random_state=42)

    def apply_scaling(self):
        self.scaled_training_data = self.scaler.fit_transform(self.training_data.to_numpy())

    def train_gmm_model(self):
        self.apply_scaling()
        self.model.fit(self.scaled_training_data)

    def get_score(self, sample_data):
        score = self.model.score_samples(sample_data)
        return score

    def get_score_threshold(self):
        scores = self.model.score_samples(self.scaled_training_data)

        mean = np.mean(scores)
        std = np.std(scores)

        # Set the threshold at 2 standard deviations below the mean
        threshold = mean - (2 * std)
        return threshold


def execute_personalised_model(lagged_train_data, lagged_test_data):
    model = PersonalisedModel(lagged_train_data, covariance_type="spherical")
    model.train_gmm_model()

    scaler = model.scaler
    anomaly_threshold = model.get_score_threshold()

    predictions = [0] * lagged_test_data.shape[0]
    for i in range(lagged_test_data.shape[0]):
        day_data = lagged_test_data.iloc[i].drop("label")
        scaled_day_data = scaler.transform(day_data.values.reshape(1, -1))
        score = model.get_score(scaled_day_data)
        if score < anomaly_threshold:
            predictions[i] = 1

    # Calculate accuracy
    accuracy = accuracy_score(lagged_test_data["label"], predictions)
    print(f"Accuracy: {accuracy:.2f}")

    # Optionally, print classification report for more insights
    print(classification_report(lagged_test_data["label"], predictions))

    del model
    del scaler
    del anomaly_threshold
    del accuracy
