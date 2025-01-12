from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from utils.generalised_model import GeneralisedModel


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

    def get_percentile_score_threshold(self, percentile=5):
        """
        Calculate the anomaly threshold based on the specified percentile of the score distribution.

        Args:
            percentile (int): The percentile value to set the threshold. Default is 5 (lower 5%).

        Returns:
            float: The anomaly threshold.
        """
        scores = self.model.score_samples(self.scaled_training_data)

        # Calculate the threshold based on the specified percentile
        threshold = np.percentile(scores, percentile)
        return threshold


def execute_personalised_model(train_data, test_data):
    model = PersonalisedModel(train_data, covariance_type="spherical")
    model.train_gmm_model()

    scaler = model.scaler
    anomaly_threshold = model.get_percentile_score_threshold()

    predictions = [0] * test_data.shape[0]
    for i in range(test_data.shape[0]):
        day_data = test_data.iloc[i].drop("label")
        scaled_day_data = scaler.transform(day_data.values.reshape(1, -1))
        score = model.get_score(scaled_day_data)
        if score < anomaly_threshold:
            predictions[i] = 1

    # Calculate accuracy
    accuracy = accuracy_score(test_data["label"], predictions)
    print(f"Accuracy: {accuracy:.2f}")

    # Optionally, print classification report for more insights
    print(classification_report(test_data["label"], predictions))

    del model
    del scaler
    del anomaly_threshold
    del accuracy


def execute_example_flow(train_data, test_data):
    model = PersonalisedModel(train_data, covariance_type="spherical")
    model.train_gmm_model()

    scaler = model.scaler
    anomaly_threshold = model.get_percentile_score_threshold()
    print(f"Personalised Model Anomaly Threshold: {anomaly_threshold}")

    predictions = [0] * test_data.shape[0]
    generalised_model = GeneralisedModel()
    header = "| Sleep Duration | Sleep Disturbances | Personalised Label | Generalised Sleep Duration Score | Generalised Sleep Duration Label | Generalised Sleep Disturbances Score | Generalised Sleep Disturbances Label |"
    print(header)
    print("-" * len(header))
    for i in range(test_data.shape[0]):
        day_data = test_data.iloc[i]
        scaled_day_data = scaler.transform(day_data.values.reshape(1, -1))
        score = model.get_score(scaled_day_data)
        output = "Normal"

        if score < anomaly_threshold:
            predictions[i] = 1
            output = "Anomaly"
            sdu_score, _ = generalised_model.get_sleep_duration_score(day_data["sleep_duration"])
            sdu_label = generalised_model.get_sleep_duration_label(day_data["sleep_duration"])

            sdi_score, _ = generalised_model.get_sleep_disturbance_score(day_data["sleep_disturbances"])
            sdi_label = generalised_model.get_sleep_disturbance_label(day_data["sleep_disturbances"])
            print(
                f"| {str(day_data['sleep_duration']):>13}  |{str(day_data['sleep_disturbances']):>18}  |{output:>18}  |{str(sdu_score):>32}  |{sdu_label:>32}  |{str(sdi_score):>36}  |{sdi_label:>36}  |"
            )
        else:
            print(
                f"| {str(day_data['sleep_duration']):>13}  |{str(day_data['sleep_disturbances']):>18}  |{output:>18}  |{'N/A':>32}  |{'N/A':>32}  |{'N/A':>36}  |{'N/A':>36}  |"
            )
    print("-" * len(header))
    del model
    del scaler
    del anomaly_threshold
