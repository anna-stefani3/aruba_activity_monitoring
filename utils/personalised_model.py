from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class PersonalisedModel:
    def __init__(
        self,
        training_data: pd.DataFrame,
        testing_data: pd.DataFrame,
        covariance_type: str = "spherical",
    ):
        self.training_data = training_data
        self.testing_data = testing_data
        self.scaler = StandardScaler()
        self.model = GaussianMixture(n_components=2, covariance_type=covariance_type, random_state=42)

    def apply_scaling(self):
        self.scaled_training_data = self.scaler.fit_transform(self.training_data)
        self.scaled_testing_data = self.scaler.transform(self.testing_data)

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
