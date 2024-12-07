import pandas as pd
from scipy import stats
from .data_injection import (
    generate_synthetic_data_using_gmm_and_poisson,
    gaussian_based_inject_anomalies_continuous_days,
)
import numpy as np
from utils.plotting import plot_all_features_with_clusters


class Preprocessing:
    def __init__(self):
        pass

    @staticmethod
    def get_cleaned_dataframe(csvfile):
        df = pd.read_csv(csvfile)
        df["Time"] = pd.to_datetime(df["Time"])
        df.set_index("Time", inplace=True)
        df["Date"] = df.index.date
        return df

    @staticmethod
    def get_stats_dataframe(df):
        df_daily_stats = df.groupby(df.index.date).apply(Preprocessing().compute_daily_stats)
        return df_daily_stats

    @staticmethod
    def get_resampled_dataframe(df):
        df = generate_synthetic_data_using_gmm_and_poisson(df)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        plot_all_features_with_clusters(df, plot_title="GMM + POISSON BASED CLUSTERS")
        return df

    @staticmethod
    def apply_train_test_split(
        df: pd.DataFrame,
        split_index: int = 730,
    ):
        train_data = df.iloc[:split_index]
        test_data = df.iloc[split_index:]
        return train_data, test_data

    @staticmethod
    def get_injected_dataframe(
        test_data: pd.DataFrame,
        features=["sleep_duration", "sleep_disturbances"],
    ):
        test_data = gaussian_based_inject_anomalies_continuous_days(
            test_data,
            features,
            injections_count=10,
            min_number_days=7,
            max_number_days=20,
        )
        return test_data

    @staticmethod
    def clean_data_anomalies(df):
        stats = df["sleep_duration"].agg(["mean", "std"]).T
        df["sleep_duration"] = df["sleep_duration"].apply(
            lambda x: (
                stats["mean"] if x < stats["mean"] - 2 * stats["std"] or x > stats["mean"] + 2 * stats["std"] else x
            )
        )

        # Clean the 'sleep_disturbance' column by capping values at 3
        df["sleep_disturbances"] = df["sleep_disturbances"].apply(lambda x: 3 if x > 3 else x)
        return df

    @staticmethod
    def time_to_decimal(t):
        return round(t.hour + t.minute / 60, 2)

    @staticmethod
    def compute_daily_stats(current_row):

        previous_activity = current_row["activity"].shift(1)

        # Count transitions from Sleeping to Bed_to_Toilet
        sleep_to_bed_to_toilet = (
            (current_row["activity"] == "Bed_to_Toilet") & (previous_activity == "Sleeping")
        ).sum()

        return pd.Series(
            {
                "sleep_duration": round((current_row["activity"] == "Sleeping").sum() / 60, 2),
                "sleep_disturbances": sleep_to_bed_to_toilet,
            }
        )


def perform_cleaning_resampling_splitting_and_data_injection(
    filename: str, split_index: int = 730, features=["sleep_duration", "sleep_disturbances"]
):
    preprocessing = Preprocessing()
    df = preprocessing.get_cleaned_dataframe(filename)

    df = preprocessing.get_stats_dataframe(df)
    plot_all_features_with_clusters(df, plot_title="ORIGINAL DATA PLOT")

    df = preprocessing.clean_data_anomalies(df)
    plot_all_features_with_clusters(df, plot_title="ORIGINAL DATA AFTER CLEANING ANOMALIES")

    df = preprocessing.get_resampled_dataframe(df)

    train_data, test_data = preprocessing.apply_train_test_split(df, split_index=split_index)
    train_data = train_data.copy()
    test_data = test_data.copy()
    train_data["label"] = 0  # 0 represent normal and 1 will represent abnormal
    test_data["label"] = 0  # 0 represent normal and 1 will represent abnormal
    injected_test_data = preprocessing.get_injected_dataframe(test_data, features=features)
    plot_all_features_with_clusters(injected_test_data, plot_title="Injected Data Plotting")
    return train_data, injected_test_data
