import pandas as pd
from scipy import stats
from .data_injection import generate_synthetic_data_using_hmm, inject_anomalies_continuous_days


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
        df = generate_synthetic_data_using_hmm(df)
        df = df.fillna(0)
        return df

    @staticmethod
    def get_injected_dataframe(
        df: pd.DataFrame,
        split_index: int = 730,
        features: str | list[str] = ["sleep_duration", "sleep_disturbances"],
    ):
        train_data = df.iloc[:split_index]
        test_data = df.iloc[split_index:]
        test_data = inject_anomalies_continuous_days(
            train_data,
            test_data,
            features,
            injections_count=10,
            min_number_days=4,
            max_number_days=20,
        )
        return train_data, test_data

    @staticmethod
    def clean_data_anomalies(df):
        # Clean the 'sleep_duration' column by replacing values outside the range [4, 12]
        # with the median value of the column.
        median_val = df["sleep_duration"].median()
        df["sleep_duration"] = df["sleep_duration"].apply(lambda x: median_val if x < 4 or x > 12 else x)

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
        eating_count = ((current_row["activity"] == "Eating") & (previous_activity != "Eating")).sum()
        cooking_count = (
            (current_row["activity"] == "Meal_Preparation") & (previous_activity != "Meal_Preparation")
        ).sum()

        # Determine the sleep start time using vectorized operations
        sleep_start_indices = (current_row["activity"] == "Sleeping") & ~previous_activity.isin(
            ["Sleeping", "Bed_to_Toilet"]
        )
        sleep_start_time = None
        if sleep_start_indices.any():
            sleep_start_time = Preprocessing().time_to_decimal(
                current_row.index[sleep_start_indices][0]
            )  # Get first occurrence

        # Determine the wake-up time using vectorized operations
        wake_up_indices = ~current_row["activity"].isin(["Sleeping", "Bed_to_Toilet"]) & (
            previous_activity == "Sleeping"
        )
        wake_up_time = None
        if wake_up_indices.any():
            wake_up_time = Preprocessing().time_to_decimal(
                current_row.index[wake_up_indices][0]
            )  # Get first occurrence

        return pd.Series(
            {
                "sleep_duration": round((current_row["activity"] == "Sleeping").sum() / 60, 2),
                "sleep_disturbances": sleep_to_bed_to_toilet,
                ## removing other features for simnplifying the process
                # "sleep_start_time": sleep_start_time,
                # "wake_up_time": wake_up_time,
                # "eating_count": eating_count,
                # "cooking_count": cooking_count,
                # "active_duration": round(
                #     (current_row["activity"].isin(["Meal_Preparation", "Wash_Dishes", "Housekeeping"])).sum() / 60, 2
                # ),
            }
        )


def perform_cleaning_resampling_splitting_and_data_injection(
    filename: str, split_index: int = 730, features=["sleep_duration", "sleep_disturbances"]
):
    preprocessing = Preprocessing()
    df = preprocessing.get_cleaned_dataframe(filename)
    df = preprocessing.get_stats_dataframe(df)
    df = preprocessing.clean_data_anomalies(df)
    df = preprocessing.get_resampled_dataframe(df)
    df["label"] = 0  # 0 represent normal and 1 will represent abnormal
    train_data, test_data = preprocessing.get_injected_dataframe(df, split_index=split_index, features=features)
    return train_data, test_data
