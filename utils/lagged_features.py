import pandas as pd


def get_lagged_features_dataframe(df, features_list, lag_size=5):
    lagged_data = pd.DataFrame()

    # Loop to create lagged columns
    for i in range(1, lag_size + 1):
        for feature in features_list:
            lagged_data[f"{feature}_{i}"] = df[feature].shift(i)

    # Label each row based on the label value for the lagged row
    lagged_data["label"] = df["label"].shift(lag_size)

    # Drop rows with NaN values due to shifting
    lagged_data.dropna(inplace=True)

    return lagged_data.reset_index(drop=True)
