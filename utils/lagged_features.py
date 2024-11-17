import pandas as pd


def get_lagged_features_dataframe(df, features_list):
    lagged_data = pd.DataFrame()

    # Loop to create lagged columns
    for i in range(1, 6):
        for feature in features_list:
            lagged_data[f"{feature}_{i}"] = df[feature].shift(i)

    # Label each row based on an aggregate over the window
    lagged_data["label"] = df["label"].rolling(5).apply(lambda x: 1 if x.sum() > 2 else 0).shift(-4)

    # Drop rows with NaN values due to shifting
    lagged_data.dropna(inplace=True)

    return lagged_data.reset_index(drop=True)
