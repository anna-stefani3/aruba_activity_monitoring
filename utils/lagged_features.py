import pandas as pd


def get_lagged_features_dataframe(df, feature_name):
    lagged_data = pd.DataFrame()

    # Loop to create lagged columns
    for i in range(1, 6):
        lagged_data[f"{feature_name}_{i}"] = df[feature_name].shift(i)
    # Drop rows with NaN values due to shifting
    lagged_data.dropna(inplace=True)

    return lagged_data.reset_index(drop=True)
