import pandas as pd


def save_fixed_interval_activity_data(filename):
    """
    Reads activity data from a file, filters activities that have a 'begin' label,
    and then resamples the activity data to fixed 1-minute intervals. The resulting
    DataFrame is saved as a CSV file.

    Parameters:
    filename (str): The name of the input file containing the activity data.
    """

    # Read the input file with specified column names
    aruba_df = pd.read_csv(
        filename, delim_whitespace=True, header=None, names=["date", "time", "device", "status", "activity", "label"]
    )

    # Combine 'date' and 'time' columns into a single datetime column 'Time'
    aruba_df["Time"] = pd.to_datetime(aruba_df["date"] + " " + aruba_df["time"], errors="coerce")

    # Filter rows where the 'label' column is 'begin'
    filtered_aruba_df = aruba_df[aruba_df["label"] == "begin"].copy()

    # Select the 'Time' and 'activity' columns for further processing
    filtered_aruba_df = filtered_aruba_df[["Time", "activity"]]

    # Set the 'Time' column as the index
    filtered_aruba_df.set_index("Time", inplace=True)

    # Resample the data to 1-minute intervals, forward filling missing values
    fixed_interval_activity_df = filtered_aruba_df.resample("1T").ffill()

    # Drop any rows with NaN values and save the result to a CSV file
    fixed_interval_activity_df.dropna().to_csv("[ARUBA]-activities_fixed_interval_data.csv")


if __name__ == "__main__":
    filename = "activity_monitoring/ARUBA/data"
    save_fixed_interval_activity_data(filename)
