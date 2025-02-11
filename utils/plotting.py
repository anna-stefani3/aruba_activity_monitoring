import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_histogram(df, column_name, plot_title="Histogram", num_bins: int = 24):
    """
    Plots a histogram for the specified column in the DataFrame with dynamic binning.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column name for which the histogram is to be plotted.

    Returns:
        None
    """
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Drop NaN values from the column
    data = df[column_name].dropna()

    # Calculate histogram data
    counts, bin_edges = np.histogram(data, bins=num_bins)
    bin_centers = range(0, (num_bins // 2) + 1, 1)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[0:-1], counts, width=(bin_edges[1] - bin_edges[0]), edgecolor="black", alpha=0.7)
    plt.title(plot_title)
    plt.xlabel(column_name)
    plt.ylabel("Count")
    plt.xticks(bin_centers, rotation=90)
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def plot_all_features_with_clusters(dataframe: pd.DataFrame, plot_title: str = "PLOT"):
    """
    Creates a single plot with subplots for all numerical features in the dataframe

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the data.
    """
    # Identify numeric features for plotting, excluding the 'cluster' column
    numeric_features = dataframe.select_dtypes(include="number").columns
    num_features = len(numeric_features)

    # Create a single figure with subplots
    fig, axes = plt.subplots(nrows=num_features, figsize=(10, 5 * num_features), sharey=True)
    fig.suptitle(plot_title)
    # Ensure axes is iterable even if there's only one feature
    if num_features == 1:
        axes = [axes]

    # Plot each feature in a subplot
    for ax, feature in zip(axes, numeric_features):
        sns.scatterplot(
            x=feature,
            y=dataframe.index,
            hue=feature,
            palette="viridis",
            data=dataframe,
            s=50,  # Marker size
            alpha=0.7,  # Transparency
            ax=ax,
        )
        # ax.set_title(f"{feature} Cluster Plotting")
        # ax.set_xlabel(feature)
        if feature == "sleep_duration":
            ax.set_xlim(0, 24)
        ax.set_ylabel("Index")
        # ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
