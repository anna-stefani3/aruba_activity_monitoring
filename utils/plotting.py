import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_clusters(dataframe: pd.DataFrame, feature_name: str, cluster_column: str = "cluster"):
    """
    Plots clusters based on a specific feature.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the data.
        feature_name (str): The name of the feature to plot.
        cluster_column (str): The name of the column representing cluster labels. Default is 'cluster'.

    Raises:
        ValueError: If the feature_name or cluster_column is not in the dataframe.
    """
    # Validate input
    if feature_name not in dataframe.columns:
        raise ValueError(f"Feature '{feature_name}' not found in the dataframe.")
    if cluster_column not in dataframe.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not found in the dataframe.")

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=feature_name,
        y=dataframe.index,  # Use index or add another feature for better visualization
        hue=cluster_column,
        palette="viridis",
        data=dataframe,
        s=50,  # Marker size
        alpha=0.7,  # Transparency for better overlap visualization
    )

    plt.title(f"Clusters based on {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Index")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
