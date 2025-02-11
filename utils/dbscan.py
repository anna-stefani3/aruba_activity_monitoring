import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull, QhullError


def test_dbscan_clustering(data: pd.DataFrame, predictions: pd.Series):
    # Apply DBSCAN
    dbscan = DBSCAN()
    clusters = dbscan.fit_predict(data)

    # Identify the largest cluster and label it as 0 (Normal)
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
    new_labels = np.where(clusters == largest_cluster, 0, 1)

    # Metrics comparison
    metrics_report = classification_report(predictions, new_labels)

    # print the metrics
    print(metrics_report)

    return new_labels


def scatter_plot_dbscan(data, labels):
    # Set up the plot
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=data.iloc[:, 0], y=data.iloc[:, 1], fill=True, cmap="Blues", alpha=0.5, levels=30)

    # Unique cluster labels
    cluster_labels = np.unique(labels)

    # Assign colors
    colors = ["blue", "red", "green", "purple", "orange"]  # More colors for extra clusters

    # Plot each cluster separately
    for idx, cluster_label in enumerate(cluster_labels):
        cluster_points = data[labels == cluster_label]

        # Scatter plot
        plt.scatter(
            cluster_points.iloc[:, 0],
            cluster_points.iloc[:, 1],
            label=f"Cluster {cluster_label}",
            alpha=0.7,
            edgecolors="black",
            s=50 if cluster_label == 0 else 80,
            color=colors[idx % len(colors)],  # Avoid index errors if clusters > colors
        )

        # Draw convex hull if the cluster has enough points and is not collinear
        if len(cluster_points) >= 3:
            try:
                hull = ConvexHull(cluster_points.values)  # Convert to NumPy array
                for simplex in hull.simplices:
                    plt.plot(
                        cluster_points.values[simplex, 0],
                        cluster_points.values[simplex, 1],
                        color=colors[idx % len(colors)],
                        linewidth=2,
                    )
            except QhullError:
                print(f"Skipping convex hull for cluster {cluster_label} (collinear points).")

    # Formatting
    plt.xlabel("Sleep Duration")
    plt.ylabel("Sleep Disturbances")
    plt.title("Cool DBSCAN Clustering with Density & Circular Separation")
    plt.legend()
    plt.show()
