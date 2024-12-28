import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report


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
