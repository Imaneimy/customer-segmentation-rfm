"""
K-Means clustering on RFM features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def scale_rfm(rfm: pd.DataFrame) -> tuple:
    features = rfm[["recency", "frequency", "monetary"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def elbow_inertia(scaled: np.ndarray, k_range: range = range(2, 9)) -> dict:
    return {k: KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled).inertia_ for k in k_range}


def silhouette_scores(scaled: np.ndarray, k_range: range = range(2, 9)) -> dict:
    return {k: round(silhouette_score(scaled, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)), 4)
            for k in k_range}


def fit_kmeans(scaled: np.ndarray, n_clusters: int = 4) -> KMeans:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(scaled)
    return km


def add_cluster_labels(rfm: pd.DataFrame, km: KMeans, scaled: np.ndarray) -> pd.DataFrame:
    df = rfm.copy()
    df["cluster"] = km.predict(scaled)
    return df


def cluster_profiles(rfm_clustered: pd.DataFrame) -> pd.DataFrame:
    return (
        rfm_clustered.groupby("cluster")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .round(1)
        .reset_index()
        .sort_values("avg_monetary", ascending=False)
    )
