import sys
from pathlib import Path
import pytest
import numpy as np
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rfm import load_transactions, compute_rfm
from clustering import scale_rfm, elbow_inertia, silhouette_scores, fit_kmeans, add_cluster_labels, cluster_profiles

DATA = Path(__file__).parent.parent / "data" / "transactions.csv"
SNAPSHOT = date(2024, 3, 31)


@pytest.fixture(scope="module")
def rfm():
    df = load_transactions(DATA)
    return compute_rfm(df, snapshot_date=SNAPSHOT)


@pytest.fixture(scope="module")
def scaled_data(rfm):
    scaled, scaler = scale_rfm(rfm)
    return scaled, scaler


# TC-CL-001
def test_scale_output_shape(rfm, scaled_data):
    scaled, _ = scaled_data
    assert scaled.shape == (len(rfm), 3)


# TC-CL-002
def test_scaled_mean_near_zero(scaled_data):
    scaled, _ = scaled_data
    assert abs(scaled.mean()) < 0.1


# TC-CL-003
def test_elbow_returns_dict(scaled_data):
    scaled, _ = scaled_data
    result = elbow_inertia(scaled, k_range=range(2, 5))
    assert isinstance(result, dict)
    assert len(result) == 3


# TC-CL-004
def test_inertia_decreases_with_k(scaled_data):
    scaled, _ = scaled_data
    inertia = elbow_inertia(scaled, k_range=range(2, 6))
    values = list(inertia.values())
    assert all(values[i] >= values[i+1] for i in range(len(values)-1))


# TC-CL-005
def test_silhouette_scores_in_range(scaled_data):
    scaled, _ = scaled_data
    sil = silhouette_scores(scaled, k_range=range(2, 5))
    for score in sil.values():
        assert -1 <= score <= 1


# TC-CL-006
def test_kmeans_cluster_count(rfm, scaled_data):
    scaled, _ = scaled_data
    km = fit_kmeans(scaled, n_clusters=4)
    rfm_c = add_cluster_labels(rfm, km, scaled)
    assert rfm_c["cluster"].nunique() == 4


# TC-CL-007
def test_cluster_labels_no_nulls(rfm, scaled_data):
    scaled, _ = scaled_data
    km = fit_kmeans(scaled, n_clusters=4)
    rfm_c = add_cluster_labels(rfm, km, scaled)
    assert rfm_c["cluster"].isna().sum() == 0


# TC-CL-008
def test_cluster_profiles_row_count(rfm, scaled_data):
    scaled, _ = scaled_data
    km = fit_kmeans(scaled, n_clusters=4)
    rfm_c = add_cluster_labels(rfm, km, scaled)
    profiles = cluster_profiles(rfm_c)
    assert len(profiles) == 4
