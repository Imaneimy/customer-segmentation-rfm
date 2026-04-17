"""
Entry point: compute RFM, run K-Means, generate charts.
"""

from pathlib import Path
from rfm import load_transactions, compute_rfm, score_rfm, assign_segment, segment_summary
from clustering import scale_rfm, elbow_inertia, silhouette_scores, fit_kmeans, add_cluster_labels, cluster_profiles
from visualizations import (
    plot_rfm_distributions,
    plot_elbow,
    plot_silhouette,
    plot_segments_revenue,
    plot_cluster_scatter,
)

DATA = Path(__file__).parent.parent / "data" / "transactions.csv"
REPORTS = Path(__file__).parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    df = load_transactions(DATA)
    print(f"Transactions: {len(df)} | Customers: {df['customer_id'].nunique()}\n")

    rfm = compute_rfm(df)
    rfm_scored = score_rfm(rfm)
    rfm_full = assign_segment(rfm_scored)

    summary = segment_summary(rfm_full)
    print("--- RFM Segments ---")
    print(summary.to_string(index=False))

    scaled, scaler = scale_rfm(rfm)
    inertia = elbow_inertia(scaled)
    sil = silhouette_scores(scaled)
    best_k = max(sil, key=sil.get)
    print(f"\nBest K (silhouette): {best_k} (score={sil[best_k]})")

    km = fit_kmeans(scaled, n_clusters=best_k)
    rfm_clustered = add_cluster_labels(rfm, km, scaled)

    profiles = cluster_profiles(rfm_clustered)
    print("\n--- K-Means Cluster Profiles ---")
    print(profiles.to_string(index=False))

    plot_rfm_distributions(rfm, str(REPORTS / "rfm_distributions.png"))
    plot_elbow(inertia, str(REPORTS / "elbow.png"))
    plot_silhouette(sil, str(REPORTS / "silhouette.png"))
    plot_segments_revenue(summary, str(REPORTS / "segment_revenue.png"))
    plot_cluster_scatter(rfm_clustered, str(REPORTS / "cluster_scatter.png"))

    print(f"\nCharts saved to {REPORTS}/")


if __name__ == "__main__":
    main()
