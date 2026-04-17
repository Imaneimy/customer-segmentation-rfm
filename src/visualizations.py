"""
Charts for the RFM segmentation project.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_rfm_distributions(rfm, out="reports/rfm_distributions.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, color in zip(axes, ["recency", "frequency", "monetary"],
                               ["#C44E52", "#4C72B0", "#55A868"]):
        ax.hist(rfm[col], bins=20, color=color, edgecolor="white")
        ax.set_title(col.capitalize(), fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Customers")
    fig.suptitle("RFM Feature Distributions", fontsize=14)
    plt.tight_layout()
    _save(fig, out)


def plot_elbow(inertia_dict, out="reports/elbow.png"):
    k_vals = list(inertia_dict.keys())
    inertias = list(inertia_dict.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_vals, inertias, marker="o", color="#4C72B0")
    ax.set_title("Elbow Method — Inertia vs K", fontsize=13)
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Inertia")
    ax.grid(alpha=0.3)
    _save(fig, out)


def plot_silhouette(sil_dict, out="reports/silhouette.png"):
    k_vals = list(sil_dict.keys())
    scores = list(sil_dict.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(k_vals, scores, color="#55A868")
    ax.set_title("Silhouette Score vs K", fontsize=13)
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Silhouette score")
    ax.set_ylim(0, 1)
    _save(fig, out)


def plot_segments_revenue(summary, out="reports/segment_revenue.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3", "#937860"]
    bars = ax.barh(summary["segment"], summary["total_revenue"],
                   color=colors[:len(summary)])
    ax.set_title("Total Revenue by RFM Segment", fontsize=13, pad=12)
    ax.set_xlabel("Revenue (€)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.invert_yaxis()
    _save(fig, out)


def plot_cluster_scatter(rfm_clustered, out="reports/cluster_scatter.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452", "#8172B3"]
    for cluster in sorted(rfm_clustered["cluster"].unique()):
        sub = rfm_clustered[rfm_clustered["cluster"] == cluster]
        ax.scatter(sub["recency"], sub["monetary"], s=20,
                   alpha=0.6, label=f"Cluster {cluster}",
                   color=colors[cluster % len(colors)])
    ax.set_title("Recency vs Monetary — K-Means Clusters", fontsize=13)
    ax.set_xlabel("Recency (days since last purchase)")
    ax.set_ylabel("Monetary (total spend €)")
    ax.legend()
    _save(fig, out)
