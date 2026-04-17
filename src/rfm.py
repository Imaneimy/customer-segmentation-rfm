"""
Compute RFM (Recency, Frequency, Monetary) scores from transaction data.
"""

import pandas as pd
from datetime import date


def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["transaction_date"])
    df = df.dropna(subset=["customer_id", "transaction_date", "amount"])
    df = df[df["amount"] > 0]
    return df


def compute_rfm(df: pd.DataFrame, snapshot_date: date = None) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = df["transaction_date"].max().date() + pd.Timedelta(days=1)

    snapshot = pd.Timestamp(snapshot_date)

    rfm = (
        df.groupby("customer_id")
        .agg(
            recency=("transaction_date", lambda x: (snapshot - x.max()).days),
            frequency=("transaction_date", "count"),
            monetary=("amount", "sum"),
        )
        .reset_index()
    )
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm


def score_rfm(rfm: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    df = rfm.copy()
    # Recency: lower is better → reverse labels
    df["r_score"] = pd.qcut(df["recency"], q=n_bins, labels=range(n_bins, 0, -1), duplicates="drop")
    df["f_score"] = pd.qcut(df["frequency"], q=n_bins, labels=range(1, n_bins + 1), duplicates="drop")
    df["m_score"] = pd.qcut(df["monetary"], q=n_bins, labels=range(1, n_bins + 1), duplicates="drop")
    df["rfm_score"] = (
        df["r_score"].astype(int) + df["f_score"].astype(int) + df["m_score"].astype(int)
    )
    return df


def assign_segment(rfm_scored: pd.DataFrame) -> pd.DataFrame:
    df = rfm_scored.copy()

    def segment(row):
        r, f = int(row["r_score"]), int(row["f_score"])
        if r >= 4 and f >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 3 and f <= 2:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r == 1 and f == 1:
            return "Lost"
        else:
            return "Needs Attention"

    df["segment"] = df.apply(segment, axis=1)
    return df


def segment_summary(rfm_full: pd.DataFrame) -> pd.DataFrame:
    return (
        rfm_full.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .round(1)
        .sort_values("total_revenue", ascending=False)
        .reset_index()
    )
