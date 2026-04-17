import sys
from pathlib import Path
import pytest
import pandas as pd
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rfm import load_transactions, compute_rfm, score_rfm, assign_segment, segment_summary

DATA = Path(__file__).parent.parent / "data" / "transactions.csv"
SNAPSHOT = date(2024, 3, 31)


@pytest.fixture(scope="module")
def df():
    return load_transactions(DATA)


@pytest.fixture(scope="module")
def rfm(df):
    return compute_rfm(df, snapshot_date=SNAPSHOT)


@pytest.fixture(scope="module")
def rfm_full(rfm):
    scored = score_rfm(rfm)
    return assign_segment(scored)


# TC-RFM-001
def test_load_returns_dataframe(df):
    assert isinstance(df, pd.DataFrame)


# TC-RFM-002
def test_no_negative_amounts(df):
    assert (df["amount"] > 0).all()


# TC-RFM-003
def test_rfm_has_three_metrics(rfm):
    for col in ["recency", "frequency", "monetary"]:
        assert col in rfm.columns


# TC-RFM-004
def test_recency_non_negative(rfm):
    assert (rfm["recency"] >= 0).all()


# TC-RFM-005
def test_frequency_positive(rfm):
    assert (rfm["frequency"] > 0).all()


# TC-RFM-006
def test_monetary_positive(rfm):
    assert (rfm["monetary"] > 0).all()


# TC-RFM-007
def test_one_row_per_customer(df, rfm):
    assert len(rfm) == df["customer_id"].nunique()


# TC-RFM-008
def test_score_rfm_adds_score_col(rfm):
    scored = score_rfm(rfm)
    assert "rfm_score" in scored.columns


# TC-RFM-009
def test_rfm_score_range(rfm):
    scored = score_rfm(rfm)
    assert scored["rfm_score"].between(3, 12).all()


# TC-RFM-010
def test_segment_assigned(rfm_full):
    assert "segment" in rfm_full.columns
    assert rfm_full["segment"].isna().sum() == 0


# TC-RFM-011
def test_segment_summary_has_all_segments(rfm_full):
    summary = segment_summary(rfm_full)
    assert "segment" in summary.columns
    assert len(summary) > 0


# TC-RFM-012
def test_summary_total_revenue_positive(rfm_full):
    summary = segment_summary(rfm_full)
    assert (summary["total_revenue"] > 0).all()
