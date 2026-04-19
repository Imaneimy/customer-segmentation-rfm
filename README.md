# customer-segmentation-rfm
![Tests](https://github.com/Imaneimy/customer-segmentation-rfm/actions/workflows/tests.yml/badge.svg)

Customer segmentation was one of the recurring questions at Orange Maroc — understanding which customers drove the most revenue, which ones were becoming inactive, and where to focus retention efforts. The answers lived in Power BI dashboards, but the underlying logic (recency, frequency, spend) was never formalized. This project does that: a full RFM pipeline from raw transactions to labeled segments, with K-Means clustering on top to let the data decide the groupings rather than using arbitrary score thresholds.

The dataset is 2,800+ transactions from 400 synthetic customers. The rule-based RFM segmentation (Champions, Loyal, At Risk, etc.) follows the standard Marketing framework. The K-Means layer then independently clusters customers in the RFM space — the two approaches are compared in the output.

## Structure

```
src/
  rfm.py              # load transactions, compute RFM, score (quartiles), assign segments
  clustering.py       # StandardScaler, elbow method, silhouette scores, K-Means, cluster profiles
  visualizations.py   # 5 charts: distributions, elbow, silhouette, segment revenue, cluster scatter
  run_segmentation.py # entry point

tests/
  test_rfm.py         # 12 unit tests TC-RFM-001→012
  test_clustering.py  # 8 unit tests TC-CL-001→008

data/
  transactions.csv    # 2,808 transactions, 400 customers
  generate_data.py    # script that generated the data

reports/              # generated charts (git-ignored)
```

## Running it

```bash
pip install -r requirements.txt
cd src
python run_segmentation.py
```

Prints the RFM segment breakdown (customers, avg recency/frequency/monetary, total revenue), the best K chosen by silhouette score, and the K-Means cluster profiles. Saves 5 charts to `reports/`.

```bash
pytest tests/ -v
```

## RFM logic

- **Recency**: days since last purchase (lower = better)
- **Frequency**: total number of transactions
- **Monetary**: total spend

Each dimension is scored 1-4 by quartile. Segments are assigned from R+F score combinations (Champions = R≥4 & F≥4, At Risk = R≤2 & F≥3, etc.).

## Why K-Means on top of rule-based RFM

Rule-based segments are easy to explain to stakeholders but rely on fixed thresholds. K-Means finds natural groupings in the data without assumptions. Running both and comparing them is useful for validating that the manual thresholds match the actual data distribution — which is the kind of cross-check I would have done before presenting segment numbers to management.

## Stack

Python, Pandas, Scikit-learn, Matplotlib, Pytest
