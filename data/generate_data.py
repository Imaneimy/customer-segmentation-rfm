"""
Generate synthetic transaction data for RFM segmentation.
Run once: python3 generate_data.py
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

SNAPSHOT = date(2024, 3, 31)

SEGMENTS = {
    "champions":     {"n": 60,  "freq": (10, 30), "days_since": (1, 30),   "spend": (100, 500)},
    "loyal":         {"n": 80,  "freq": (5, 15),  "days_since": (15, 60),  "spend": (60, 250)},
    "at_risk":       {"n": 100, "freq": (2, 6),   "days_since": (60, 150), "spend": (30, 150)},
    "lost":          {"n": 80,  "freq": (1, 3),   "days_since": (150, 365),"spend": (10, 80)},
    "new":           {"n": 80,  "freq": (1, 2),   "days_since": (1, 20),   "spend": (20, 120)},
}

rows = []
cid = 1
for segment, cfg in SEGMENTS.items():
    for _ in range(cfg["n"]):
        customer_id = f"C{cid:04d}"
        freq = random.randint(*cfg["freq"])
        last_days = random.randint(*cfg["days_since"])
        last_purchase = SNAPSHOT - timedelta(days=last_days)
        for t in range(freq):
            days_before = random.randint(0, 365)
            tx_date = last_purchase - timedelta(days=days_before * t // max(freq, 1))
            amount = round(random.uniform(*cfg["spend"]), 2)
            rows.append({
                "customer_id": customer_id,
                "transaction_date": tx_date.isoformat(),
                "amount": amount,
                "segment_label": segment,
            })
        cid += 1

random.shuffle(rows)

out = Path(__file__).parent / "transactions.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["customer_id", "transaction_date", "amount", "segment_label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} transactions for {cid-1} customers to {out}")
