#!/usr/bin/env python3
"""Aggregate results.json files as mean +/- sample standard deviation."""

import json
from pathlib import Path

import pandas as pd


rows = []
for path in Path("runs").glob("*/*/seed_*/results.json"):
    result = json.loads(path.read_text())
    config = result["config"]
    rows.append(
        {
            "dataset": config["dataset"],
            "split": config["split"],
            "ablation": config["ablation"],
            "seed": config["seed"],
            "auroc": result["test"]["auroc"],
            "auprc": result["test"]["auprc"],
        }
    )

if not rows:
    raise SystemExit("No runs/*/*/seed_*/results.json files found")

frame = pd.DataFrame(rows)
summary = (
    frame.groupby(["dataset", "split", "ablation"])[["auroc", "auprc"]]
    .agg(["count", "mean", "std"])
    .reset_index()
)
summary.to_csv("runs/summary.csv", index=False)
print(summary.to_string(index=False))
