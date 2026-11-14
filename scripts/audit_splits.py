#!/usr/bin/env python3
"""Download pair CSVs only and report duplicate/leaking DTI pairs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import snapshot_download

from e3dti.data import audit_splits, read_splits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bindingdb", "biosnap", "human"], default="biosnap")
    parser.add_argument("--split", default="random")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    root = snapshot_download(
        "Xingqiao-2002/DTI",
        repo_type="dataset",
        allow_patterns=[f"main/{args.dataset}/{args.split}/*"],
    )
    frames = read_splits(root, args.dataset, args.split, args.seed)
    report = audit_splits(frames)
    report.update(
        dataset=args.dataset,
        split=args.split,
        sizes={name: len(frame) for name, frame in frames.items()},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
