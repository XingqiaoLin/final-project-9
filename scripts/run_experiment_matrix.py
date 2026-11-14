#!/usr/bin/env python3
"""Run seeds, 3D ablations, and dataset/split combinations consistently."""

import argparse
import subprocess
import sys


DATASETS = {
    "bindingdb": ("random", "cluster"),
    "biosnap": ("random", "cluster"),
    "human": ("random", "cold"),
}
ABLATIONS = ("full", "no_drug_3d", "no_protein_3d", "no_3d")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--all-ablations", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS, default="biosnap")
    parser.add_argument("--split", default="random")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    combinations = (
        [(dataset, split) for dataset, splits in DATASETS.items() for split in splits]
        if args.all_datasets
        else [(args.dataset, args.split)]
    )
    ablations = ABLATIONS if args.all_ablations else ("full",)
    for dataset, split in combinations:
        for ablation in ablations:
            for seed in args.seeds:
                command = [
                    sys.executable,
                    "train_e3dti.py",
                    "--dataset",
                    dataset,
                    "--split",
                    split,
                    "--ablation",
                    ablation,
                    "--seed",
                    str(seed),
                    *args.extra,
                ]
                print("RUN", " ".join(command), flush=True)
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
