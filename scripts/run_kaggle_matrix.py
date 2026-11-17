"""Work through the experiment matrix inside one session-limited GPU session.

Free Kaggle and Colab sessions are killed on a timer and without warning, and
the weekly GPU quota is far smaller than the full matrix in
docs/experiment_protocol.md. This script is the bridge between the two: it
launches runs one at a time with --resume, so that

- a run interrupted mid-way restarts at its next epoch rather than from zero,
- a run that already finished is skipped in milliseconds,
- and the whole cell can simply be re-run in a fresh session until the matrix
  is complete.

It deliberately does NOT run the full 72-run matrix. At a measured ~4.6 minutes
per epoch on a T4, 72 runs is roughly 277 GPU-hours, or nine weeks of quota.
RUNS below is the reduced matrix that still answers the project's question.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent

# Ordered by what the project would most regret not having.
#
# The cluster split carries the out-of-domain claim, so it comes first, and
# `full` is paired with `shuffled_coords` seed by seed. If the quota runs out
# part-way we are then left with complete, comparable pairs rather than three
# `full` runs and nothing to compare them against.
#
# biosnap/random seed 1 full is omitted: it is run separately as the first
# timing run. If it is missing, --resume picks it up anyway.
RUNS = [
    ("biosnap", "cluster", "full", 1, False),
    ("biosnap", "cluster", "shuffled_coords", 1, False),
    ("biosnap", "cluster", "full", 2, False),
    ("biosnap", "cluster", "shuffled_coords", 2, False),
    ("biosnap", "cluster", "full", 3, False),
    ("biosnap", "cluster", "shuffled_coords", 3, False),
    ("biosnap", "random", "full", 2, False),
    ("biosnap", "random", "full", 3, False),
    ("biosnap", "random", "shuffled_coords", 1, False),
    ("biosnap", "random", "shuffled_coords", 2, False),
    ("biosnap", "random", "shuffled_coords", 3, False),
]

# Human is a separate, lower-priority block: it is not worth spending quota on
# before the matched-capacity DrugBAN baseline exists, because the project's
# claim is comparative and there is nothing yet to compare against.
#
# Its training set is 3,041 cleaned rows against BioSNAP's 19,189, so an epoch
# costs roughly a fifth as much and a run lands well under an hour.
#
# The shipped and cleaned arms are deliberately interleaved seed by seed. The
# contamination being measured is 12 pairs in a 304-row test set, which is
# small enough to disappear into seed-to-seed spread if the two arms are
# compared as independent means. Paired by seed -- same initialisation, same
# batch order, only the data differing -- the seed noise largely cancels and
# the per-seed differences can at least be read for a consistent sign. Report
# them as three paired differences, not as two means with error bars, and say
# plainly that the comparison is underpowered at this sample size.
HUMAN_RUNS = [
    ("human", "cold", "full", 1, True),
    ("human", "cold", "full", 1, False),
    ("human", "cold", "full", 2, True),
    ("human", "cold", "full", 2, False),
    ("human", "cold", "full", 3, True),
    ("human", "cold", "full", 3, False),
    ("human", "cold", "shuffled_coords", 1, True),
    ("human", "cold", "shuffled_coords", 2, True),
    ("human", "cold", "shuffled_coords", 3, True),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="/kaggle/working/runs",
        help="must outlive the session; /kaggle/working on Kaggle, a mounted "
        "Drive folder on Colab",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=10.5,
        help="stop launching new runs once this much wall clock has passed. "
        "Kaggle kills a session near 12 hours; stopping short of that ends on "
        "a finished run instead of a half-written epoch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="passed through unchanged on every resume, because the cosine "
        "schedule is derived from it",
    )
    parser.add_argument(
        "--include-human",
        action="store_true",
        help="append the human/cold block, shipped against cleaned, paired by "
        "seed. Hold this back until the DrugBAN baseline exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and what is already done, then exit",
    )
    return parser.parse_args()


def results_path(output_root: Path, dataset, split, ablation, seed, clean) -> Path:
    return (
        output_root
        / f"{dataset}_{split}{'_cleaned' if clean else ''}"
        / ablation
        / f"seed_{seed}"
        / "results.json"
    )


def describe(dataset, split, ablation, seed, clean) -> str:
    suffix = " [cleaned]" if clean else ""
    return f"{dataset}/{split} {ablation} seed={seed}{suffix}"


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)

    planned = RUNS + (HUMAN_RUNS if args.include_human else [])
    pending = [run for run in planned if not results_path(output_root, *run).exists()]
    done = len(planned) - len(pending)
    print(f"matrix: {len(planned)} runs, {done} already finished, {len(pending)} to go")

    if args.dry_run:
        for run in planned:
            marker = "done" if results_path(output_root, *run).exists() else "todo"
            print(f"  [{marker}] {describe(*run)}")
        return 0

    started = time.time()
    completed, failed = [], []

    for run in pending:
        elapsed_hours = (time.time() - started) / 3600
        if elapsed_hours > args.budget_hours:
            print(
                f"\nstopping at {elapsed_hours:.1f}h of a {args.budget_hours}h "
                f"budget with {len(pending) - len(completed) - len(failed)} runs "
                f"left. Commit the notebook, then re-run this in a new session.",
                flush=True,
            )
            break

        print(f"\n=== {describe(*run)}  (+{elapsed_hours:.1f}h)", flush=True)
        dataset, split, ablation, seed, clean = run
        command = [
            sys.executable,
            "train_e3dti.py",
            "--dataset", dataset,
            "--split", split,
            "--ablation", ablation,
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--output-root", str(output_root),
            "--resume",
        ]
        if clean:
            command.append("--clean-splits")
        result = subprocess.run(command, cwd=REPOSITORY)

        if result.returncode == 0:
            completed.append(run)
        else:
            # One bad run must not cost the rest of the session. Record it and
            # keep going; the failure is visible in the summary below.
            failed.append(run)
            print(
                f"!!! {describe(*run)} exited {result.returncode}, continuing",
                flush=True,
            )

    print("\n" + "=" * 60)
    print(f"finished this session: {len(completed)}")
    for run in failed:
        print(f"FAILED: {describe(*run)}")

    print("\nresults on disk:")
    for path in sorted(output_root.rglob("results.json")):
        data = json.loads(path.read_text())
        config = data["config"]
        print(
            f"  {config['dataset']}/{config['split']} {config['ablation']} "
            f"seed={config['seed']} best_epoch={data['best_epoch']} "
            f"AUROC={data['test']['auroc']:.4f} AUPRC={data['test']['auprc']:.4f}"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
