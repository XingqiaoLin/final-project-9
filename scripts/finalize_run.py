"""Write results.json for a run that was stopped before it finished.

train_e3dti.py evaluates on test once, after the epoch loop, so a run killed
part-way leaves a perfectly good best.pt and no reported numbers. That is the
right default -- it keeps test untouched until model selection is over -- but
it means an interrupted run looks like nothing happened.

This recovers it. Model selection already happened: best.pt holds the epoch
with the highest validation AUROC, which is the checkpoint the finished run
would have used anyway. Nothing here peeks at test to make a choice; the
threshold still comes from validation, exactly as in the full path.

    python scripts/finalize_run.py runs/biosnap_random/full/seed_1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_e3dti
from train_e3dti import (
    build_model,
    choose_device,
    make_loaders,
    predict,
    prepare_data,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="directory holding best.pt")
    parser.add_argument("--device", default="auto")
    options = parser.parse_args()

    checkpoint_path = options.run / "best.pt"
    if not checkpoint_path.exists():
        print(f"no best.pt in {options.run}", file=sys.stderr)
        return 1
    if (options.run / "results.json").exists():
        print("results.json already exists, nothing to do")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # A checkpoint written by an older commit will not carry options added
    # since -- --arch, --clean-splits and --drug-chirality all postdate the
    # first runs. Start from today's defaults and lay the saved values on top,
    # so recovering an old run does not crash on an attribute that did not
    # exist when it was saved. Defaults are the off position for every one of
    # those flags, which is what those runs actually used.
    saved = sys.argv
    sys.argv = ["train_e3dti.py"]
    try:
        args = train_e3dti.parse_args()
    finally:
        sys.argv = saved
    for key, value in checkpoint["args"].items():
        setattr(args, key, value)
    args.device = options.device
    device = choose_device(args.device)
    print(f"recovering {options.run} from epoch {checkpoint['epoch']}")

    frames, drug_graphs, protein_graphs, split_audit, cleaning = prepare_data(args)
    loaders = make_loaders(args, frames, drug_graphs, protein_graphs)
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint["model"])

    val_y, val_p, validation = predict(model, loaders["val"], device)
    test_y, test_p, test = predict(model, loaders["test"], device)

    precision, recall, thresholds = precision_recall_curve(val_y, val_p)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    threshold = float(thresholds[np.argmax(f1[:-1])])
    tn, fp, fn, tp = confusion_matrix(
        test_y, test_p >= threshold, labels=[0, 1]
    ).ravel()
    test.update(
        threshold=threshold,
        accuracy=(tp + tn) / len(test_y),
        sensitivity=tp / max(tp + fn, 1),
        specificity=tn / max(tn + fp, 1),
        precision=tp / max(tp + fp, 1),
        f1=2 * tp / max(2 * tp + fp + fn, 1),
    )

    history_path = options.run / "history.csv"
    epochs_run = len(pd.read_csv(history_path)) if history_path.exists() else None

    result = {
        "config": checkpoint["args"],
        "device": str(device),
        "best_epoch": int(checkpoint["epoch"]),
        "epochs_run": epochs_run,
        # Flagged so this can never be mistaken for a completed run when the
        # results are read back or tabulated.
        "finalized_early": True,
        "n_pairs": {name: len(frame) for name, frame in frames.items()},
        "split_audit": split_audit,
        "cleaning": cleaning,
        "validation": validation,
        "test": test,
    }
    with open(options.run / "results.json", "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    print(json.dumps(test, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
