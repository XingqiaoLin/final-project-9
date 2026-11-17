#!/usr/bin/env python3
"""Train one reproducible E3DTI run from the command line."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from e3dti.data import (
    PairDataset,
    audit_splits,
    collate_pairs,
    drug_graph,
    load_drug_conformers,
    load_protein_structures,
    protein_graph,
    read_splits,
    stable_id,
)
from e3dti.model import E3DTI


VALID_SPLITS = {
    "bindingdb": ("random", "cluster"),
    "biosnap": ("random", "cluster"),
    "human": ("random", "cold"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=VALID_SPLITS, default="biosnap")
    parser.add_argument("--split", default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--drug-layers", type=int, default=4)
    parser.add_argument("--protein-layers", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--ablation", choices=sorted(E3DTI.ABLATIONS), default="full")
    parser.add_argument("--max-train-pairs", type=int)
    parser.add_argument("--max-val-pairs", type=int)
    parser.add_argument("--max-test-pairs", type=int)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from last.pt in the output directory if one exists, and "
        "skip the run entirely if results.json is already there. Intended for "
        "session-limited platforms such as Kaggle and Colab.",
    )
    return parser.parse_args()


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


CHECKPOINT_NAME = "last.pt"


def capture_rng_state():
    """Batch order and dropout draw from these, so a resumed run must restore
    them or it is not the same run the seed promised."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def save_checkpoint(path, payload):
    """Write via a temporary file: a session killed mid-write must not leave a
    corrupt checkpoint behind, which would cost the whole run."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def stratified_cap(frame, maximum, seed):
    if maximum is None or maximum >= len(frame):
        return frame.reset_index(drop=True)
    if maximum < frame["Y"].nunique():
        raise ValueError("Sample cap must be at least the number of label classes")
    selected, _ = train_test_split(
        frame,
        train_size=maximum,
        random_state=seed,
        stratify=frame["Y"],
    )
    return selected.reset_index(drop=True)


def prepare_data(args):
    root = snapshot_download(
        "Xingqiao-2002/DTI",
        repo_type="dataset",
        allow_patterns=[
            f"main/{args.dataset}/{args.split}/*",
            "main/drug_sdf/drug_conformers.sdf.gz",
            "main/protein_af3/protein_ca.npz",
        ],
    )
    frames = read_splits(root, args.dataset, args.split)
    original_audit = audit_splits(frames)
    caps = {
        "train": args.max_train_pairs,
        "val": args.max_val_pairs,
        "test": args.max_test_pairs,
    }
    frames = {
        name: stratified_cap(frame, caps[name], args.seed + i)
        for i, (name, frame) in enumerate(frames.items())
    }

    conformers = load_drug_conformers(
        Path(root) / "main/drug_sdf/drug_conformers.sdf.gz"
    )
    structures = load_protein_structures(
        Path(root) / "main/protein_af3/protein_ca.npz"
    )
    for name, frame in frames.items():
        frame = frame.copy()
        frame["did"] = frame["SMILES"].map(stable_id)
        frame["pid"] = frame["Protein"].map(stable_id)
        covered = frame["did"].isin(conformers) & frame["pid"].isin(structures)
        if not covered.all():
            print(f"{name}: dropping {(~covered).sum()} pairs without both structures")
        frames[name] = frame[covered].reset_index(drop=True)

    combined = pd.concat(frames.values(), ignore_index=True)
    drug_graphs = {
        drug_id: drug_graph(conformers[drug_id])
        for drug_id in combined["did"].unique()
    }
    # Read the flag off the ablation table rather than restating the list here,
    # so a new ablation cannot silently get the wrong protein edge topology.
    # Spatial k-NN edges are built only when the protein branch keeps geometry;
    # the shuffled_* ablations do keep it, and randomise the coordinates inside
    # the model instead.
    protein_uses_geometry = E3DTI.ABLATIONS[args.ablation][1]
    protein_graphs = {
        protein_id: protein_graph(
            *structures[protein_id], geometry_neighbors=protein_uses_geometry
        )
        for protein_id in combined["pid"].unique()
    }
    return frames, drug_graphs, protein_graphs, original_audit


def make_loaders(args, frames, drug_graphs, protein_graphs):
    loaders = {}
    for name, frame in frames.items():
        shuffle = name == "train"
        dataset = PairDataset(frame, drug_graphs, protein_graphs)
        loaders[name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=shuffle,
            collate_fn=collate_pairs,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            drop_last=shuffle,
        )
    if len(loaders["train"]) == 0:
        raise ValueError("Training set is smaller than one full batch")
    return loaders


def move_graph(graph, device):
    return {key: value.to(device, non_blocking=True) for key, value in graph.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for drug, protein, label in loader:
        logits, _ = model(move_graph(drug, device), move_graph(protein, device))
        labels.append(label.numpy())
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    y = np.concatenate(labels)
    probability = np.concatenate(probabilities)
    loss = F.binary_cross_entropy(
        torch.tensor(probability).clamp(1e-7, 1 - 1e-7), torch.tensor(y)
    ).item()
    return y, probability, {
        "auroc": roc_auc_score(y, probability),
        "auprc": average_precision_score(y, probability),
        "loss": loss,
    }


def main():
    args = parse_args()
    if args.split not in VALID_SPLITS[args.dataset]:
        raise ValueError(f"{args.dataset} supports {VALID_SPLITS[args.dataset]}, not {args.split!r}")
    seed_everything(args.seed)
    device = choose_device(args.device)
    output = (
        Path(args.output_root)
        / f"{args.dataset}_{args.split}"
        / args.ablation
        / f"seed_{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    print(f"device={device} output={output}")

    if args.resume and (output / "results.json").exists():
        print(f"{output / 'results.json'} already exists; nothing to do")
        return

    started = time.time()
    frames, drug_graphs, protein_graphs, split_audit = prepare_data(args)
    print(
        "pairs=",
        {name: len(frame) for name, frame in frames.items()},
        "split_audit=",
        split_audit,
    )
    loaders = make_loaders(args, frames, drug_graphs, protein_graphs)
    model = E3DTI(
        dim=args.dim,
        drug_layers=args.drug_layers,
        protein_layers=args.protein_layers,
        ablation=args.ablation,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(loaders["train"])
    warmup_steps = max(1, args.warmup_epochs * len(loaders["train"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1, (step + 1) / warmup_steps)
        * 0.5
        * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)),
    )

    history = []
    best_auroc = -math.inf
    best_state = None
    bad_epochs = 0
    start_epoch = 1

    checkpoint_path = output / CHECKPOINT_NAME
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_auroc = checkpoint["best_auroc"]
        best_state = checkpoint["best_state"]
        bad_epochs = checkpoint["bad_epochs"]
        start_epoch = checkpoint["epoch"] + 1
        restore_rng_state(checkpoint["rng"])
        print(
            f"resumed at epoch {start_epoch} of {args.epochs} "
            f"(best val_auroc={best_auroc:.4f}, {bad_epochs} epochs without "
            f"improvement)"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for drug, protein, label in loaders["train"]:
            logits, _ = model(move_graph(drug, device), move_graph(protein, device))
            loss = F.binary_cross_entropy_with_logits(logits, label.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        _, _, validation = predict(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(loaders["train"]),
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        improved = validation["auroc"] > best_auroc
        if improved:
            best_auroc = validation["auroc"]
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            torch.save(
                {"model": best_state, "args": vars(args), "epoch": epoch},
                output / "best.pt",
            )
        else:
            bad_epochs += 1
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_auroc={validation['auroc']:.4f} val_auprc={validation['auprc']:.4f}"
            + (" *" if improved else "")
        )
        save_checkpoint(
            checkpoint_path,
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best_auroc": best_auroc,
                "best_state": best_state,
                "bad_epochs": bad_epochs,
                "rng": capture_rng_state(),
                "args": vars(args),
            },
        )
        # Written every epoch, not once at the end. A run killed by a session
        # timeout used to leave no readable progress file at all: the history
        # survived only inside last.pt, which is fine for resuming but useless
        # for watching a long run. Fifty rows of CSV costs nothing.
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        if bad_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")
    model.load_state_dict(best_state)
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
    result = {
        "config": vars(args),
        "device": str(device),
        "best_epoch": int(np.argmax([row["val_auroc"] for row in history]) + 1),
        "n_pairs": {name: len(frame) for name, frame in frames.items()},
        "split_audit": split_audit,
        "validation": validation,
        "test": test,
        "elapsed_seconds": time.time() - started,
    }
    with open(output / "results.json", "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
