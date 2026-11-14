# E3DTI reproduction

This branch turns the original teaching notebook into a testable experiment
pipeline while keeping `egnn_dti.ipynb` readable.

## Setup on macOS

```bash
uv venv --python 3.11 ~/06763/FinalProject/.venv-e3dti
source ~/06763/FinalProject/.venv-e3dti/bin/activate
uv pip install -e .
```

The notebook defaults to `workers=0` because its dataset classes live in the
Jupyter kernel. The command-line runner also defaults to zero for portability.

## 1–5: correctness before training

Run the batch-independence and E(3)-invariance tests:

```bash
python -m unittest discover -s tests -v
```

Audit exact `(SMILES, Protein)` overlap without downloading structures:

```bash
python scripts/audit_splits.py --dataset biosnap --split random
```

The BAN implementation passes drug and protein padding masks explicitly. It
zeros both invalid attention entries and projected padding features, preventing
FCNet biases from making a prediction depend on the largest graph in a batch.

## 6: one smoke run

This checks the complete pipeline but is not a scientific result:

```bash
python train_e3dti.py \
  --dataset biosnap --split random --seed 42 \
  --epochs 1 --batch 4 --dim 32 \
  --drug-layers 1 --protein-layers 1 \
  --max-train-pairs 32 --max-val-pairs 16 --max-test-pairs 16 \
  --device cpu --output-root runs-smoke
```

## 7: multiple seeds

After the smoke run succeeds, run at least three seeds with the full model:

```bash
python scripts/run_experiment_matrix.py \
  --dataset biosnap --split random --seeds 1 2 3
```

Use five seeds for the final report when compute allows.

## 8: 3D ablations

Four controlled variants keep the message-passing architecture but disable
distance and coordinate updates on selected encoders. When protein 3D is off,
the protein graph also uses sequence-nearest rather than spatial-nearest edges:

- `full`
- `no_drug_3d`
- `no_protein_3d`
- `no_3d`

Run all four:

```bash
python scripts/run_experiment_matrix.py \
  --dataset biosnap --split random --seeds 1 2 3 \
  --all-ablations
```

`no_3d` is an internal geometry ablation, not the official DrugBAN baseline.
The official baseline uses a DGL/DGLLife molecular GCN plus a protein CNN and
must be run from the DrugBAN implementation with the same CSV splits, seeds,
early-stopping criterion, and metrics. Do not label the internal ablation as
DrugBAN in a report. See `docs/experiment_protocol.md` for the fair-comparison
checklist and the cluster-split validation caveat.

## 9: all supported benchmark splits

The six combinations are BindingDB random/cluster, BioSNAP random/cluster, and
Human random/cold.

```bash
python scripts/run_experiment_matrix.py \
  --all-datasets --seeds 1 2 3
```

All datasets plus all ablations is expensive:

```bash
python scripts/run_experiment_matrix.py \
  --all-datasets --all-ablations --seeds 1 2 3
```

Summarize completed runs as count, mean, and sample standard deviation:

```bash
python scripts/summarize_results.py
```

The summary is written to `runs/summary.csv`.

## Output layout

```text
runs/<dataset>_<split>/<ablation>/seed_<seed>/
├── best.pt
├── history.csv
└── results.json
```

Only validation AUROC selects the checkpoint. Test labels are used once after
model selection; the classification threshold is selected on validation data.
