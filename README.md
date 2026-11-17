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

## 6b: long runs on Kaggle or Colab

Free GPU sessions are time-limited and are killed without warning, so any run
longer than a session needs `--resume`:

```bash
python train_e3dti.py \
  --dataset biosnap --split random --seed 1 \
  --output-root /kaggle/working/runs \
  --resume
```

`--resume` writes `last.pt` after every epoch (atomically, so a session killed
mid-write cannot corrupt it) and restores the model, optimizer, scheduler,
history, early-stopping counters and the Python/NumPy/Torch RNG state. Re-run
the identical command after a session dies and training continues from the next
epoch. If `results.json` is already in the output directory the run is skipped,
so the same matrix command can be re-issued until everything finishes.

Pass the same `--epochs` on every resume: the cosine schedule is derived from
it, and changing it mid-run changes the learning rate for the remaining epochs.

`--output-root` must point somewhere that outlives the session
(`/kaggle/working` on Kaggle, a mounted Drive folder on Colab). The default
`runs/` lives in the container and disappears with it.

To work through a whole matrix unattended inside session limits:

```bash
python scripts/run_kaggle_matrix.py \
  --output-root /kaggle/working/runs --budget-hours 10.5
```

It launches one run at a time with `--resume`, skips runs that already have a
`results.json`, keeps going past a failed run rather than losing the session to
it, and stops launching new runs before the platform's kill timer so the
session ends on a finished run. Re-run the same command in a new session until
the matrix is complete. `--dry-run` prints the plan and what is already done.

The matrix it runs is the reduced one, not the 72-run matrix in
`docs/experiment_protocol.md`. At a measured ~4.6 minutes per epoch on a T4,
72 runs is roughly 277 GPU-hours — about nine weeks of a 30-hour weekly quota.

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

### The capacity-matched control

The four variants above remove an input pathway, so a drop in score can be
argued away as the model having less to work with. Three further variants
answer the question without that confound by keeping architecture, parameter
count and every code path identical to `full`, and randomly reassigning the
coordinates among each graph's nodes on every forward pass:

- `shuffled_coords`
- `shuffled_drug_coords`
- `shuffled_protein_coords`

A model that scores the same as `full` under `shuffled_coords` was not using
the geometry. Run the comparison with:

```bash
python scripts/run_experiment_matrix.py \
  --dataset biosnap --split cluster --seeds 1 2 3 \
  --shuffle-control
```

See `docs/experiment_protocol.md` for the two caveats worth stating when
reporting these: the protein contact topology survives the shuffle, and the
shuffle arms carry extra within-run noise.

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
├── best.pt          # best validation AUROC so far
├── last.pt          # resume point, written every epoch (--resume only)
├── history.csv
└── results.json     # written once the run finishes; its presence means done
```

Only validation AUROC selects the checkpoint. Test labels are used once after
model selection; the classification threshold is selected on validation data.
