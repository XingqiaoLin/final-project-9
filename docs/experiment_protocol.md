# Experiment protocol

## Correctness gates

Do not start full training unless both model tests pass and the split audit has
been recorded with the run. The audit describes the supplied benchmark; it does
not silently remove pairs, because doing so would change comparability.

## Seeds and reporting

Use the same fixed CSV split for every seed. Seeds change initialization,
dropout, and batch order, not train/validation/test membership. Report test
AUROC and AUPRC as mean +/- sample standard deviation, plus the number of seeds.

## Geometry ablations

All E3DTI ablations keep the BAN and decoder fixed. These are controlled E3DTI
ablations, not DrugBAN. There are two families, and they answer different
questions.

**Switch-off family** — `no_drug_3d`, `no_protein_3d`, `no_3d`. Disabling drug
geometry removes distance features and coordinate updates from the drug EGNN.
Disabling protein geometry additionally replaces spatial-nearest protein edges
with sequence-nearest edges.

These remove an input pathway: the RBF features go to zero and the
coordinate-update MLP stops contributing. A drop in score is therefore open to
the objection that the model simply had less to work with, which is the
confound raised in the check-in feedback.

**Shuffle family** — `shuffled_coords`, `shuffled_drug_coords`,
`shuffled_protein_coords`. This is the capacity-matched control. Architecture,
parameter count and every code path are identical to `full`; a fresh random
permutation reassigns the coordinates among the nodes of each graph on every
forward pass, so each molecule keeps its own coordinate set and only the
question of which atom sits where is destroyed. A model that scores the same as
`full` under this ablation was not using the geometry.

Prefer the shuffle family for any claim of the form "3D structure helps".
Report the switch-off family alongside it, not instead of it.

Two caveats to state when reporting the shuffle family:

- It randomises *metric* geometry only. The protein graph's spatial k-NN edges
  are built once from the true coordinates and are not rebuilt, so contact
  topology survives the shuffle. The protein result therefore measures whether
  continuous geometry helps *beyond* the contact graph. The drug graph is fully
  connected, so it carries no such topology and the drug result is clean.
- Because the permutation is redrawn each forward pass, evaluation of these
  ablations is stochastic. It is seeded and is restored with the rest of the
  RNG state on resume, so a run is reproducible, but the shuffle arms carry an
  extra source of within-run noise that `full` does not.

## Official DrugBAN baseline

Use the official repository in a separate legacy environment because its
documented stack uses Python 3.8, PyTorch 1.7.1, DGL, and DGLLife. Do not mix its
dependencies into the E3DTI Python 3.11 environment.

Official in-domain command:

```bash
python main.py --cfg configs/DrugBAN.yaml --data biosnap --split random
```

Before comparing results, make these protocol items identical:

1. CSV rows and split membership.
2. Seeds.
3. Early-stopping metric and patience.
4. Validation-selected decision threshold.
5. AUROC/AUPRC implementations.
6. Whether exact duplicate pairs are retained.

For cluster experiments, do not use `target_test.csv` for both validation and
testing. Create the same stratified 10% source validation split used by E3DTI,
or clearly label the protocols as non-comparable. The official DrugBAN runner's
non-adaptation cross-domain path uses `test_target_dataset` for both validation
and test, so its unmodified output is not a fair comparison to this runner.

Record official DrugBAN results separately as:

```text
runs-drugban/<dataset>_<split>/seed_<seed>/results.json
```

Do not merge DrugBAN and E3DTI tables until the protocol checklist above is
satisfied.
