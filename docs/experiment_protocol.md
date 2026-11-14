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

All E3DTI ablations keep the BAN and decoder fixed. Disabling drug geometry
removes distance features and coordinate updates from the drug EGNN. Disabling
protein geometry additionally replaces spatial-nearest protein edges with
sequence-nearest edges. These are controlled E3DTI ablations, not DrugBAN.

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
