# Data integrity audit

Audit of the six shipped DrugBAN splits, run before any full training.
Date: 17 November 2026.

All numbers below come from `e3dti.data.audit_splits` and `read_splits` over the
CSVs in `Xingqiao-2002/DTI`, on the files as shipped.

## 1. Fixed: the cluster validation carve-out tracked the training seed

`read_splits` passed the training seed as the `random_state` of the 10%
validation carve-out from `source_train.csv`. Cluster splits ship no validation
file, so each seed produced a different validation set:

| Split | Validation pairs common to seeds 1, 2 and 3 | After fix |
|---|---|---|
| biosnap/cluster | 9 of 977 — **0.9%** | 977 — 100% |
| bindingdb/cluster | 10 of 1,492 — **0.7%** | 1,492 — 100% |
| biosnap/random | 2,740 of 2,740 — 100% | 100% (unchanged) |

This contradicted `docs/experiment_protocol.md`:

> Use the same fixed CSV split for every seed. Seeds change initialization,
> dropout, and batch order, not train/validation/test membership.

**Why it mattered.** On random splits, seed spread measures model variance. On
cluster splits it measured model variance *plus* split-resampling variance, and
the two were reported in the same table. It also changed which checkpoint was
selected and which decision threshold was chosen, on every seed. Cluster splits
carry the project's out-of-domain claim, where the 2D control already sits in a
wide, near-chance band (0.554–0.697 AUROC), so an unknown part of that spread
was the validation set moving rather than the model.

**Fix.** `VALIDATION_SPLIT_SEED = 0` in `e3dti/data.py`; `read_splits` no longer
takes a seed. Guarded by `tests/test_split_stability.py`.

**Consequence.** Any cluster-split run produced before this commit is not
comparable with runs after it and should be discarded.

## 2. The held-out splits do hold out — confirmed

The proposal claims cluster and cold-pair test rows use drugs and proteins never
seen in training. That claim holds exactly:

| Split | Test rows | Test drug in train | Test protein in train | Both |
|---|---|---|---|---|
| bindingdb/cluster | 1,779 | 0.0% | 0.0% | 0.0% |
| biosnap/cluster | 901 | 0.0% | 0.0% | 0.0% |
| human/cold | 311 | 0.0% | 0.0% | 0.0% |

## 3. How much of each random split is recognition

The same measurement on the in-domain splits quantifies the project's premise —
that a model can score well on standard benchmarks by recognising a protein it
has already seen:

| Split | Test rows | Test drug in train | Test protein in train | **Both** |
|---|---|---|---|---|
| biosnap/random | 5,480 | 97.6% | **99.8%** | **97.4%** |
| bindingdb/random | 9,836 | 89.0% | 97.8% | 86.9% |
| human/random | 1,191 | 64.0% | 80.9% | 49.5% |

On BioSNAP random, 97.4% of test pairs are built from a drug *and* a protein the
model saw during training.

## 4. Open: duplication and leakage in the shipped files

Exact `(SMILES, Protein)` overlap between frames, as shipped:

| Split | Train | Val | Test | Dup in train | Train ∩ val | Train ∩ test | Val ∩ test |
|---|---|---|---|---|---|---|---|
| bindingdb/random | 34,423 | 4,920 | 9,836 | 0 | 0 | 0 | 0 |
| bindingdb/cluster | 13,426 | 1,492 | 1,779 | 0 | 0 | 0 | 0 |
| biosnap/random | 19,189 | 2,740 | 5,480 | 3 | 1 | 3 | 0 |
| biosnap/cluster | 8,787 | 977 | 901 | 1 | 0 | 0 | 0 |
| human/random | 4,176 | 598 | 1,191 | 0 | 0 | 0 | 0 |
| **human/cold** | 3,435 | 155 | 311 | **394** | 0 | 0 | **12** |

`human/cold` is the problem. 394 duplicated training rows is 11.5% of that
split's training set, and 12 pairs appear in **both validation and test** — 3.9%
of a 311-row test set. Validation selects the checkpoint *and* the decision
threshold, so those 12 pairs feed directly into the reported test metrics. The
check-in names human/cold as one of the two results carrying the out-of-domain
claim.

Note the proposal states Human's 731 duplicate pairs were removed, leaving 5,997
rows. Whatever cleaning produced that number did not reach the cold split.

`biosnap/random`'s leakage is real but small: 3 leaked pairs in 5,480 test rows
(0.05%), below the resolution of the comparison.

### Decision needed

The pipeline reports this audit and changes nothing, by design —
`docs/experiment_protocol.md` keeps the shipped rows so results stay comparable
with published DrugBAN numbers. That is defensible for train/test leakage, which
DrugBAN also carried. It is harder to defend for human/cold's validation-test
overlap, which corrupts our own model selection rather than the comparison.

Options for human/cold, for the team to choose:

1. Train on the shipped split, report the contamination as a stated caveat.
2. Deduplicate and remove the 12 val∩test pairs, run both DrugBAN and E3DTI on
   the cleaned split, and report it as non-comparable with published numbers.
3. Run both and report the difference. On a 3,901-row dataset the extra cost is
   small, and the gap is itself a result.

Option 3 is recommended: it preserves comparability and measures how much the
contamination was worth.

## Reproducing

```bash
python scripts/audit_splits.py --dataset human --split cold
python -m unittest discover -s tests -v
```
