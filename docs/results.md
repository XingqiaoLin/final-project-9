# Results

Recorded here because `runs*/` is gitignored and the Kaggle sessions that
produced these numbers were wiped twice. This file is the durable copy.

Date: 17 November 2026. Branch `yogesh/drug-egnn`, commit `1c603e7`.

## Three seeds, with error bars

Run on Colab, results written to Drive. BioSNAP **cluster** split, 10 epochs
per arm, seeds 1/2/3. This supersedes the single-seed section below for every
claim about whether a difference is real.

| Arm | test AUROC | test AUPRC |
|---|---|---|
| DrugBAN matched (2D) | **0.6775 ± 0.0064** | 0.6985 ± 0.0014 |
| E3DTI full (3D) | **0.7235 ± 0.0093** | 0.7339 ± 0.0125 |
| E3DTI, coordinates shuffled | **0.7107 ± 0.0138** | 0.7218 ± 0.0172 |

Mean ± sample standard deviation over three seeds. Per seed:

```
drugban_matched   0.6801  0.6822  0.6702
full              0.7127  0.7290  0.7287
shuffled_coords   0.7169  0.7204  0.6949
```

**The 3D-aware model beats the capacity-matched 2D baseline by +0.046 AUROC**,
against seed spreads of 0.006 and 0.009. The gap is roughly five times the
noise, so it is real.

**Full versus shuffled is +0.0127, and that is inside the noise** — the two
standard deviations are 0.009 and 0.014, and the arms swap places across
seeds: shuffled wins seed 1, full wins seeds 2 and 3.

The single-seed run had shuffled *ahead* of full by 0.018, which looked like
evidence that the geometry was actively unhelpful. Three seeds show that was
noise and the ordering is not stable. The supportable conclusion is neither
"shuffling helps" nor "geometry helps":

> No detectable difference between real and shuffled coordinates. The
> 3D-aware architecture beats the 2D baseline out-of-domain; the predicted
> coordinates cannot be shown to contribute.

That is the same conclusion the single-seed run pointed at, now for a
defensible reason rather than an accident of one random initialisation.

## The headline comparison

BioSNAP **cluster** split — test rows use drugs **and** proteins that never
appear in training (verified: 0.0% overlap on both). One seed, 10 epochs for
every arm, so capacity *and* training budget are matched.

| Arm | best epoch | val AUROC | val AUPRC | **test AUROC** | **test AUPRC** |
|---|---|---|---|---|---|
| DrugBAN matched (2D) | 8 | 0.8866 | 0.9003 | **0.6590** | 0.6773 |
| E3DTI full (3D) | 9 | 0.8491 | — | **0.7085** | 0.7259 |
| E3DTI, coordinates shuffled | 9 | 0.8494 | — | **0.7267** | 0.7432 |

Two findings, pointing in different directions:

1. **The 3D-aware model beats the capacity-matched 2D baseline out-of-domain**
   by about 0.05 AUROC, with parameter counts 0.14% apart and an identical
   epoch budget.
2. **Destroying the geometry does not hurt it.** Shuffling which atom holds
   which coordinate, every forward pass, leaves the test score unchanged or
   marginally better.

So the gain is real but is **not coming from the predicted coordinates**. The
defensible statement is:

> 3D-aware architecture helped out-of-domain. Predicted 3D coordinates did not.

Without the shuffle ablation the obvious report would have been "3D structure
helps on unseen targets, +0.05 AUROC" — a clean-sounding claim with the wrong
mechanism behind it. Whatever E3DTI does better must come from elsewhere: the
message-passing scheme, the protein contact graph, or the atom featurisation.

### The inversion worth noticing

The 2D baseline **wins on validation** (0.8866 vs 0.8491) and **loses on test**
(0.6590 vs 0.7085). A model that looks better in-distribution generalises worse
to genuinely unseen targets. This is invisible on the random split, where 97.4%
of test pairs reuse a drug and a protein already seen.

### DrugBAN baseline, full metrics

Threshold 0.4558, selected on validation:

| | |
|---|---|
| accuracy | 0.6227 |
| sensitivity | 0.3510 |
| specificity | 0.8973 |
| precision | 0.7756 |
| F1 | 0.4833 |

The validation-to-test collapse for this arm is **0.8866 → 0.6590**, a drop of
0.23 AUROC on meeting unseen drugs and proteins.

## In-domain

BioSNAP random, E3DTI full, 50-epoch configuration, seed 1: best **validation**
AUROC **0.9221** at epoch 37 (epoch 41 reached, 0.9209).

Labelled validation deliberately: the session was lost before the test
evaluation ran, and the checkpoint went with it. Do not quote this as a test
number.

## How to reproduce

```bash
python train_e3dti.py --dataset biosnap --split cluster --seed 1 --epochs 10 \
  --output-root runs-fast                                    # E3DTI full
python train_e3dti.py --dataset biosnap --split cluster --seed 1 --epochs 10 \
  --arch drugban_matched --output-root runs-fast             # 2D baseline
python train_e3dti.py --dataset biosnap --split cluster --seed 1 --epochs 10 \
  --ablation shuffled_coords --output-root runs-fast         # geometry control
```

Roughly 20 minutes per arm on a Kaggle T4, or about 10 for the baseline, which
is far cheaper per epoch.

The DrugBAN arm was run twice, once before a session loss and once after, and
returned identical numbers to four decimal places with the same best epoch.
That is an unplanned but welcome check on the seed discipline.

## Limits

- **One seed.** The 0.018 gap between full and shuffled cannot be separated
  from noise. The supportable claim is that shuffling does *not hurt*, never
  that it helps. Seeds 2 and 3 are the single most valuable thing left to run.
- **10 epochs, not 50.** Chosen to fit a deadline. Every arm got the same
  budget, so the comparison is fair, but none of them converged.
- **Protein contact topology survives the shuffle.** The k-NN edges are built
  once from the true coordinates and are not rebuilt, so for the protein branch
  the precise claim is that continuous geometry *beyond contacts* contributes
  nothing. The drug graph is fully connected, so the drug side is a clean test.
  Rebuilding protein edges from shuffled coordinates would close this.
- **One dataset.** BioSNAP only. A 50-epoch run costs ~3.9 GPU-hours and the
  full matrix in `experiment_protocol.md` is ~277.
- The baseline is a reimplementation inside this pipeline, not the official
  DrugBAN repository, so these numbers are **not** comparable with published
  DrugBAN results. See `README.md` for why that trade was made.

## Supporting measurements

From `docs/data_integrity_audit.md`, reproducible on CPU with no GPU needed:

| Measurement | Value |
|---|---|
| BioSNAP random test pairs with drug **and** protein seen in training | 97.4% |
| BindingDB random, same | 86.9% |
| Human random, same | 49.5% |
| cluster / cold splits, same | 0.0% |
| BioSNAP drugs with a defined stereocentre | 47.7% |
| Cluster validation pairs common to seeds 1/2/3, before the fix | 0.9% |
| After the fix | 100% |
| `human/cold` duplicate training rows | 394 |
| `human/cold` pairs in both validation and test | 12 |
