---
marp: true
theme: course
paginate: true
header: "06-763 / Final Project"
footer: "Systems and Toolchains for AI Engineers"
---

<!-- _class: title -->

# E3DTI: does predicted 3D structure help on unseen drug targets?

## Final project, Team 9

**Xingqiao · Yogesh · Jessie · Anthony**

<!-- 7 minutes with questions. Budget: 0:30 title+question, 1:30 the benchmark
     problem and the result, 1:00 what we built, 1:00 the two controls,
     0:45 chirality, 1:00 the defects, 1:00 where we are and what would change
     our mind.

     The audit and the collapse on slide 5 are the strongest material. If you
     are running long, cut the architecture slide, never the controls. -->

---

## The claim we are testing

Drug-target interaction models are scored on benchmarks where the test set
reuses proteins the model has already seen.

- **Hypothesis:** E(3)-equivariant geometry should help most where that
  crutch is removed, on **unseen** drugs and targets
- **Baseline:** DrugBAN, a 2D graph model, bilinear attention reused unchanged
- **Ours:** the same attention, with 3D-aware drug and protein encoders

<!-- Say "unseen" clearly. The whole talk turns on that word. -->

---

<!-- _class: section -->

# First: can the benchmark answer the question at all?

---

## Most of the benchmark is recognition, not prediction

Share of test pairs whose drug **and** protein both appear in training:

| Split | Test rows | Both seen |
|---|---|---|
| BioSNAP random | 5,480 | **97.4%** |
| BindingDB random | 9,836 | 86.9% |
| Human random | 1,191 | 49.5% |
| BioSNAP / BindingDB cluster, Human cold | — | **0.0%** |

On the standard split, a model can score well by remembering.

<span class="source">Measured with `e3dti.data.audit_splits` over the shipped DrugBAN CSVs.</span>

<!-- 60 seconds, do not rush it. The 0.0% row is the good news: the held-out
     splits really do hold out, so the next slide means something. -->

---

## What that costs: the out-of-domain collapse

BioSNAP **cluster** split — unseen drugs *and* unseen proteins.
Capacity-matched 2D baseline, 10 epochs, seed 1, best epoch 8.

| | AUROC | AUPRC |
|---|---|---|
| Validation | 0.8866 | 0.9003 |
| **Test** | **0.6590** | **0.6773** |

**A 0.23 AUROC collapse** the moment the model meets what it has not seen.

<span class="source">Threshold 0.456 selected on validation · accuracy 0.623 · specificity 0.897 · F1 0.483</span>

<!-- This is the result. Pause here. Same numbers reproduced exactly after a
     session loss forced a rerun, which is itself a check on the seed
     discipline. -->

---

## What we built

Two E(3)-equivariant encoders feeding DrugBAN's attention, unchanged:

- **Drug branch** — RDKit ETKDGv3 + MMFF94 conformers, fully connected graph
- **Protein branch** — AlphaFold 3 Cα traces, k-NN graph (K=16), RBF distances
- **Fusion** — DrugBAN's `BANLayer` / `FCNet` / `MLPDecoder`, reused as-is

Messages depend only on pairwise **distances**, so the model is invariant to
rotation and translation by construction.

<span class="source">Invariance is asserted in tests, not assumed: `tests/test_model_correctness.py`.</span>

<!-- 45 seconds. Resist explaining EGNN message passing; the test is the point,
     not the equations. -->

---

## Making the comparison fair

"3D helps" is easy to claim and easy to fake. Two controls:

| Control | What it rules out |
|---|---|
| **Capacity-matched baseline** — 1,649,040 vs 1,646,686 parameters, **0.14%** apart | a win that is really "the 3D model is bigger" |
| **Shuffled coordinates** — identical architecture, parameters and code path; coordinates permuted among each molecule's own atoms every forward pass | a win that is really "the 3D model had more inputs" |

If shuffling costs nothing, the model was never using the geometry.

<!-- This answers the check-in feedback directly. Worth saying that the
     capacity confound was raised by a reviewer and this is the fix. -->

---

## The drug branch could not see chirality

E(3) includes **reflections**. A reflection preserves every distance, so a
molecule and its mirror image embed **identically** — not approximately.

| Split | Drugs | With a stereocentre |
|---|---|---|
| BioSNAP random | 4,504 | **47.7%** |
| BindingDB random | 14,643 | 45.1% |
| Human cold | 1,802 | 43.6% |

Enantiomers routinely differ in activity. Fixed by adding the tetrahedral tag
from the **molecular graph** — the geometry cannot carry it.

<span class="source">Asserted at <code>atol=0</code> in `tests/test_chirality.py`; 6,091 tagged atoms confirmed in the cached conformers.</span>

<!-- 45 seconds. The method is the point: limitation derived from the
     architecture, measured for prevalence, fixed, pinned by a test. -->

---

<!-- _class: section -->

# Three defects found before we trained anything

---

## What the audit caught

| Defect | Effect | Status |
|---|---|---|
| Validation split keyed to the training seed | seeds 1/2/3 shared **0.9%** of validation pairs; seed spread mixed model variance with split-resampling variance | fixed, now 100% |
| `human/cold`: 394 duplicate training rows | 11.5% of the training set silently upweighted | opt-in clean |
| `human/cold`: 12 pairs in **both** validation and test | validation picks the checkpoint *and* the threshold, then those pairs are scored as held out | opt-in clean |

Cleaning is a flag, not a default: the shipped rows keep results comparable
with published DrugBAN numbers.

<!-- 60 seconds. The first row is a real reproducibility bug that would have
     inflated every error bar on the splits carrying our main claim. -->

---

## Where we are, honestly

**Measured**

- cluster, 2D matched baseline: **test AUROC 0.659** (validation 0.887)
- cluster, E3DTI, same 10-epoch budget: reached epoch 6,
  validation 0.708 → **0.836**, still climbing
- in-domain E3DTI: **validation** AUROC **0.9221** at epoch 37

**Limits we are not hiding**

- one seed, one dataset, a 10-epoch budget — a 50-epoch run costs 3.9 GPU-hours
  and the full matrix is ~277
- the E3DTI cluster arm had not converged when the budget ran out
- the `human/cold` contamination comparison is underpowered at n=304

<!-- Claim the collapse as the result. Claim E3DTI vs baseline as in progress:
     at epoch 6 it is below the baseline but has not converged, and that is a
     budget limitation, not a finding. Do not overclaim. -->

---

## What would change our mind

We will conclude 3D helps **only** if, on the held-out splits:

1. `full` beats `shuffled_coords` by more than the seed spread, **and**
2. `full` beats the capacity-matched baseline

Anything less and the honest answer is that predicted structure did not help
here — and that is worth reporting.

<!-- Close on this. 20 seconds, then questions. -->

---

<!-- _class: section -->

# Thank you — questions?

---

## Credits and AI use

**Reused unchanged:** DrugBAN `BANLayer`, `FCNet`, `MLPDecoder`
(Bai et al., *Nat. Mach. Intell.* 2023). Splits from the DrugBAN release.
Conformers via RDKit; structures from AlphaFold 3.

**AI assistance:** Claude Code was used for parts of the experiment pipeline,
the split audit, the chirality fix, the capacity-matched baseline,
checkpoint/resume support and this deck. Every generated change is attributed
in the git history via `Co-Authored-By` trailers and summarised in
`CREDITS.md`. **No results were generated, estimated or interpolated by AI** —
every number here comes from running the code in this repository.

<span class="source">Per the course generative-AI policy: disclose where and how AI assistants were used.</span>
