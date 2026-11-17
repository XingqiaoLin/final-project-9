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

<!-- 7 minutes with questions. Budget: 0:30 title+question, 1:30 why the
     benchmark cannot answer it, 1:30 what we built, 1:00 the control,
     1:30 what we found before training, 1:00 results and limits.
     The audit is the strongest material. If you are running long, cut the
     architecture slide, not the audit. -->

---

## The claim we are testing

Drug-target interaction models are scored on benchmarks where the test set
reuses proteins the model has already seen.

- **Hypothesis:** E(3)-equivariant geometry should help most where that
  crutch is removed, on **unseen** drugs and targets
- **Baseline:** DrugBAN, a 2D graph model, bilinear attention unchanged
- **Our model:** the same attention, with 3D-aware drug and protein encoders

<div class="definition">

**E(3)-equivariant**: predictions are unchanged by rotating, translating or
reflecting the input coordinates.

</div>

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

<!-- This is the slide that earns the project. 60 seconds, do not rush it.
     The 0.0% row is the good news: the held-out splits really do hold out. -->

---

## What we built

Two E(3)-equivariant encoders feeding DrugBAN's attention, unchanged:

- **Drug branch** — RDKit ETKDGv3 + MMFF94 conformers, fully connected graph
- **Protein branch** — AlphaFold 3 Cα traces, k-NN graph (K=16), RBF distances
- **Fusion** — DrugBAN's `BANLayer` / `FCNet` / `MLPDecoder`, reused as-is

Messages depend only on pairwise **distances**, so the model is invariant to
rotation, translation and reflection by construction.

<span class="source">Invariance is asserted in CI, not assumed: `tests/test_model_correctness.py`.</span>

<!-- 45 seconds. Resist explaining EGNN message passing in detail; the
     invariance test is the point, not the equations. -->

---

## The control that makes it a fair test

Switching geometry off also removes parameters, so a drop in score could just
mean a smaller model.

Instead we **shuffle the coordinates** among each molecule's own atoms:

- identical architecture, identical parameter count, identical code path
- a fresh permutation every forward pass
- only *which atom sits where* is destroyed

<div class="definition">

If the model scores the same with shuffled coordinates, it was **not using
the geometry**.

</div>

<!-- This answers the check-in feedback directly. Worth saying out loud that
     the capacity confound was raised by a reviewer and this is the fix. -->

---

## The drug branch could not see chirality

Messages are built from **distances**. A reflection preserves every distance,
so a molecule and its mirror image embed **identically** — not approximately,
exactly.

| Split | Drugs | With a stereocentre |
|---|---|---|
| BioSNAP random | 4,504 | **47.7%** |
| BindingDB random | 14,643 | 45.1% |
| Human cold | 1,802 | 43.6% |

Enantiomers routinely differ in activity, so for nearly half the drugs the
model was blind to what decides the answer. Fixed by adding the tetrahedral
tag from the **molecular graph** — geometry cannot carry it.

<span class="source">Asserted at <code>atol=0</code> in `tests/test_chirality.py`; 6,091 tagged atoms confirmed in the cached conformers.</span>

<!-- 45 seconds. The point is the method: a limitation derived from the
     architecture, measured for prevalence, fixed, then pinned by a test. -->

---

<!-- _class: section -->

# What we found before we trained anything

---

## Three defects in the pipeline and the data

| Defect | Effect | Status |
|---|---|---|
| Validation split keyed to the training seed | Seeds 1/2/3 shared **0.9%** of validation pairs; seed spread mixed model variance with split variance | fixed |
| `human/cold`: 394 duplicate training rows | 11.5% of training upweighted | opt-in clean |
| `human/cold`: 12 pairs in **both** validation and test | Validation picks the checkpoint *and* the threshold, then those pairs are scored as held out | opt-in clean |

Cleaning is a flag, not a default: the shipped rows keep results comparable
with published DrugBAN numbers.

<!-- 60 seconds. The first row is a genuine reproducibility bug that would
     have inflated every error bar on the splits carrying our main claim. -->

---

## Where we are

**In-domain, BioSNAP random, seed 1, one run:** validation AUROC **0.914**

- roughly level with the 2D baseline — which is the *expected* result
- 97.4% of those test pairs are recognition, so 3D has little to add there
- the cluster split is where the hypothesis is actually tested, and those
  runs are still in flight

**Honest limits**

- 3 seeds, not 5; one dataset, not three — ~4.6 min/epoch on a free T4 makes
  the full 72-run matrix about 277 GPU-hours
- the `human/cold` contamination comparison is underpowered at n=304

<!-- Do not oversell. A controlled negative result is a result. -->

---

## What would change our mind

We will conclude 3D helps **only** if, on the held-out splits:

1. `full` beats `shuffled_coords` by more than the seed spread, **and**
2. `full` beats the matched-capacity DrugBAN baseline

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
the split audit, the shuffled-coordinate ablation, checkpoint/resume support
and this deck. Every generated change is attributed in the git history via
`Co-Authored-By` trailers and summarised in `CREDITS.md`. All results were
produced by running the code, and we can explain and defend each number.

<span class="source">Per the course generative-AI policy: disclose where and how AI assistants were used.</span>
