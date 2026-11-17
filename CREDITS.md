# Credits and AI use

The course generative-AI policy asks us to disclose where and how AI assistants
were used, to cite generated code and text, and to be able to explain and
defend everything we submit. This file is that disclosure for the
`yogesh/drug-egnn` branch.

## Reused work

| Component | Source | How it is used |
|---|---|---|
| `BANLayer`, `FCNet`, `MLPDecoder` | DrugBAN (Bai et al., *Nature Machine Intelligence*, 2023) | Reused unchanged, so the comparison isolates the encoders |
| Benchmark splits | DrugBAN release (BindingDB, BioSNAP, Human) | Used as shipped by default; `--clean-splits` is opt-in and documented |
| Drug conformers | RDKit ETKDGv3 + MMFF94 | Generated once and cached |
| Protein structures | AlphaFold 3 Cα traces with pLDDT | Loaded from the team's Hugging Face dataset |
| EGNN formulation | Satorras, Hoogeboom & Welling, ICML 2021 | Reimplemented |
| Slide theme `slides/themes/course.css` | `jkitchin/f26-06763`, course repository | Copied unmodified so the deck matches the course format |

## AI assistance

Claude Code (Anthropic) was used as a pair programmer on this branch. Every
commit it contributed to carries a `Co-Authored-By: Claude ...` trailer, so
`git log` is the authoritative record. Its contributions were:

- **Reproducibility fix** — diagnosing and fixing the cluster validation split
  being keyed to the training seed (`5b7d2f9`)
- **Data integrity audit** — the measurements in `docs/data_integrity_audit.md`
- **Checkpoint/resume support** for session-limited free GPUs (`878df49`), and
  the verification that it survives a real `SIGKILL` (`8812473`)
- **The shuffled-coordinate ablation** and its tests (`a0b5b7e`)
- **Matrix runner** for working through experiments inside quota (`94099e2`)
- **`--clean-splits`** and the seed-paired `human/cold` design (`7d400a0`)
- **This deck** (`slides/slides.md`)

What AI was *not* used for: no results were generated, estimated or
interpolated by an AI assistant. Every number in the report and the deck comes
from running the code in this repository, and the commands that produced them
are in `README.md` and `notebooks/kaggle_train.ipynb`.

## Things we checked rather than assumed

Recorded here because they are the parts we can defend most concretely:

- E(3) invariance and batch-padding independence are asserted in tests, not
  claimed in prose (`tests/test_model_correctness.py`)
- The shuffled-coordinate ablation is pinned to have parameters identical to
  the full model, so a score gap cannot be a capacity artifact
  (`tests/test_shuffled_coords_ablation.py`)
- `--resume` was verified by killing a run mid-flight and checking the restart
  was continuous (`scripts/verify_resume.py`)
- Cleaning was verified against the real `human/cold` files, which turned up
  duplicate rows inside validation and test that the first audit had missed

## Known limitations we are not hiding

- The matched-capacity DrugBAN baseline had not been run at the time of
  writing; without it the comparative claim is incomplete
- Results use 3 seeds on one dataset, not the full matrix, because a 50-epoch
  run costs ~3.9 GPU-hours and the full matrix is ~277
- The `human/cold` shipped-vs-cleaned comparison is underpowered at n=304 and
  should be read as paired per-seed differences, not as two means
- `egnn_dti.ipynb` contains a second, independent copy of the model; it is the
  original teaching notebook and has not been kept in step with `e3dti/`
