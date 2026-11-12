# Lab #5 - System Selection and Reproducibility Record

## People and collaboration

- Working mode: [x] Solo  [ ] Pair
- Student A (name / GitHub): Moutaz Khalfalla / [moutazsalah](https://github.com/moutazsalah)
- Student B (name / GitHub; if pair): Not applicable
- Fork owner: moutazsalah
- Collaborator invitation accepted? Not applicable
- Individual work differentiation: Not applicable (solo); all work in this record is my own.

## First shortlist - catalogue only

| Candidate | Catalogue-based reason | One concern to investigate |
|---|---|---|
| A — timm (#16, Core CV) | Widely used image-classification library with many pretrained models and documented evaluation | Very large codebase: can it be scoped and run on a laptop? |
| B — Segmentation Models PyTorch (#15, Core CV) | Smaller, well-structured segmentation library; reuses timm encoders | Is there an evaluation ground truth beyond example notebooks? |

## Evidence-based candidate assessment

Use Y = Yes; P = Partial; N = No; U = Unknown. Checked on 2026-11-12 in local clones: timm at tag `v1.0.31`, SMP at `main` (latest tag `v0.6.0`).

| Criterion | Candidate A: timm + evidence | Candidate B: SMP + evidence |
|---|---|---|
| C1 Public repository / licence / data rights | **P** — public repo, code Apache-2.0 ([LICENSE](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/LICENSE)). Pretrained weights inherit ImageNet's non-commercial research terms; some weights are CC-BY-NC ([README → Licenses](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/README.md#licenses)). Fine for coursework. | **Y** — public repo, MIT ([LICENSE](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/LICENSE)); encoder weights come from timm and inherit its terms |
| C2 Pinned revision / runtime | **Y** — tag `v1.0.31` (2026-11-03), SHA `43b549bf…`; `requires-python >= 3.10` ([pyproject.toml](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/pyproject.toml)); runs on Python 3.13.3 + torch 2.15.0 | **Y** — tag `v0.6.0` (2026-10-26); CI tests Python 3.12–3.14 |
| C3 Nontrivial source + refactoring | **Y** — ~166k lines of Python in `timm/` (8 subpackages). Lizard: 5,958 functions, avg CCN 2.9, 92 over thresholds; e.g. `_load_weights` in `timm/models/vision_transformer.py` CCN 68 | **Y** — ~18k lines of Python; smaller but nontrivial (encoders, decoders, losses, metrics) |
| C4 Tests and expected-output **oracle** | **P** — `tests/` with 588 test functions, parametrised over models. Oracles are mostly **property checks** (batch size, no NaNs, feature shapes, e.g. [`tests/test_models.py` L201–202](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/tests/test_models.py#L201-L202)), not exact expected predictions | **Y** — 83 test functions; coverage measured in CI ([tests.yml](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/.github/workflows/tests.yml)) |
| C5 AI/recommender **ground-truth** source and terms | **Y** — [`results/`](https://github.com/huggingface/pytorch-image-models/tree/v1.0.31/results): top-1/top-5 for ~1,550 models on ImageNet-1k val + 5 test sets (Real labels, V2, Sketch, -A, -R), sources cited in [results/README.md](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/results/README.md). ImageNet: registration, non-commercial research only | **P** — metrics module (IoU, F1…) and CamVid / buildings example notebooks; no published results table; dataset terms not stated (U) |
| C6 Evaluation procedure and metrics | **Y** — [`validate.py`](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/validate.py) (top-1/top-5, loss); [`benchmark.py`](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/benchmark.py) (throughput) | **P** — `smp.metrics` + training loops in notebooks; no single evaluation script |
| C7 Upstream GitHub history | **Y** — 3,393 commits, 232 contributors, 401 commits in the last 12 months, frequent tags | **Y** — 898 commits, 81 contributors, 320 commits in the last 12 months |
| C8 CI/DevOps possibility | **Y** — GitHub Actions [`tests.yml`](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/.github/workflows/tests.yml) (Python 3.10/3.14 × 10 pytest markers), docs build, TruffleHog. **No coverage step** → concrete Lab 7 opportunity | **Y** — lint (Ruff), tests on Ubuntu/macOS/Windows with Codecov, PyPI release workflow |
| C9 Laptop setup feasibility | **Y** — installed from source on MacBook M3 Pro (macOS 14.6), CPU; ~19 img/s inference; 1,583 tests in 1 min 43 s. Apple GPU (MPS) crashed → CPU only | **U** — not installed (expected easy: same torch + timm stack) |
| C10 Continuation to Labs 6-9 / credible question | **Y** — e.g. "Do the published accuracies of pretrained models hold on an independent labelled subset, and are the data-pipeline (`timm/data`) functions where complexity and change concentrate?" | **P** — good for Labs 6/7; weak ground truth limits Lab 8 |

### Ground-truth verification notes

- Candidate A (timm): functional test assertion: [`tests/test_models.py` L201–202](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/tests/test_models.py#L201-L202) (`outputs.shape[0] == batch_size`, no NaNs) — a shape/sanity oracle, not an expected-prediction oracle; documented expected output: [Quickstart](https://huggingface.co/docs/timm/quickstart#using-pretrained-models-for-inference) prints top-5 probabilities for a fixed image, which I reproduced exactly. Label source: ImageNet-1k human labels (+ ReaL relabelling, ImageNetV2, Sketch, -A, -R), per [results/README.md](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/results/README.md). Train/test protocol: trained on ImageNet-1k train, evaluated on the 50k validation set; the README notes this set was also used for model selection, so it is "not a true test set" — the extra test sets partly address this. Data terms: ImageNet for non-commercial research only; some weights CC-BY-NC.
- Candidate B (SMP): functional tests in `tests/` (forward-pass and loss checks); ground truth only via example datasets (CamVid, building masks) in notebooks; no fixed train/test protocol or results file in the repo; dataset terms not documented in the repo (Unknown).

## Decision

- Final SUT and short description: **timm v1.0.31** (PyTorch Image Models): a library of ~1,300 image-classification architectures and pretrained weights, with data loading, training and validation scripts.
- Why it is better suited than rejected alternatives: it is the only candidate with a published, cited AI ground truth and an evaluation script (C5, C6), its history is the richest (C7), and it already installs and runs from my fork on my laptop (C9).
- Rejected alternatives and reasons: SMP — smaller and with better CI coverage, but no published results or fixed evaluation protocol, so Lab 8 would rest on example notebooks. Cornac was not considered because another student already registered it.
- Backup SUT and trigger for switching: **SMP v0.6.0**. Switch if the instructor considers timm too large to scope, or if Lab 8 evaluation on a labelled subset cannot be run on CPU in reasonable time.
- Biggest unresolved risk / mitigation: **size** (~166k lines, `timm/models` alone ~102k). Mitigation: fix a scope per lab, e.g. `timm/data` (~9k lines) for static analysis and history, and 2–3 pretrained models for evaluation. Second risk: full ImageNet is too large for the laptop → use a small labelled subset (e.g. Imagenette) and report that numbers are not 1:1 comparable with `results/`.

## Repository provenance and pair access

- Original upstream repo URL: https://github.com/huggingface/pytorch-image-models
- Selected original tag (if any): `v1.0.31` — commit `43b549bfe9d680bc94284cfd0b611152d18e97d7`
- Fork URL: https://github.com/moutazsalah/pytorch-image-models
- Working branch: [`semester-investigation`](https://github.com/moutazsalah/pytorch-image-models/tree/semester-investigation)
- Git and runtime versions: git 2.39.5 (Apple Git-154); Python 3.13.3; pip 26.3; torch 2.15.0 (CPU); macOS 14.6, Apple M3 Pro, 18 GB
- Applicable code licence (link + note): [Apache-2.0](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/LICENSE) — permits use, modification and redistribution of the code
- External dataset licence/limits: ImageNet — registration required, non-commercial research only ([image-net.org/download](https://image-net.org/download)); pretrained weights assumed to inherit it; some weights CC-BY-NC 4.0 ([README → Licenses](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/README.md#licenses))
- Partner collaborator invitation accepted? (pair only): Not applicable

Setup commands:

```bash
git clone https://github.com/moutazsalah/pytorch-image-models.git
cd pytorch-image-models
git remote add upstream https://github.com/huggingface/pytorch-image-models.git
git fetch upstream --tags
git switch -c semester-investigation v1.0.31
git rev-parse HEAD   # 43b549bfe9d680bc94284cfd0b611152d18e97d7
git push -u origin semester-investigation
```

## Actual execution - not just installation

Script: [`exploration/try_timm.py`](exploration/try_timm.py); full output: [`exploration/try_timm_output.txt`](exploration/try_timm_output.txt).

| Check | Official documentation link | Exact local command / path + input | Observed output / short log + evidence link | Status |
|---|---|---|---|---|
| Install/build source-aware baseline | [Installation](https://huggingface.co/docs/timm/installation) | `python3.13 -m venv .venv && pip install -e <fork>`; then `pytest tests/test_layers.py tests/test_optim.py` | `timm.__file__` → `<fork>/timm/__init__.py`, version 1.0.31; **1,583 passed, 279 skipped** in 1 min 43 s | Success |
| Functionality 1 (real behaviour) | [Quickstart → Using Pretrained Models for Inference](https://huggingface.co/docs/timm/quickstart#using-pretrained-models-for-inference) | `python exploration/try_timm.py`, section [1]: `create_model('mobilenetv3_large_100', pretrained=True)` + `create_transform(**resolve_model_data_config(model))` on the documented `cat.jpg` | Top-5 `[281, 282, 285, 673, 670]`: tabby cat 0.5101, tiger cat 0.2249, Egyptian cat 0.1835, … — **identical to the docs** ([output](exploration/try_timm_output.txt)) | Success |
| Functionality 2 (different behaviour) | [Feature Extraction → Create a feature map extraction model](https://huggingface.co/docs/timm/feature_extraction#create-a-feature-map-extraction-model) | same script, section [2]: `create_model('resnest26d', features_only=True, pretrained=True)` on random input `(2, 3, 224, 224)` | 5 maps `(2,64,112,112)` … `(2,2048,7,7)`; `feature_info` channels `[64,256,512,1024,2048]`, reductions `[2,4,8,16,32]` — matches the docs ([output](exploration/try_timm_output.txt)) | Success |

- Is the installed code from the local source fork, or only a separately installed wheel/image? **From the local source fork** (editable install of `moutazsalah/pytorch-image-models` at `v1.0.31`; verified with `timm.__file__`).
- Blockers and fallback date/action: no blockers for CPU use. Apple GPU (MPS) crashes with torch 2.15 on macOS 14.6 → all work on CPU. Full ImageNet not downloaded → choose a small labelled subset before Lab 8.

## Approval / action

- Instructor status: Pending
- Changes needed before Lab #6: fix the analysis scope (proposed: `timm/data`); select the labelled evaluation subset for Lab 8.
- Date of record: 2026-11-12

## Appendix - LLM-assisted feasibility for Labs 6-9 (verified)

**Prompt used:** "Is timm (huggingface/pytorch-image-models, v1.0.31) suitable as a system under test for (1) static analysis / technical debt / refactoring, (2) DevOps / CI/CD / observability, (3) AI / computer-vision quality evaluation, (4) an empirical mini-study? For each, suggest investigations, tools and the main concern."

| Future lab | Suitable? | Possible investigation / tools | Main concern |
|---|---|---|---|
| Lab 6 — Static analysis, technical debt, refactoring | Yes | Lizard / Radon, Ruff / Pylint, SonarQube on `timm/data`. *Verified:* Lizard runs on the package; hotspots e.g. `_load_weights` (CCN 68), `transforms_factory.py` (avg CCN 15) | Many similar model files distort whole-repo averages → scope to a subsystem |
| Lab 7 — DevOps, CI/CD, observability | Yes | Analyse GitHub Actions matrix; add a coverage step in the fork; time tests per marker. *Verified:* `tests.yml` has 2 × 10 jobs and **no coverage step** | Full suite is heavy → run selected markers |
| Lab 8 — AI / CV quality | Yes | `validate.py` on a labelled subset; compare with `results/*.csv`; distribution shift (Sketch/-R). *Verified:* results CSVs (~1,550 models, 6 test sets) and `validate.py` present | Full ImageNet too large → subset; numbers not 1:1 |
| Lab 9 — Empirical mini-study | Yes | Mine history with PyDriller: complexity vs changes, bug-fix commits vs tests. *Verified:* 3,393 commits, 232 contributors, 401 commits in last year | Bug-fix commits not consistently labelled → manual checks |

**LLM used:** Claude (Opus 5.5, via Claude Code), 2026-11-12. Each claim was then checked in the repository or by running it.

**LLM claims that did not hold or needed correction:**
- *"Licence: Yes (Apache-2.0)."* Only true for the code. The README states the pretrained weights inherit ImageNet's non-commercial research terms and some are CC-BY-NC → C1 downgraded to **Partial**.
- *"Testing: Yes, 588 test functions."* The count is right, but the assertions are property checks (batch size, no NaNs, feature shapes), not expected-output oracles → C4 downgraded to **Partial**.
- *"Runs fine on Apple Silicon."* Only on CPU: the Apple GPU backend (MPS) crashed with torch 2.15 on macOS 14.6.
- *"Lab 8: compare against the published ImageNet results."* Not reproducible 1:1 on a laptop: ImageNet needs registration and is several GB, and the validation set was also used for model selection (`results/README.md`). A labelled subset is needed and the comparison must be reported as approximate.
- The first install was from a separate clone, not the fork; the template asks for the fork, so I reinstalled from it and confirmed with `timm.__file__`.
