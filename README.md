# GreenVision — W8A1 + W9A1 Submission

A plant disease classifier built on **EfficientNet-B0** (pre-trained on ImageNet) and the **PlantVillage** dataset — 54,306 leaf images across **38** disease + healthy classes covering 14 crops, plus a **39th `Background_without_leaves` negative class** trained from non-plant photos so the model can reject non-leaf input. Training runs are tracked with **MLflow**, and the trained model is served behind a **FastAPI** endpoint.

This repository contains submissions for two assignments:

- **[W8A1] GreenVision Research & Implementation Design** — design phase (research log, 8 design decisions, AI guardrails, presentations)
- **[W9A1] GreenVision Training Pipeline** — implementation phase (runnable training pipeline, trained model registered as `GreenVision` @ Production in MLflow at **99.73% val accuracy / 99.73% test accuracy**, training report)

---

## 📋 W8A1 Submission Materials — where to find each deliverable

Every grading criterion in the assignment maps to one of the files below. Direct file paths from the repository root:

### Required deliverables

| # | Assignment requirement | File path | Rubric criterion |
|---|---|---|---|
| 1 | **Research log** — 3+ sessions, follow-up questions, evidence of wrestling with concepts | [`docs/RESEARCH_LOG.md`](./docs/RESEARCH_LOG.md) | Research depth (12 pts) |
| 2 | **Implementation guide** — all 8 design decisions addressed with reasoning, code snippets, cited sources | [`IMPLEMENTATION_GUIDE.md`](./IMPLEMENTATION_GUIDE.md) | Implementation guide completeness (16 pts) |
| 3 | **Copilot AI context** — project description, dataset details, critical constants, code conventions, architecture notes | [`.github/copilot-instructions.md`](./.github/copilot-instructions.md) | Implementation guide completeness (16 pts) |
| 4 | **Copilot Agent guardrails** — what Agent CAN do, what Agent MUST NOT do, files Agent should not modify | [`.github/agent.md`](./.github/agent.md) | Implementation guide completeness (16 pts) |
| 5 | **Audience observation notes** — notes from at least 5 peer presentations (filled in during W9D3) | [`docs/AUDIENCE_NOTES.md`](./docs/AUDIENCE_NOTES.md) | Peer engagement (10 pts) |

### Presentation materials (used live in class — not file-submitted but tracked here for reference)

| Day | Content | File path |
|---|---|---|
| **W9D2** (Tue, teaching topic) | Interactive AdamW vs. Adam visual demo — neural network animation with simulated gradients, the bug/fix story baked into the optimizer toggle | [`adamw-vs-adam-demo.html`](./adamw-vs-adam-demo.html) |
| **W9D3** (Wed, implementation guide) | One-slide-per-decision outline for the 8-minute talk — sidebar TOC, keyboard navigation, locked-choice bullets | [`presentations/w9d3-decisions.html`](./presentations/w9d3-decisions.html) |

---

## 📋 W9A1 Submission Materials — where to find each deliverable

| # | Assignment requirement | File path | Rubric criterion |
|---|---|---|---|
| 1 | **Training pipeline** — runnable end-to-end | [`scripts/train.py`](./scripts/train.py) + [`src/greenvision/`](./src/greenvision/) | Pipeline correctness (15 pts) |
| 2 | **Trained model** — `GreenVision` @ Production, >80% val acc | MLflow Registry (file store: `./mlruns/`) — promoted via [`scripts/promote.py`](./scripts/promote.py) | Pipeline correctness (15 pts) |
| 3 | **Strategy documented** — fine-tuning justified | [`IMPLEMENTATION_GUIDE.md`](./IMPLEMENTATION_GUIDE.md) Decisions 2 + 5 | Strategy justification (12.5 pts) |
| 4 | **Design consistency** — ImageNet norm, class names preserved | [`src/greenvision/data/transforms.py`](./src/greenvision/data/transforms.py) + [`class_names.py`](./src/greenvision/data/class_names.py) | Design consistency (12.5 pts) |
| 5 | **MLflow + context files** — experiments tracked + AI guardrails | [`src/greenvision/training/mlflow_utils.py`](./src/greenvision/training/mlflow_utils.py) + [`.github/copilot-instructions.md`](./.github/copilot-instructions.md) + [`.github/agent.md`](./.github/agent.md) | MLflow + context (10 pts) |
| 6 | **`docs/TRAINING_REPORT.md`** — final results + reflection | [`docs/TRAINING_REPORT.md`](./docs/TRAINING_REPORT.md) | Required deliverable |
| 7 | **MLflow screenshots** — val accuracy curve + Registry Production | [`docs/screenshots/Val_accuracy.png`](./docs/screenshots/Val_accuracy.png), [`model_registry.png`](./docs/screenshots/model_registry.png), bonus [`training_curves.png`](./docs/screenshots/training_curves.png) | Required deliverable |

### Reference

- W8A1 assignment brief: [`ASSIGNMENT.md`](./ASSIGNMENT.md)
- W9A1 assignment brief: [`ASSIGNMENT_W9A1.md`](./ASSIGNMENT_W9A1.md)
- Build plan + Claude Code prompts used during W9A1: [`docs/IMPLEMENTATION_PLAN.md`](./docs/IMPLEMENTATION_PLAN.md) + [`docs/CLAUDE_PROMPTS.md`](./docs/CLAUDE_PROMPTS.md)

---

## What's in this repository

```
GreenVision/
├── ASSIGNMENT.md                       # Full assignment brief (reference)
├── IMPLEMENTATION_GUIDE.md             # ⭐ Deliverable: 8 design decisions
├── README.md                           # This file
├── adamw-vs-adam-demo.html             # ⭐ Tuesday W9D2 teaching demo
├── lesson.md                           # Quinn's personal AdamW study notes
├── requirements.txt
├── .gitignore
│
├── .github/
│   ├── copilot-instructions.md         # ⭐ Deliverable: AI context
│   └── agent.md                        # ⭐ Deliverable: Agent guardrails
│
├── docs/
│   ├── RESEARCH_LOG.md                 # ⭐ Deliverable: 5 research sessions
│   └── AUDIENCE_NOTES.md               # ⭐ Deliverable: peer-presentation notes
│
├── presentations/
│   └── w9d3-decisions.html             # ⭐ Wednesday W9D3 presentation outline
│
├── src/greenvision/                    # Code skeleton (implemented in W9–W10)
│   ├── data/                           #   ImageFolder, transforms, splits
│   ├── models/                         #   EfficientNet-B0 + custom head
│   ├── training/                       #   Two-phase loop, MLflow, optim, schedulers
│   └── inference/                      #   Predict + preprocessing
├── api/                                # FastAPI app (skeleton; built in W9–W10)
├── notebooks/                          # Exploratory work
├── configs/                            # YAML / dotenv configs
├── scripts/                            # CLI entry points (train, eval, export)
├── tests/                              # Unit tests
│
├── data/                               # raw/ + processed/ (gitignored)
└── artifacts/                          # checkpoints/ + mlruns/ (gitignored)
```

Items marked **⭐** are the W8A1 submission deliverables.

---

## Project summary

**GreenVision** classifies an uploaded leaf image into one of **39 classes** — 38 PlantVillage disease + healthy classes across 14 crops, plus a 39th `Background_without_leaves` negative class for rejecting non-leaf inputs — using EfficientNet-B0 fine-tuned on PlantVillage.

**Target users:** growers, agronomists, and ag-tech apps that need a quick triage signal before reaching for more expensive diagnostics.

**Architecture (the four non-negotiables from the assignment):**

- **Base model:** EfficientNet-B0, pre-trained on ImageNet
- **Dataset:** PlantVillage — 54,306 leaf images across 38 disease + healthy classes (14 crops) + a `Background_without_leaves` folder = **39 classes total**
- **Experiment tracking:** MLflow (nested runs, file-backed)
- **Serving:** FastAPI (multipart primary, base64 secondary, Pydantic responses)

For the full design rationale behind every choice (head structure, two-phase fine-tuning with gradual unfreezing, AdamW with decoupled weight decay, ReduceLROnPlateau, train/inference parity, and the rest), see [`IMPLEMENTATION_GUIDE.md`](./IMPLEMENTATION_GUIDE.md). For the research that informed those choices, see [`docs/RESEARCH_LOG.md`](./docs/RESEARCH_LOG.md).

---

## Status

**Design phase complete (W8A1) + implementation phase complete (W9A1).**

- All eight design decisions are locked and documented in `IMPLEMENTATION_GUIDE.md`
- Training pipeline is runnable end-to-end via `python scripts/train.py --attempt-id N`
- Model is trained, registered as `GreenVision` v3 @ **Production** in MLflow Registry
- Final results: **val 99.73% / test 99.73%** (1/39 random baseline = 2.56%, so +97.17 pp improvement)
- **Next:** W10A1 — build the FastAPI serving layer that loads `models:/GreenVision/Production` (the design is locked in `IMPLEMENTATION_GUIDE.md` Decision 8).

**For graders:** every rubric criterion for both W8A1 and W9A1 can be checked against a single file listed in the submission materials tables above.
