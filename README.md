# GreenVision — W8A1 Submission

A plant disease classifier built on **EfficientNet-B0** (pre-trained on ImageNet) and the **PlantVillage** dataset — 54,306 leaf images across **38** disease + healthy classes covering 14 crops. Training runs are tracked with **MLflow**, and the trained model is served behind a **FastAPI** endpoint.

This repository is the design-phase submission for assignment **[W8A1] GreenVision Research & Implementation Design**.

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

### Reference

- Full assignment brief: [`ASSIGNMENT.md`](./ASSIGNMENT.md)
- Project description and status: see the rest of this README

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

**GreenVision** classifies an uploaded leaf image into one of 38 PlantVillage classes (disease state × crop, plus healthy classes) across 14 crops, using EfficientNet-B0 fine-tuned on PlantVillage.

**Target users:** growers, agronomists, and ag-tech apps that need a quick triage signal before reaching for more expensive diagnostics.

**Architecture (the four non-negotiables from the assignment):**

- **Base model:** EfficientNet-B0, pre-trained on ImageNet
- **Dataset:** PlantVillage — 54,306 images, 38 classes, 14 crops
- **Experiment tracking:** MLflow (nested runs, file-backed)
- **Serving:** FastAPI (multipart primary, base64 secondary, Pydantic responses)

For the full design rationale behind every choice (head structure, two-phase fine-tuning with gradual unfreezing, AdamW with decoupled weight decay, ReduceLROnPlateau, train/inference parity, and the rest), see [`IMPLEMENTATION_GUIDE.md`](./IMPLEMENTATION_GUIDE.md). For the research that informed those choices, see [`docs/RESEARCH_LOG.md`](./docs/RESEARCH_LOG.md).

---

## Status

**Design phase complete.** All eight design decisions are locked and documented; this repository contains the W8A1 deliverables only. The actual training code (currently a skeleton under `src/greenvision/` and `api/`) is built out during Weeks 9–10 of the course following the roadmap laid out in `IMPLEMENTATION_GUIDE.md`.

**For graders:** every rubric criterion can be checked against a single file listed in the "W8A1 Submission Materials" table above.
