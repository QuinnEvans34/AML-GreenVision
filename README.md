# GreenVision — W8A1 + W9A1 + W10A1 Submission

A plant disease classifier built on **EfficientNet-B0** (pre-trained on ImageNet) and the **PlantVillage** dataset — 54,306 leaf images across **38** disease + healthy classes covering 14 crops, plus a **39th `Background_without_leaves` negative class** trained from non-plant photos so the model can reject non-leaf input. Training runs are tracked with **MLflow**, and the trained model is served behind a **FastAPI** endpoint.

This repository contains submissions for three assignments:

- **[W8A1] GreenVision Research & Implementation Design** — design phase (research log, 8 design decisions, AI guardrails, presentations)
- **[W9A1] GreenVision Training Pipeline** — implementation phase (runnable training pipeline, trained model registered as `GreenVision` @ Production in MLflow at **99.73% val accuracy / 99.73% test accuracy**, training report)
- **[W10A1] GreenVision Final Submission & Presentation** — full system + presentation (FastAPI inference backend, Next.js dashboard with 4 React-Three-Fiber 3D visualizations, treatment knowledge base, live demo orchestration)

## 🚀 Quick start

```bash
# One command starts FastAPI + Next.js dashboard + MLflow UI together
./scripts/demo.sh

# Then open:
#   http://localhost:3000           — dashboard (the demo surface)
#   http://localhost:3000/analytics — 3D visualization suite
#   http://localhost:8000/docs      — FastAPI Swagger
#   http://localhost:5001           — MLflow UI
```

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

---

## 📋 W10A1 Submission Materials — where to find each deliverable

| # | Assignment requirement | File path | Rubric ref |
|---|---|---|---|
| 1 | **FastAPI inference endpoint** — `/health` + `/predict` | [`api/main.py`](./api/main.py) + [`api/routes/`](./api/routes/) + [`api/inference.py`](./api/inference.py) | Part 1 |
| 2 | **Loads from MLflow Registry** | `models:/GreenVision/Production` via lifespan in [`api/inference.py`](./api/inference.py) | Part 1 |
| 3 | **Class names from MLflow artifact** | [`api/inference.py:load_class_names()`](./api/inference.py) reads `artifacts/checkpoints/class_names.json` | Part 1 |
| 4 | **Same validation transforms** | API imports `eval_tfms` from [`src/greenvision/data/transforms.py`](./src/greenvision/data/transforms.py) — same object as training | Part 1 |
| 5 | **Treatment knowledge base** | [`data/treatments.json`](./data/treatments.json) — 39 entries with severity, action steps, citations to UMN / Penn State / Cornell / UC IPM | Part 1 |
| 6 | **Dashboard** | [`web/`](./web/) — Next.js 16 + React 19 + Tailwind 4 + shadcn/ui | Part 2 |
| 7 | **Image upload + preview** | [`web/components/upload-card.tsx`](./web/components/upload-card.tsx) | Part 2 |
| 8 | **Health check + informative API-down error** | [`web/app/page.tsx`](./web/app/page.tsx) — mount-time `health()` + alert banner | Part 2 |
| 9 | **Disease name formatted for readability** | [`web/lib/format-class-name.ts`](./web/lib/format-class-name.ts) — `"Tomato___Late_blight"` → `"Tomato — Late blight"` | Part 2 |
| 10 | **Visual confidence bands** | [`web/components/confidence-badge.tsx`](./web/components/confidence-badge.tsx) — high/moderate/medium/low color-coded chips | Part 2 |
| 11 | **AI disclaimer** | [`web/components/ai-disclaimer.tsx`](./web/components/ai-disclaimer.tsx) — always present, stronger at low confidence | Part 2 |
| 12 | **End-to-end integration** | [`scripts/demo.sh`](./scripts/demo.sh) starts all three services with pre-flight checks | Part 3 |
| 13 | **3D visualization suite (beyond rubric)** | [`web/components/viz/`](./web/components/viz/) — 4 React-Three-Fiber scenes + 2D overview + sortable per-class table | Beyond Part 2 |
| 14 | **Presentation script + Q&A defense** | [`docs/PRESENTATION_SCRIPT.md`](./docs/PRESENTATION_SCRIPT.md) — 10-min beat sheet + 10 Q&A paragraphs + backup paths | Part 4 |
| 15 | **Design decisions for W10A1** | [`IMPLEMENTATION_GUIDE.md`](./IMPLEMENTATION_GUIDE.md) Decisions 9-13 — treatments KB, UX patterns, viz stack, Next.js arch, demo orchestration | Part 1 + 2 |

### W10 build plan

- [`ASSIGNMENT_W10A1.md`](./ASSIGNMENT_W10A1.md) — assignment reference
- [`docs/W10_IMPLEMENTATION_PLAN.md`](./docs/W10_IMPLEMENTATION_PLAN.md) — 9-phase build plan with per-phase verification
- [`docs/PRESENTATION_SCRIPT.md`](./docs/PRESENTATION_SCRIPT.md) — Wednesday presentation runbook

---

### Reference

- W8A1 assignment brief: [`ASSIGNMENT.md`](./ASSIGNMENT.md)
- W9A1 assignment brief: [`ASSIGNMENT_W9A1.md`](./ASSIGNMENT_W9A1.md)
- W10A1 assignment brief: [`ASSIGNMENT_W10A1.md`](./ASSIGNMENT_W10A1.md)
- W9A1 build plan + prompts: [`docs/IMPLEMENTATION_PLAN.md`](./docs/IMPLEMENTATION_PLAN.md) + [`docs/CLAUDE_PROMPTS.md`](./docs/CLAUDE_PROMPTS.md)

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
├── src/greenvision/                    # Training-time Python package
│   ├── data/                           #   ImageFolder, transforms, splits, class_names
│   ├── models/                         #   EfficientNet-B0 + custom head
│   └── training/                       #   Two-phase loop, MLflow, optim, schedulers, registry
├── api/                                # FastAPI serving (W10A1)
│   ├── main.py                         #   app + lifespan + CORS
│   ├── inference.py                    #   model load + predict_image
│   ├── treatments.py                   #   KB loader
│   ├── schemas.py                      #   Pydantic v2 models
│   └── routes/                         #   /health, /predict
├── web/                                # Next.js dashboard (W10A1)
│   ├── app/                            #   pages: /, /analytics, /about
│   ├── components/                     #   upload-card, prediction-result, etc.
│   ├── components/viz/                 #   4 React-Three-Fiber scenes + 2D overview + per-class table
│   ├── lib/                            #   types, api-client, format-class-name, use-training-data
│   └── public/training_data.json       #   baked from MLflow by export script
├── scripts/                            # CLI entry points
│   ├── train.py                        #   training orchestrator
│   ├── promote.py                      #   promote a version to Production
│   ├── export_mlflow_for_dashboard.py  #   bake training_data.json
│   ├── demo.sh                         #   one-command full demo launch
│   ├── demo_reset.sh                   #   free demo ports
│   └── demo_static.sh                  #   backup demo path (no API)
│
├── data/raw/PlantVillage/              # dataset (gitignored)
├── data/treatments.json                # 39-entry treatment KB (W10A1, committed)
├── mlruns/                             # MLflow tracking + Model Registry (gitignored)
└── artifacts/checkpoints/              # class_names.json, best.pt (gitignored)
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

**Design phase (W8A1) + training pipeline (W9A1) + serving + dashboard + presentation (W10A1) all complete.**

- All 13 design decisions are locked and documented in `IMPLEMENTATION_GUIDE.md` (8 from W8A1 + 5 added for W10A1).
- Training pipeline runnable end-to-end via `python scripts/train.py --attempt-id N`.
- Model trained, registered as `GreenVision` v3 @ **Production** in MLflow Registry — val 99.73% / test 99.73%.
- Serving layer: FastAPI loading from the Registry, dashboard built on Next.js + React Three Fiber with 4 interactive 3D scenes.
- Full demo launches with one command: `./scripts/demo.sh`.

**For graders:** every rubric criterion for W8A1, W9A1, and W10A1 maps to a single file listed in the submission materials tables above.
