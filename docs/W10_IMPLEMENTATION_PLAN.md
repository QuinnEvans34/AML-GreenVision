# GreenVision Final Submission & Presentation — Implementation Plan (W10A1)

> Build plan for the serving layer (FastAPI), dashboard (Next.js), 3D visualization suite, and W10D3 presentation. Translates the design in [`IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md) Decisions 8–13 into ordered phases. Each phase is independently verifiable so problems surface early instead of compounding into a 4-hour debug session at 11pm Tuesday.
>
> Reference: [`ASSIGNMENT_W10A1.md`](../ASSIGNMENT_W10A1.md). Final presentation script lives at [`docs/PRESENTATION_SCRIPT.md`](./PRESENTATION_SCRIPT.md) (to be filled in during Phase 8).

---

## Pre-flight checklist (do this BEFORE Phase 0)

- [ ] W9A1 commit pushed to `main` and visible on GitHub.
- [ ] `GreenVision` v3 @ Production verified loadable via `mlflow.pytorch.load_model("models:/GreenVision/Production")`.
- [ ] `artifacts/checkpoints/class_names.json` exists with 39 entries.
- [ ] Node.js 20+ installed (`node --version`). Required for Next.js 14.
- [ ] `npm` works (`npm --version`).
- [ ] FastAPI + Uvicorn already in `requirements.txt` (verify with `pip show fastapi`).
- [ ] At least one healthy + one diseased test image saved to `data/test_demo/` for end-to-end testing.

---

## Build phases — overview

| # | Phase | Goal | New files | Time | Depends on |
|---|---|---|---|---|---|
| 0 | Design lock | Decisions 9–13 in IMPLEMENTATION_GUIDE.md; this plan committed | `ASSIGNMENT_W10A1.md`, `docs/W10_IMPLEMENTATION_PLAN.md`, IMPLEMENTATION_GUIDE.md edits | 1 h | — |
| 1 | Treatment KB | 39 entries with severity, action steps, citations | `data/treatments.json` | 3 h | 0 |
| 2 | FastAPI backend | `/health` + `/predict` working against Production model | `api/*` | 3 h | 1 |
| 3 | Next.js scaffold | Three blank routes render with nav + dark mode | `web/*` | 2 h | — (parallel to 2) |
| 4 | Upload + predict UI | Image upload → real prediction → treatment displayed | `web/app/page.tsx`, `web/components/upload-card.tsx`, `web/components/prediction-result.tsx` | 3 h | 2, 3 |
| 5 | MLflow data export | `web/public/training_data.json` baked from `./mlruns/` | `scripts/export_mlflow_for_dashboard.py` | 2 h | — (parallel to 4) |
| 6 | 3D visualizations | All four scenes interactive on `/analytics` | `web/components/viz/*` | 8 h | 5 |
| 7 | Demo orchestration + integration | `./scripts/demo.sh` works; 4 integration scenarios pass | `scripts/demo.sh`, `scripts/demo_static.sh` | 2 h | 2, 4, 6 |
| 8 | Presentation prep | 10-min script written + 2 dry runs delivered | `docs/PRESENTATION_SCRIPT.md` | 3 h | 7 |
| 9 | Final commit + submit | README updated, repo public, latest commit attributed correctly | `README.md` edits | 1 h | 8 |

**Total active development time: ~28 hours over 5 days.**

---

## Schedule against the W10D3 deadline (Wed Jun 3, 11:59pm)

| Day | Date | Phases | Hours |
|---|---|---|---|
| **Fri** | May 29 (today) | 0 + start 1 | 4 h |
| **Sat** | May 30 | Finish 1 + all of 2 + start 3 | 7 h |
| **Sun** | May 31 | Finish 3 + all of 4 + all of 5 | 7 h |
| **Mon** | Jun 1 | All of 6 (3D viz — the big day) | 8 h |
| **Tue** | Jun 2 | 7 + 8 (integration + presentation) | 5 h |
| **Wed** | Jun 3 | 9 + live presentation | 2 h + present |

**Mon is the risk day.** If 3D viz overruns past 6 PM, cut the architecture-3D scene (lowest-value of the four) and ship without it. The other three scenes (confusion matrix, per-class bars, training curves) are the rubric/Q&A value-adds.

---

## Phase 0 — Design lock

**Goal.** Lock the W10 design decisions in writing BEFORE typing any code. Same discipline as W8A1 → W9A1.

**Deliverables:**

- ✅ `ASSIGNMENT_W10A1.md` — assignment reference (created)
- This file (`docs/W10_IMPLEMENTATION_PLAN.md`)
- `IMPLEMENTATION_GUIDE.md` Decisions 9–13 written and locked

**Verification:** Read through Decisions 9–13. Each has a "Working choice. ✅ Locked." marker, a Reasoning section, and a "Where it shows up in code" sketch.

---

## Phase 1 — Treatment knowledge base

**Goal.** Build `data/treatments.json` with all 39 entries so the FastAPI backend has lookup data ready when it's built.

**Structure per entry:**

```json
{
  "<class_name_raw>": {
    "display_name": "Crop — Disease (Title Case)",
    "is_healthy": false,
    "is_background": false,
    "severity": "low" | "medium" | "high",
    "summary": "One-sentence plain-language description.",
    "time_sensitivity": "When the farmer should act.",
    "action_steps": [
      "Step 1 — what to do",
      "Step 2 — when",
      "Step 3 — how to verify it worked",
      "Step 4 — what to do next if symptoms persist"
    ],
    "sources": [
      {"name": "Cornell PMEP", "url": "https://..."},
      {"name": "Penn State Extension", "url": "https://..."}
    ]
  }
}
```

**Special cases:**

- **Healthy classes (12)** — `is_healthy: true`, `summary` = "No disease detected", `action_steps` = maintenance tips, no severity.
- **`Background_without_leaves`** — `is_background: true`, response is a user-facing rejection ("No leaf detected — please retake the photo with a single leaf in frame").
- **`Tomato___Spider_mites Two-spotted_spider_mite`** — pest, not disease. Treatment recommendations are integrated pest management (IPM) — predator mites, water sprays, miticide families.

**Source policy:**

- Cite at least **2 sources per disease** from US university extension services or CABI Crop Protection Compendium.
- Preferred sources: UC IPM, Penn State Extension, Cornell PMEP, UMN Extension, Mississippi State Extension, NC State Extension.
- Use department-level URLs when specific articles might move (e.g. `https://extension.umn.edu/diseases` instead of a deep article link).
- Use **chemical families** ("copper-based fungicide", "broad-spectrum protectant") rather than specific brand names where possible — formulation-specific recommendations vary by state regulations.

**Verification:** `python -c "import json; d=json.load(open('data/treatments.json')); print(len(d), 'entries'); [print(k) for k in sorted(set(json.load(open('artifacts/checkpoints/class_names.json'))) - set(d.keys()))]"` — should print `39 entries` and **no missing keys**.

**Risk:** Treatment text could expose the project to "wait, are you giving real medical-style advice?" criticism. Mitigation: every entry ends with the standard disclaimer "Recommendations are educational; consult your local agricultural extension agent for region-specific guidance." The dashboard also surfaces the AI disclaimer banner from Decision 12.

---

## Phase 2 — FastAPI backend

**Goal.** Stand up `/health` + `/predict` against the registered Production model.

**Files:**

```
api/
├── __init__.py
├── main.py              # FastAPI app + lifespan
├── routes/
│   ├── __init__.py
│   ├── health.py        # GET /health
│   └── predict.py       # POST /predict
├── schemas.py           # Pydantic request/response models
├── inference.py         # load_model, preprocess, run_inference, postprocess
└── treatments.py        # lookup wrapper
```

**Implementation order within Phase 2:**

1. `schemas.py` — Pydantic models for `HealthResponse`, `PredictionResponse`, `TopKPrediction`. Mirror the spec in IMPLEMENTATION_GUIDE Decision 8.
2. `inference.py` — load model from MLflow Registry via lifespan, expose `predict(pil_image) -> PredictionResponse`. Reuse `eval_tfms` from `src/greenvision/data/transforms.py` — DO NOT re-implement.
3. `treatments.py` — load `data/treatments.json` once at startup, expose `get_treatment(class_name) -> Treatment`.
4. `routes/health.py` — returns `{ "status": "ok", "model_loaded": True, "num_classes": 39, "model_version": "GreenVision/Production" }`.
5. `routes/predict.py` — multipart file → PIL Image → blur/size checks → inference → treatment lookup → JSON response.
6. `main.py` — wire everything together. CORS middleware for `http://localhost:3000`.

**Verification:**

```bash
# Terminal 1
PYTHONPATH=src .venv/bin/uvicorn api.main:app --reload --port 8000

# Terminal 2
curl http://localhost:8000/health
# → {"status":"ok","model_loaded":true,...}

curl -F "file=@data/test_demo/healthy_apple.jpg" http://localhost:8000/predict | jq .
# → {"label":"Apple___healthy","display_name":"Apple (healthy)","confidence":0.99,...}
```

---

## Phase 3 — Next.js scaffold

**Goal.** A blank Next.js app on `:3000` with three routes, dark mode, and the shadcn/ui foundation installed.

**Implementation order:**

1. `npx create-next-app@latest web --typescript --tailwind --app --no-src-dir --import-alias "@/*"`
2. `cd web && npx shadcn@latest init` (pick "New York" style, slate base color)
3. Install shadcn primitives we'll need: `npx shadcn@latest add button card badge alert separator tabs progress`
4. Install React Three Fiber stack: `npm install three @react-three/fiber @react-three/drei`
5. Install dark mode: `npm install next-themes`
6. Build `app/layout.tsx` — `ThemeProvider`, top nav with three routes + theme toggle
7. Build three blank routes: `app/page.tsx`, `app/analytics/page.tsx`, `app/about/page.tsx` — each a `<Card>` saying "TODO"
8. Configure `next.config.js` with API rewrite: `/api/:path*` → `http://localhost:8000/:path*` (avoids CORS during dev)

**Verification:** `npm run dev` → three routes render with working dark-mode toggle and nav.

---

## Phase 4 — Upload + predict UI

**Goal.** A user can upload a leaf JPG, see a preview, click Diagnose, get a real prediction back from FastAPI, and see disease + confidence + treatment.

**Components:**

- `UploadCard` — drag-and-drop or file picker, preview, "Diagnose" button (disabled until file selected)
- `PredictionResult` — receives `PredictionResponse`, renders disease display name (formatted via `lib/format-class-name.ts`), confidence badge, treatment steps, sources, AI disclaimer
- `ConfidenceBadge` — colored chip: high (≥0.7, emerald), medium (0.4–0.7, amber), low (<0.4, rose)
- `AIDisclaimer` — bottom of result card. Always present. Stronger language when confidence < 0.7

**State flow:**

```
UploadCard (local state)
  ↓ selectedFile, previewURL
  ↓ onClick "Diagnose" → fetch('/api/predict', { method: 'POST', body: FormData })
  ↓ setPrediction(response)
PredictionResult (driven by prediction state)
```

**Verification:**

- Upload a healthy leaf JPG from `data/test_demo/` → high-confidence prediction renders within 2s
- Stop FastAPI (Ctrl+C in terminal 1) → upload now shows "API not reachable — please ensure the backend is running on port 8000" instead of crashing
- Upload same image twice → identical prediction JSON

---

## Phase 5 — MLflow data export

**Goal.** Bake all the training data the dashboard needs into a static `web/public/training_data.json` at build time. The presentation never queries MLflow live — that's a stability risk we don't need.

**Exported data structure:**

```json
{
  "experiment": "greenvision",
  "best_run": {
    "attempt_id": "002",
    "phase2_run_id": "6fcedd73ad6a4bd48b11154ff57cee7f",
    "best_val_acc": 0.9973,
    "best_epoch_global": 18,
    "total_epochs": 24,
    "training_time_seconds": <from MLflow>
  },
  "epoch_metrics": [
    {"epoch": 0, "phase": "phase1", "train_loss": ..., "train_acc": ..., "val_loss": ..., "val_acc": ...},
    ...
  ],
  "per_class_metrics": [
    {"class_index": 0, "class_name": "Apple___Apple_scab", "precision": ..., "recall": ..., "f1": ..., "support": ...},
    ...
  ],
  "confusion_matrix": [[...], [...], ...]  // 39×39 int matrix
}
```

**Script:** `scripts/export_mlflow_for_dashboard.py` reads from `./mlruns/` using the `MlflowClient`, walks the parent + phase1 + phase2 children, pulls metric history, regenerates the per-class report and confusion matrix from the saved test predictions, writes the JSON.

**Verification:**

```bash
python scripts/export_mlflow_for_dashboard.py
jq '.epoch_metrics | length' web/public/training_data.json   # → 24
jq '.per_class_metrics | length' web/public/training_data.json   # → 39
jq '.confusion_matrix | length' web/public/training_data.json    # → 39
```

---

## Phase 6 — 3D visualizations (the big day)

**Goal.** Four 3D scenes on `/analytics`, each interactive (mouse-drag rotation, hover tooltips where it makes sense).

**Order — easiest to hardest:**

### Scene 1 — Per-class 3D bar chart (2 h)

- 39 bars arranged in a 7×6 grid (one empty slot)
- Bar height = recall, color = precision (gradient from rose @ 0.0 to emerald @ 1.0)
- Click a bar → shows tooltip with class name + metrics
- Camera: `OrbitControls`, initial angle ~30° elevation

### Scene 2 — 3D confusion matrix (2.5 h)

- 39×39 grid where each cell is a 3D cube
- Cube height = log(confusion_count + 1) to compress the diagonal
- Diagonal cubes (correct predictions) tinted emerald; off-diagonal tinted rose
- Click a cube → tooltip: "Predicted X, actually Y — 12 instances"
- Camera: top-down 30° pitch initially, rotatable

### Scene 3 — Training curves in 3D (1.5 h)

- Two curves (train loss + val loss) plotted as 3D ribbons through space
- Third axis = epoch, vertical axis = loss/acc, depth axis = train vs val
- Phase boundary (epoch 3) marked with a colored plane
- Hover an epoch → tooltip with all four metrics

### Scene 4 — Animated 3D architecture diagram (2 h)

- EfficientNet-B0 represented as 9 stacked blocks (`features[0]` through `features[8]` + classifier)
- A glowing "data particle" travels through the blocks when "Play Inference" is clicked
- Blocks change color as the particle enters them (cool blue → warm orange to suggest activation)
- Classifier block has 39 dots (one per class); the predicted-class dot lights up when the particle arrives

**Cut-line:** If we're past 6 PM Monday and Scene 4 isn't working, ship without it. The other three carry the presentation.

**Verification:** Open `/analytics`, all scenes render at 30+ FPS. No console errors.

---

## Phase 7 — Demo orchestration + integration testing

**Goal.** One command starts the whole demo environment. Four integration scenarios pass.

**`scripts/demo.sh`:**

```bash
#!/usr/bin/env bash
# Starts FastAPI, Next.js, and MLflow UI together for the W10D3 demo.
# Usage: ./scripts/demo.sh
set -e
cd "$(dirname "$0")/.."

# Use concurrently from npm so all three start in one terminal
cd web && npm install --silent && cd ..
npx concurrently \
  --names "API,UI,MLflow" \
  --prefix-colors "cyan,magenta,yellow" \
  "PYTHONPATH=src .venv/bin/uvicorn api.main:app --port 8000" \
  "cd web && npm run dev" \
  "mlflow ui --backend-store-uri file:./mlruns --port 5001"
```

**`scripts/demo_static.sh`** — backup that runs only Next.js with a `NEXT_PUBLIC_DEMO_MODE=static` flag that intercepts `/api/predict` calls and returns cached responses. Insurance against a backend crash on stage.

**Integration scenarios (from the assignment):**

1. ✅ Upload a healthy leaf → predicts healthy with high confidence
2. ✅ Upload a diseased leaf → correct disease + treatment displayed
3. ✅ Stop FastAPI → dashboard shows informative error, doesn't crash
4. ✅ Upload same image twice → identical predictions (deterministic — `model.eval()` + same seed isn't a thing at inference but determinism falls out of `eval_tfms` being non-random)

---

## Phase 8 — Presentation prep

**Goal.** A 10-minute talk delivered twice in dry runs before the real thing.

**Script structure (`docs/PRESENTATION_SCRIPT.md`):**

| Time | Segment | What's on screen |
|---|---|---|
| 0:00–1:00 | Problem + stakeholder | Slide 1 (or "Home" route hero text) — farmer photo + the elevator pitch |
| 1:00–2:30 | Architecture | Repo tree screenshot or `/about` route — name each piece |
| 2:30–4:30 | Training results | `/analytics` route — call out 99.73%, walk through per-class bars |
| 4:30–7:00 | Live demo | `/` route — upload healthy + diseased + a non-leaf image |
| 7:00–8:30 | Edge case + reflection | Confidence < 0.4 scenario — "what we'd do differently" |
| 8:30–10:00 | Buffer + transition to Q&A | Architecture-3D scene playing in background as a visual |

**Q&A defense doc** — one paragraph per likely question (10 candidates from the master plan above).

**Dry runs:** Tuesday afternoon (record yourself) + Tuesday evening (with someone who'll push back). Adjust pacing.

---

## Phase 9 — Final polish + submit

- Update `README.md` with W10A1 deliverables table (parallel to W8A1 + W9A1 tables)
- Add `web/` setup instructions to README
- Commit attribution check (`qevans@student.neumont.edu`)
- `git push origin main`
- Verify on github.com
- Submit URL to Canvas: `https://github.com/QuinnEvans34/AML-GreenVision`

---

## Risk register

| Risk | Mitigation |
|---|---|
| 3D viz overrun | Cut architecture-3D scene; the other three are sufficient |
| `models:/GreenVision/Production` doesn't load in FastAPI | Workaround: load directly from `mlruns/models/GreenVision/version-3/` filesystem path; gives us a Plan B path if the registry URI mechanism fails |
| Next.js + FastAPI port collision on stage | Both ports (3000, 8000) explicitly bound in scripts; pre-flight check in `demo.sh` that the ports are free |
| MPS inference behaves differently than CPU | Test on CPU device too; if MPS gives weird results, fall back to CPU for the demo |
| Treatment text gets challenged in Q&A | Every entry has cited sources you can read aloud; the AI disclaimer at the bottom of every prediction is the explicit user-facing guardrail |
| Background_without_leaves not visually obvious as the "model rejects this" path | Dashboard has dedicated UI state for this class — different card layout, no treatment shown, prominent retake CTA |
| Network in classroom is bad | Localhost-only demo, no cloud dependencies, no network needed past `npm install` which is already done |
