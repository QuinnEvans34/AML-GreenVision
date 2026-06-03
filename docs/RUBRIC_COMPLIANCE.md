# W10P1 Rubric Compliance Audit

> Grader-facing single-page audit. Every requirement from the W10P1 assignment
> rubric mapped to the file or dashboard view that satisfies it. Click any
> file path in the GitHub view to inspect the implementation.

**Full assignment text:** [`ASSIGNMENT_W10A1.md`](../ASSIGNMENT_W10A1.md)
**Presentation runbook:** [`docs/PRESENTATION_SCRIPT.md`](./PRESENTATION_SCRIPT.md)
**Speaker crib sheet:** [`docs/SPEAKER_NOTES.md`](./SPEAKER_NOTES.md)

---

## Part 1 — FastAPI inference endpoint

| Rubric requirement | Where satisfied | Status |
|---|---|---|
| `GET /health` confirms API reachable + model loaded | [`api/routes/health.py`](../api/routes/health.py) | ✅ |
| `POST /predict` accepts multipart upload, returns prediction | [`api/routes/predict.py`](../api/routes/predict.py) | ✅ |
| Load model from `models:/GreenVision/Production` | [`api/inference.py:load_production_model()`](../api/inference.py) | ✅ |
| Load class names from the MLflow artifact training saved | [`api/inference.py:load_class_names()`](../api/inference.py) reads `artifacts/checkpoints/class_names.json` (the same artifact `scripts/train.py` logs to the phase2 child run) | ✅ |
| Preprocess uploaded images with the same transforms as validation | [`api/inference.py`](../api/inference.py) does `from greenvision.data.transforms import eval_tfms` — **same Python object** training uses | ✅ |
| Response includes disease name, confidence, healthy?, treatment | [`api/schemas.py:PredictionResponse`](../api/schemas.py) — fields: `class_name`, `display_name`, `confidence`, `is_healthy`, `is_background`, `top_k`, `treatment`, `warnings`, `model_version`, `inference_time_ms` | ✅ |

### Part 1 — Design decisions (documented before building)

| Design question | Decision | Documented in |
|---|---|---|
| How do you handle non-image uploads? Corrupted files? | Size check 413 → empty 400 → PIL decode 400 with clear message → no model call. Dashboard surfaces the error without crashing. | [`IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md) Decision 8 + dashboard `/about` → Design decisions |
| What do you return when confidence is very low (<40%)? | Standard 200 response with a `warnings` array containing the low-confidence message. Dashboard hides treatment behind "Show anyway" disclosure and strengthens the AI disclaimer. | [`IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md) Decision 10 + dashboard `/about` → Design decisions |
| Where does treatment recommendation text come from? | Static knowledge base at [`data/treatments.json`](../data/treatments.json) — 39 entries with severity, action steps, time sensitivity, and ≥2 citations to US extension publications. | [`IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md) Decision 9 + dashboard `/about` → Design decisions |

---

## Part 2 — Dashboard

| Rubric requirement | Where satisfied | Status |
|---|---|---|
| Image upload with preview | [`web/components/upload-card.tsx`](../web/components/upload-card.tsx) — drag-drop + click + test-set picker, preview rendered in shared preview area | ✅ |
| Health check on load + informative error if FastAPI down | [`web/app/page.tsx`](../web/app/page.tsx) — mount-time `health()` call, surfaces `Alert` with restart command if API unreachable | ✅ |
| Display disease name (formatted, not raw) | [`web/lib/format-class-name.ts`](../web/lib/format-class-name.ts) + `display_name` field in treatments KB | ✅ |
| Display confidence score | [`web/components/confidence-badge.tsx`](../web/components/confidence-badge.tsx) + hero number in [`prediction-result.tsx`](../web/components/prediction-result.tsx) | ✅ |
| Display treatment recommendation | [`web/components/prediction-result.tsx`](../web/components/prediction-result.tsx) — action steps + time-sensitivity callout + cited sources | ✅ |
| Visual indication of confidence level (high/medium/low) | 4-tier `ConfidenceBadge` + tone-coded card accent strip + severity chip | ✅ |
| AI disclaimer for critical agricultural decisions | [`web/components/ai-disclaimer.tsx`](../web/components/ai-disclaimer.tsx) — always present, strengthened at low confidence | ✅ |

### Part 2 — Design decisions

| Design question | Decision | Documented in |
|---|---|---|
| How do you format class names for display? | KB-provided `display_name` per class + frontend fallback formatter. Examples in dashboard `/about` → Design decisions. | [`IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md) Decision 10 + [`web/lib/format-class-name.ts`](../web/lib/format-class-name.ts) |
| How do you visually communicate low confidence to a farmer? | 4-band color system + behavioral degradation (treatment gating at <40%) + strengthened disclaimer copy. | [`IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md) Decision 10 + dashboard `/about` → Design decisions |
| What context makes a prediction actionable for your stakeholder? | Summary + severity tier + time-sensitivity window + numbered action steps + cited extension sources + Top-3 alternatives. | [`IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md) Decisions 9 & 10 + dashboard `/about` → Design decisions |

---

## Part 3 — End-to-end integration

### Test scenarios

| Scenario | How we verify | Where |
|---|---|---|
| Upload a healthy leaf → predicts healthy with high confidence | Test-set picker → pick any `___healthy` thumbnail → predict | Dashboard `/` |
| Upload a diseased leaf → correct disease + treatment | Test-set picker → pick a known disease class | Dashboard `/` |
| Stop FastAPI manually → dashboard informative error, no crash | Ctrl+C the API → dashboard shows "Backend not reachable" alert with restart command | Dashboard `/` mount-time health check |
| Upload the same image twice → identical predictions | Deterministic `eval_tfms` + `model.eval()` + `torch.no_grad()` — provable by uploading any thumbnail twice | Dashboard `/` |

### Integration checklist

| Item | Where satisfied | Status |
|---|---|---|
| Preprocessing in serve matches validation transforms exactly | API imports `eval_tfms` from `greenvision.data.transforms` — **same Python object reference** | ✅ |
| Class names in serve loaded from MLflow artifact in correct order | [`api/inference.py:load_class_names()`](../api/inference.py) reads `artifacts/checkpoints/class_names.json` — the artifact `scripts/train.py` logs to the phase2 run | ✅ |
| Model in `.eval()` mode before inference | [`api/inference.py:load_production_model()`](../api/inference.py) calls `.eval()` after `to(device)` | ✅ |
| FastAPI and dashboard can run simultaneously on different ports | uvicorn :8000 + next dev :3000 + Next.js rewrite proxies `/api/*` → `:8000` so same-origin in browser. Orchestrated by [`scripts/demo.sh`](../scripts/demo.sh). The rubric mentions Streamlit; we use Next.js — Part 2 only requires "a user interface" (see [`IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md) Decision 12). | ✅ |

---

## Part 4 — Final presentation segments

| Segment | Where in dashboard | Talk to it from |
|---|---|---|
| Problem + stakeholder | Tab 1 (`/`) hero text | [`SPEAKER_NOTES.md`](./SPEAKER_NOTES.md) §0:00–1:00 |
| Architecture | Tab 2 (`/analytics`) → **Architecture** → **System pipeline** (full project) → optional **Model** dive | [`SPEAKER_NOTES.md`](./SPEAKER_NOTES.md) §1:00–3:00 |
| Training results | Tab 2 → **Training** tab — KPI strip + curves; explicit "Source: MLflow attempt_002" tag under KPI strip | [`SPEAKER_NOTES.md`](./SPEAKER_NOTES.md) §3:00–4:30 |
| Live demo | Tab 1 → Test-set picker → 3 thumbnails (healthy, diseased, background) | [`SPEAKER_NOTES.md`](./SPEAKER_NOTES.md) §5:30–7:30 |
| Reflection | Verbal — supported by `/analytics` Per-class tab if needed | [`SPEAKER_NOTES.md`](./SPEAKER_NOTES.md) §7:30–9:00 |
| MLflow screenshot equivalent | KPI strip on `/analytics` + source attribution line — same data MLflow logged | (no separate file needed) |

---

## Deliverables (final submission)

| Deliverable | Where | Status |
|---|---|---|
| Complete repository | [github.com/QuinnEvans34/AML-GreenVision](https://github.com/QuinnEvans34/AML-GreenVision) | ✅ |
| Serving layer | [`api/`](../api/) + [`web/`](../web/) | ✅ |
| Model in Production | `models:/GreenVision/Production` v3 (verified loadable by `scripts/demo.sh` pre-flight) | ✅ |
| Final implementation guide | [`IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md) — 13 design decisions locked, Resolved/Open sections | ✅ |
| Training report | [`docs/TRAINING_REPORT.md`](./TRAINING_REPORT.md) | ✅ |
| End-to-end demo | `./scripts/demo.sh` → upload → prediction → treatment | ✅ |
| Presentation (W10D3) | [`docs/PRESENTATION_SCRIPT.md`](./PRESENTATION_SCRIPT.md) + [`docs/SPEAKER_NOTES.md`](./SPEAKER_NOTES.md) | ✅ |

---

## Q&A defense — 12 anticipated questions with prepared answers

All 5 sample questions from the rubric + 7 additional likely questions, each
answered in one paragraph with talking-point structure. See
[`docs/PRESENTATION_SCRIPT.md`](./PRESENTATION_SCRIPT.md) Q&A defense section.

| # | Question | Section in script |
|---|---|---|
| 1 | Why two-phase fine-tuning vs full unfreeze? | Q1 |
| 2 | What does the 0.27% wrong look like? | Q2 (sample) |
| 3 | Walk through Diagnose → result | Q3 (sample) |
| 4 | Why ImageNet normalization? | Q4 (sample) |
| 5 | Near-tied top-2 predictions? | Q5 (sample) |
| 6 | Why AdamW vs Adam? | Q6 — answered via Optimizer tab |
| 7 | Why Next.js over Streamlit? | Q7 |
| 8 | Where do treatments come from? | Q8 |
| 9 | What if FastAPI crashes mid-demo? | Q9 |
| 10 | PlantVillage → field gap? | Q10 |
| 11 | Why `eval()` AND `torch.no_grad()`? | Q11 |
| 12 | How do you verify identical predictions for the same image? | Q12 |

---

**Submission status: ready.**
