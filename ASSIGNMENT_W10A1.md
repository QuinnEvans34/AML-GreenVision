# [W10A1] GreenVision Final Submission & Presentation

**Due:** Wednesday by 11:59pm
**Points:** 50
**Submitting:** a website URL
**Available until:** Jun 3 at 11:59pm

---

## Overview

Build the serving layer for GreenVision (FastAPI endpoint + Dashboard), integrate it with your trained model from Assignment 9, and present the complete end-to-end system to the class.

> A farmer should be able to upload a leaf photo and receive: disease name, confidence score, and treatment recommendation. **That's the product.**

---

## Instructions

### Part 1 — FastAPI inference endpoint

Build an API that accepts image uploads and returns disease predictions.

**Requirements:**

- `GET /health` endpoint — confirms the API is reachable and model is loaded
- `POST /predict` endpoint — accepts image file upload (multipart/form-data), returns prediction
- Load model from MLflow Registry: `models:/GreenVision/Production`
- Load class names from the same MLflow artifact that training saved
- Preprocess uploaded images with the **same transforms used for validation** (deterministic, no augmentation)
- Response includes: disease name, confidence (0–1), whether leaf is healthy, treatment recommendation

**Your design decisions:**

- How do you handle non-image uploads? Corrupted files?
- What do you return when confidence is very low (<40%)?
- Where does treatment recommendation text come from?

**Document these in your implementation guide before building.**

---

### Part 2 — Dashboard

Build a user interface that calls your FastAPI endpoint and displays results for a non-technical stakeholder.

**Requirements:**

- Image upload with preview
- Health check on load — informative error if FastAPI is not running
- Display: disease name (formatted for readability, not raw class string), confidence score, treatment recommendation
- Visual indication of confidence level (high/medium/low)
- AI disclaimer for critical agricultural decisions

**Your design decisions:**

- How do you format class names for display? (e.g., `Tomato___Late_blight` → "Tomato — Late blight")
- How do you visually communicate low confidence to a farmer?
- What context makes a prediction actionable for your stakeholder?

> **Note on the dashboard stack.** The assignment references Streamlit in the Part 3 integration checklist, but Part 2 only requires "a user interface that calls your FastAPI endpoint." For GreenVision we're building this as a **Next.js (React) application** instead of Streamlit — see `IMPLEMENTATION_GUIDE.md` Decision 12 for the architecture rationale.

---

### Part 3 — End-to-end integration

Verify the complete system works from upload to diagnosis.

**Test scenarios:**

- Upload a healthy leaf image from your validation set → should predict healthy with high confidence
- Upload a diseased leaf image → should predict correct disease with treatment
- Stop FastAPI manually → dashboard should show informative error, not crash
- Upload the same image twice → should get identical predictions (deterministic inference)

**Integration checklist:**

- Preprocessing in `serve.py` matches validation transforms exactly
- Class names in `serve.py` loaded from MLflow artifact in correct order
- Model in `eval()` mode before inference
- FastAPI and the dashboard can run simultaneously on different ports

---

### Part 4 — Final presentation (W10D3, 10 min + 2 min Q&A)

Present the complete GreenVision system to the class.

**Structure:**

| Segment | Content |
|---|---|
| **Problem + stakeholder** | Who uses GreenVision and what decision they make with it |
| **Architecture** | What pieces exist — model, data pipeline, serving layer — show your repo structure or diagram |
| **Training results** | Final val accuracy, strategy used, comparison to naive baseline, MLflow screenshot |
| **Live demo** | Upload a test image → show diagnosis + treatment. Handle one edge case (low confidence or wrong file type). |
| **Reflection** | One design decision that changed during implementation, one thing you'd improve with more time |

**Q&A:** Instructor and peers ask about your design choices. **Defend your reasoning** — "the tutorial / AI said so" is not reasoning.

**Sample Q&A questions:**

- "Why did you choose [your fine-tuning strategy] instead of [alternative]?"
- "Your model is 89% accurate. What does the 11% it gets wrong look like?"
- "Walk me through what happens from clicking 'Diagnose' to seeing the result."
- "Why ImageNet normalization? What breaks if you use different values?"
- "How do you handle a prediction where the top two classes have nearly equal confidence?"

---

## Learning Outcomes

- Build and evaluate end-to-end machine learning pipelines that transform raw data into deployed, business-facing predictions.
- Apply supervised learning algorithms, feature engineering techniques, and model evaluation methods to real-world datasets.
- Design and automate production ML workflows that ingest, validate, transform, and retrain models.
- Communicate model results, technical decisions, and system limitations in plain language to non-technical stakeholders.
- Assess the operational risks of deployed ML systems and apply appropriate monitoring strategies.

---

## Deliverables

Submit your final `aml-greenvision` repository and present in class.

| Item | Requirement |
|---|---|
| **Complete repository** | All code for data loading, training, serving, and dashboard |
| **Serving layer** | FastAPI with `/health` and `/predict` functional, Next.js dashboard connected |
| **Model in Production** | GreenVision model promoted to Production stage in MLflow |
| **Final implementation guide** | All decisions documented, updated through implementation, includes final architecture diagram or description |
| **Training report** | From Assignment 9, updated with any final adjustments |
| **End-to-end demo** | Working system: upload → prediction → treatment display |
| **Presentation** | Delivered in class W10D3 (10 min + 2 min Q&A) |

---

## GreenVision-specific deliverable additions (beyond the baseline)

These aren't in the rubric but are non-negotiable for our "go big or go home" execution:

| Item | Why we're building it |
|---|---|
| **Treatment knowledge base** (`data/treatments.json`) | 39 entries with summary, action steps, severity, and cited sources. The "treatment recommendation" requirement is the wildcard — a curated KB is the difference between credible advice and demo-grade filler. |
| **3D confusion matrix viz** | Live, interactive 39×39 height-field rendered in the dashboard. Lets the presenter point at specific class confusions during Q&A instead of describing them. |
| **3D per-class accuracy bar chart** | 39 vertical bars, height = recall, color = precision. Rotatable. |
| **3D training-curves scene** | Loss + accuracy plotted with epoch as the third axis — for the "training results" segment. |
| **Animated 3D architecture diagram** | EfficientNet-B0 layers as 3D blocks with animated data flow during the "walk me through inference" Q&A answer. |
| **One-command demo orchestration** | `./scripts/demo.sh` starts FastAPI + Next.js + MLflow UI together so the presentation is one terminal command. |
