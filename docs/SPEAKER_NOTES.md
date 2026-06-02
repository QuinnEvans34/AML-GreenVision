# Speaker notes · W10P1 presentation

> Tight crib sheet for the actual presentation. Use this on a phone or
> second monitor at podium level. Don't read aloud — glance, then look up
> and talk. Full script: [`PRESENTATION_SCRIPT.md`](./PRESENTATION_SCRIPT.md).

**One window, two tabs, light mode, 125% zoom.**

```
Tab 1: localhost:3000           (Diagnose)
Tab 2: localhost:3000/analytics (5 viz tabs)
```

---

## 0:00–1:00 · Problem + stakeholder · **Tab 1**

**Click:** none — stay on the hero.

**Say (3 beats):**
- Farmer · 24-hour decision window · late blight vs early blight
- Wrong fungicide costs money + damages plant
- Confident-wrong is worse than honestly-uncertain

**Land:** "The product is: a phone photo of a leaf → disease + confidence + treatment cited to extension publications."

**Rubric:** *Problem + stakeholder*

---

## 1:00–3:00 · Architecture · **Tab 2 → Architecture → System pipeline**

**Click:** System pipeline sub-tab (default).

**Say (point at each row):**
- **Phase 1 — training pipeline:** PlantVillage 54,306 × 39 classes → 80/10/10 stratified split (seed 42) → ImageNet normalization → EfficientNet-B0 + 39-class head → AdamW + MLflow logging
- **Phase 2 — registry:** Best model logged via `mlflow.pytorch.log_model` → `scripts/promote.py` transitions to Production stage → URI is `models:/GreenVision/Production`
- **Phase 3 — serving:** dashboard POSTs multipart upload → FastAPI loads model from Registry at startup → `eval_tfms` is imported (no drift possible) → treatment looked up in `data/treatments.json` → typed JSON response → dashboard renders tone-coded result

**Land:** "Every box names the file. Every Rubric tag points at a specific assignment requirement."

**Optional dive:** click **Model** sub-tab → Play inference for 5 seconds → back to System.

**Rubric:** *Architecture* (full project, not just neural net)

---

## 3:00–4:30 · Training results · **Tab 2 → Training**

**Click:** Training tab.

**Say:**
- KPI strip: **99.73% val · 99.73% test · 0.000% gap · baseline 2.56%**
- Source line: "These come from MLflow run attempt_002, tracking URI `file:./mlruns`"
- Curves: Phase 1 shaded green (head warmup, 3 epochs), Phase 2 unfreezing (21 epochs)
- Loss panel: log-shape descent — characteristic of fine-tuning

**Land:** "Zero gap between val and test means we're not overfitting to the validation set."

**Rubric:** *Training results · final val accuracy · strategy · baseline comparison · MLflow screenshot equivalent*

---

## 4:30–5:30 · Optimizer story · **Tab 2 → Optimizer · click Run optimizers**

**Click:** Optimizer tab → "Run optimizers" button.

**Say (as lines draw):**
- Same start, same surface, three optimizers
- **SGD (rose):** raw gradient, gets pushed around by bumps
- **Adam (blue):** adaptive per-axis, weight decay couples to gradient
- **AdamW (emerald):** decouples weight decay → applies directly to params
- "Trajectories are real — JS running the actual update equations on the surface"

**Land:** "AdamW is what trained GreenVision. The optimizer choice is defensible from this visualization."

**Rubric:** *Architecture + Q&A defense for "why AdamW"*

---

## 5:30–7:30 · Live demo · **Tab 1 (back to Diagnose)**

**Click sequence — three thumbnails:**

1. **Healthy leaf** (Apple healthy or Tomato healthy):
   - "Pick from held-out test set. Model has never seen this leaf."
   - Result: emerald accent, maintenance tips, ~99% confidence, 3 sources
   - "Inference ~140 ms on this MacBook, end to end"

2. **Diseased leaf** (Tomato Late blight or Potato Late blight):
   - "Held-out diseased leaf."
   - Result: rose accent strip, big confidence number in rose
   - "Severity chip says high. Time-sensitivity callout says 'act within 24 hours' — actionable language"
   - Click one source link: "Real Cornell extension publication"

3. **Background** (the Background tile):
   - "Non-leaf input. The model could have confidently misclassified this."
   - Result: amber accent, no treatment, retake guidance
   - "39th class — `Background_without_leaves` — is what graceful failure looks like"

**Rubric:** *Live demo · upload → diagnosis → treatment · edge case handled*

---

## 7:30–9:00 · Reflection · **Stay on result or switch to Per-class tab**

**Say (two reflections):**

**Design that changed:**
- W8A1 design: `mlflow.log_artifact("best.pt")` + API does `torch.load`
- W9A1 required: MLflow Model Registry approach
- Changed before training, IMPLEMENTATION_GUIDE Decisions 7+8 reflect the pivot
- Cleaner deployment story

**What I'd improve:**
- PlantVillage is studio shots, 99.73% is in-distribution
- Field photos drop 30–50 pp in the literature
- Next iteration: field-photo augmentation + two-stage system (in-distribution detector → classifier)
- Dashboard already guides users to plain-background photos — UI affordance, not a fix

**Rubric:** *Reflection · one design decision changed · one thing to improve*

---

## 9:00–10:00 · Transition

"Architecture, results, demo, reflection. Questions?"

---

## Q&A defense — speed-cards (300 words total)

> Glance at the question, look up, talk. For full paragraphs see
> [`PRESENTATION_SCRIPT.md`](./PRESENTATION_SCRIPT.md) Q&A section.

**Why two-phase fine-tuning?**
Catastrophic forgetting · random head's gradients would trash pretrained features · Phase 1 protects backbone · Phase 2 lets backbone adapt at 10× smaller LR.

**What does the 0.27% wrong look like?**
[Per-class or Confusion tab] 15/5545 misclassifications, mostly tomato-late vs tomato-early — visually similar pathologies · no cross-crop errors.

**Walk through Diagnose → result.**
Browser FormData POST → Next rewrite → FastAPI body read → size check → PIL decode → blur check → `eval_tfms` → `to(MPS)` → forward under `no_grad()` → softmax + argsort top-3 → class_names lookup → treatments lookup → Pydantic JSON → dashboard renders → 140ms.

**Why ImageNet normalization?**
EfficientNet's first conv layer is calibrated for that distribution · changing values silently degrades predictions · API imports the same `eval_tfms` object so drift impossible.

**Near-tied top-2 confidence?**
4-band degradation: ≥85 emerald · 70–85 muted · 40–70 amber warning · <40 rose, treatment hidden behind "Show anyway", primary CTA becomes Retake, disclaimer strengthens. Top-3 always shown.

**Why AdamW over Adam?**
[Optimizer tab] Adam folds L2 decay into gradient then divides by √v̂ — breaks L2 behavior · AdamW applies decay directly to params · 1–2 pp generalization gain.

**Why Next.js over Streamlit?**
Part 2 says "user interface" — Streamlit mentioned only in Part 3 example · 3D viz suite needed React Three Fiber · polish matters for non-technical stakeholder · Decision 12 in IMPLEMENTATION_GUIDE.

**Treatment source?**
[Click source link] 39 entries in `data/treatments.json` · ≥2 citations per disease to US extension publications (Cornell, Penn State, UMN, UC IPM) · chemical families not brands (EPA regs change) · AI disclaimer points to local agent.

**FastAPI crashes?**
`scripts/demo_static.sh` runs UI only · Analytics + About still work · Diagnose surfaces the API-down alert (which is itself a required integration scenario).

**Field photos?**
No, not at 99% — PlantVillage is studio · field photos drop 30–50 pp in literature · dashboard guides toward plain-background single-leaf photos · long-term: augmentation + OOD detector.

**`eval()` AND `no_grad()`?**
Different things both needed: `eval()` flips dropout off + BN to stored stats · `no_grad()` skips autograd graph to save memory + speed · both wrong = wrong predictions, neither = working but inefficient.

**Identical predictions for the same image?**
Determinism is structural: non-stochastic `eval_tfms` + `model.eval()` disables dropout/BN sampling + `no_grad()` doesn't add randomness + MPS forward pass deterministic. Click the same thumbnail twice, audience sees identical confidence.

---

## Emergency exits

| If… | Do… |
|---|---|
| FastAPI dies | `./scripts/demo_static.sh` in a fresh terminal |
| Ports locked | `./scripts/demo_reset.sh` then `./scripts/demo.sh` |
| 3D scene black | Refresh that tab — `demo.sh` keeps running |
| Architecture animation fails | System pipeline is the primary architecture answer |
| Browser hangs | Cmd+R the tab — state is recoverable |

**Last line:** if something is truly broken, say *"the dashboard is set up to handle the API being unavailable — let me show you that scenario."* Switch to `demo_static.sh`, demonstrate the alert, then walk through Analytics. That's still ~80% of the rubric content.
