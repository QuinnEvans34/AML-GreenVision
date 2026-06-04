# GreenVision · W10P1 (W10D3) Presentation Script

> **10 minutes + 2 minutes Q&A.** Single browser, two tabs. No MLflow UI, no
> terminal, no slides. Everything the audience sees comes from the dashboard.

This script is the runbook. For the tight, glance-down crib sheet during
the talk, see [`docs/SPEAKER_NOTES.md`](./SPEAKER_NOTES.md).
For the grader-facing audit, see [`docs/RUBRIC_COMPLIANCE.md`](./RUBRIC_COMPLIANCE.md).

---

## Pre-demo checklist (do this 10 minutes before)

In one terminal at the repo root:

```bash
./scripts/demo.sh
```

Wait for:

```
✓ Ports 8000, 3000, 5001 are free
✓ models:/GreenVision/Production loads cleanly
✓ web/node_modules is present
✓ web/public/training_data.json is present
[API]    INFO:     Uvicorn running on http://0.0.0.0:8000
[UI]     ▲ Next.js 16.2.6
```

Then open **ONE browser window** with **TWO tabs** in this order:

1. **Tab 1 — http://localhost:3000** — Diagnose (the live demo surface)
2. **Tab 2 — http://localhost:3000/analytics** — Analytics with five tabs

> **Important:** the MLflow UI is running on :5001 but **you will not open
> it**. The dashboard is the single source of truth — KPI strip, source
> attribution, and architecture diagram all reference MLflow without
> needing the UI.

Last-minute setup:

- Browser zoom **125%** (auditorium projection)
- Force **light mode** (better contrast under projector lighting; toggle is
  top-right)
- Open `data/test_demo/` or a local folder with 1 known-healthy and 1
  known-diseased image as backup if the test-set picker tab fails
- Have `./scripts/demo_reset.sh` and `./scripts/demo_static.sh` open in a
  spare terminal as emergency exits

---

## 10-minute beat sheet (dashboard-only)

### 0:00 – 1:00 — Problem + stakeholder (Rubric: Problem + stakeholder)

**On screen:** Tab 1 (`/`), top of page only — the hero with the
"Diagnose a leaf" headline and the explicit subtitle that names the
FastAPI endpoint.

**Script (≈60s):**
> "If you're a smallholder farmer and your tomato leaves are spotting,
> you have a 24-hour decision window. Late blight will take your crop in
> a week. Early blight is annoying but you have time. The two diseases
> look almost identical to an untrained eye, and the wrong fungicide
> costs you money and damages the plant.
>
> GreenVision is the front door: a farmer takes a phone photo of a single
> leaf, uploads it through this dashboard, the dashboard POSTs to a
> FastAPI endpoint that loads the trained model from the MLflow Registry,
> and the response is a disease name plus a confidence score plus a
> treatment recommendation cited to a US university extension publication.
> The stakeholder is the farmer, the decision is what to spray, and the
> design constraint is that being confidently wrong is worse than being
> honestly uncertain."

### 1:00 – 3:00 — Architecture (Rubric: Architecture)

**On screen:** Tab 2 (`/analytics`), default tab is **Architecture →
System pipeline**.

This is where you address the *full* project, not just the neural net.
The system pipeline diagram has every box you need to point at.

**Script (≈120s) — point at each box as you say it:**
> "The rubric says architecture, so I'll show the whole pipeline.
> There are three phases.
>
> **Phase 1: training pipeline.** [Point at row 1.] PlantVillage gives
> us 54,306 images across 39 classes. Stratified 80/10/10 split with
> seed 42 — the test set is held out, the model has never seen it.
> Preprocessing is ImageNet normalization plus resize and center crop —
> the exact same `eval_tfms` is imported by the API at serving time, so
> there's no chance of drift. The model is EfficientNet-B0 pretrained on
> ImageNet, with a custom 39-class head we attached. Training uses AdamW
> with decoupled weight decay — I'll come back to that. MLflow logs
> every parameter and every per-epoch metric.
>
> **Phase 2: the registry.** [Point at row 2.] The best model is
> registered as `GreenVision` and a separate `promote.py` script moves
> it to the Production stage. That's the contract between training and
> serving — the API knows where to find the model via the URI
> `models:/GreenVision/Production`.
>
> **Phase 3: serving.** [Point at row 3.] The dashboard you're looking
> at calls FastAPI with a multipart upload. The API loads the production
> model at startup via the MLflow lifespan, runs inference with the
> imported `eval_tfms`, looks up the predicted class in a curated
> treatment knowledge base, and returns a typed Pydantic response. The
> dashboard renders that response with confidence-band coloring and
> severity gating.
>
> Every box you see has the actual file path under it. Every box tagged
> with a Rubric badge satisfies a specific assignment requirement."

Optional dive: click the **Model** sub-tab to show the EfficientNet-B0
animation. Click **Play inference** for 5 seconds, narrate briefly. Then
return to System pipeline.

### 3:00 – 4:30 — Training results (Rubric: Training results)

**On screen:** Tab 2 → **Training** tab.

**Script (≈90s) — point at the KPI strip first, then the curves:**
> "Training results. KPI strip up top — these numbers come from MLflow.
> Best validation accuracy 99.73% at epoch 18. Held-out test accuracy
> also 99.73%, with a zero-gap between validation and test, which means
> we're not overfitting to val. Random baseline at 1 of 39 classes is
> 2.56%, so we're 97 percentage points above naive guessing.
>
> Below the KPI strip you see the source attribution — MLflow run
> `attempt_002`, tracking URI `file:./mlruns`, model URI
> `models:/GreenVision/Production`. Everything reproducible.
>
> [Point at the curves.] Two phases visible. Phase 1 is the head-only
> warmup, shaded green — three epochs, backbone frozen, classifier head
> training at LR 1e-3. Phase 2 is gradual unfreezing — 21 more epochs,
> backbone at LR 1e-4, head at 1e-3. Phase 2 starts at the boundary and
> you can see the accuracy quickly settle near 100%."

### 4:30 – 5:30 — The optimizer story (Rubric: Architecture + Q&A defense)

**On screen:** Tab 2 → **Optimizer** tab. Click **Run optimizers**.

**Script (≈60s):**
> "I want to spend a minute on the optimizer because it's the part
> people ask about. Same start point, same loss surface, three
> optimizers descending.
>
> [Lines draw live.] The red trajectory is plain SGD — it follows the
> raw gradient and gets pushed around by the bumps in the surface. The
> blue is Adam — adaptive per-axis learning rate, faster, but L2 weight
> decay couples to the gradient and gets scaled by the same adaptive
> rate. The emerald is AdamW — decoupled weight decay applied directly
> to the parameters before the gradient step. That's the algorithm I
> used to train GreenVision. **The trajectories you see are real — I'm
> running the actual update equations in JavaScript on the actual
> surface.**"

### 5:30 – 7:30 — Live demo (Rubric: Live demo + edge case)

**On screen:** Tab 1 (`/`) → Test-set tab (default).

**Script + actions (≈120s):**

1. **Healthy upload (≈30s):**
   > "Let me show the happy path." [Click a Tomato healthy or Apple
   > healthy thumbnail.]
   >
   > "Picked an image from the held-out test partition — the model has
   > never seen this leaf. [Result loads.] Emerald accent strip,
   > 100% confidence, maintenance tips, three cited sources. Inference
   > round-trip was 140 milliseconds end-to-end on this MacBook."

2. **Diseased upload (≈60s):**
   > [Click a Tomato Late blight or Potato Late blight thumbnail.]
   >
   > "Different leaf, also held-out. Now look at the result card. Rose
   > accent on the left edge — that's the severity signal, automatic for
   > anything tagged high-severity. The big confidence number is in
   > rose to match. Time-sensitivity callout says 'act within 24 hours'
   > — that's actionable language, not jargon. Numbered treatment steps
   > starting with the highest-leverage one. Every source link goes to
   > a real extension publication — Cornell, Penn State." [Click a
   > source link to demonstrate.]

3. **Edge case (≈30s):**
   > [Click the Background tile in the chip row.]
   >
   > "Non-leaf input. The model could have confidently misclassified
   > this as some disease, but it didn't. Amber accent, no treatment
   > shown, clear retake guidance. The 39th class in the training set
   > is a `Background_without_leaves` negative class — this is what
   > graceful failure looks like when the input is off-distribution.
   > The dashboard surfaces a completely different UI state."

### 7:30 – 9:00 — Reflection (Rubric: Reflection)

**On screen:** Stay on the most recent result, or jump to Tab 2 →
**Per-class** tab to point at the bottom-5 list.

**Script (≈90s):**
> "Two reflections.
>
> **One design decision that changed during implementation.** The W8A1
> design saved the model as a plain `.pt` artifact and the API did a
> direct `torch.load`. The W9A1 rubric required the MLflow Model
> Registry approach — log the model, register a version, promote to
> Production, load via URI. So I changed the design before training.
> Cleaner deployment story, slightly more dependency at the API
> boundary — a fine trade. The IMPLEMENTATION_GUIDE notes the pivot
> explicitly in Decisions 7 and 8.
>
> **One thing I'd improve with more time.** PlantVillage is studio
> photos — uniform lighting, single leaf on neutral background, same
> camera. The 99.73% number is in-distribution accuracy. Field photos
> with variable lighting, complex backgrounds, hands holding the leaf
> — the literature says PlantVillage-trained models drop 30 to 50
> percentage points on those. With another week I'd add a second model
> trained on field-photo augmentation and have the API route between
> them based on a blur/complexity score. The dashboard does already
> guide users toward plain-background, focused, single-leaf photos —
> that's a UI affordance to keep inference on the trained distribution,
> not a fix for the underlying limitation."

### 9:00 – 10:00 — Transition to Q&A

> "Architecture, results, demo, reflection. Questions?"

---

## Q&A defense — 12 paragraphs, conversational

The five sample questions from the rubric, plus seven more I'd expect a
sharp instructor or peer to ask, with one-paragraph gists. Don't memorize
verbatim; rehearse the shape.

### Q1. "Why did you choose two-phase fine-tuning instead of full unfreeze from epoch 1?"

Catastrophic forgetting. The pretrained backbone is in a useful region
of parameter space and the classifier head is random. With everything
unfrozen at the same LR, the random head's huge initial gradients flow
backward through the network and trash the ImageNet features. Phase 1
protects the backbone absolutely while the head settles. Phase 2 then
lets the backbone fine-tune at a 10× smaller LR so it makes controlled
adjustments instead of overwriting. Gradual unfreezing inside Phase 2
is because the deepest layers are the most ImageNet-specific and need
the most adaptation.

### Q2. "Your model is 99.73% accurate. What does the 0.27% it gets wrong look like?"

[Switch to Per-class or Confusion tab.] 15 misclassifications out of
5,545 test images. The top confusions are between visually similar
tomato pathologies — `Tomato___Late_blight` predicted as
`Tomato___Early_blight`, three instances. Both produce dark spotty
leaf lesions; the differentiating signal is lesion edge sharpness, which
is a fine-texture cue. No errors cross crop boundaries — the model never
confused an apple disease for a tomato one. That's the kind of mistake
a human agronomist would also make from a phone photo.

### Q3. "Walk me through what happens from clicking Diagnose to seeing the result."

[Optimizer or Architecture tab.] The browser builds a FormData with the
file and POSTs to `/api/predict`. Next.js rewrites that to
`localhost:8000/predict` so it's same-origin. FastAPI reads the body,
size-checks (413 if > 10 MB), PIL-decodes (400 if undecodable), runs a
blur check via variance of Laplacian, applies the exact `eval_tfms`
imported from the training package — `Resize(256)`, `CenterCrop(224)`,
`ToTensor`, `Normalize(ImageNet mean, std)` — moves the tensor to MPS,
runs a forward pass under `torch.no_grad()`, softmaxes the 39 logits,
argsorts to get top-3. The top class index looks up `class_names[i]`
from the artifact training saved, which is the key into
`treatments.json` for severity + action steps + sources. The response
goes back as Pydantic JSON. The dashboard renders the result card with
tone-coded colors and severity gating. About 140 milliseconds
end-to-end on MPS.

### Q4. "Why ImageNet normalization? What breaks if you use different values?"

EfficientNet-B0 was pretrained with images normalized by ImageNet's
per-channel mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224,
0.225]`. The first conv layer's weights are calibrated against inputs
already shifted and scaled to that distribution. If you swap to
PlantVillage's per-channel mean, the first layer sees inputs outside
its expected distribution. The network still produces outputs — they
just degrade silently. The architecture lock specifies ImageNet
normalization at both training and inference. The API imports the
exact same `eval_tfms` object so there's no chance of drift.

### Q5. "How do you handle a prediction where the top two classes have nearly equal confidence?"

The UI degrades through four confidence bands. At ≥85%, the result is
shown normally with alternatives visible but de-emphasized. At 70–85%
the band shifts to a more muted emerald. At 40–70% we add an explicit
"model is moderately confident" warning. Below 40%, the treatment is
hidden behind a "Show anyway" disclosure — the primary call-to-action
becomes "Retake photo" and the AI disclaimer text strengthens. The
top-3 alternatives are always visible so the user can see what the
model was almost confident about. That gating exists because the
failure mode at low confidence is the farmer applying the wrong
fungicide, which is harmful.

### Q6. "Why AdamW specifically over Adam?"

[Optimizer tab.] Show the formula card. Adam folds L2 weight decay into
the gradient and then divides by √v̂ — the per-axis adaptive rate. So
the weight decay term gets scaled by that adaptive rate, which means
it isn't behaving like L2 regularization anymore. AdamW applies the
decay directly to the parameters before the gradient step. That
restores L2 regularization under adaptive learning rates. Empirically
1–2 percentage points generalization gain on ImageNet-class tasks.
That's why we used it.

### Q7. "Why didn't you use Streamlit like the rubric mentions?"

The assignment Part 2 only requires "a user interface that calls your
FastAPI endpoint" — the Streamlit reference is in Part 3's integration
checklist as an example. Streamlit is excellent for an internal data
tool in 30 minutes. It's not the right tool for a polished,
animated, interactive surface targeting a non-technical stakeholder.
The 3D visualization suite uses React Three Fiber, which gives me
declarative scene graphs and proper React lifecycle. The form layout,
accessibility, dark mode, typography precision — they all matter for
the farmer-facing flow. Document this choice in
`IMPLEMENTATION_GUIDE.md` Decision 12.

### Q8. "Where do those treatment recommendations come from?"

[Click a source link.] Every disease entry in `data/treatments.json`
cites at least two US university extension publications — UMN, Penn
State, Cornell, UC IPM, NC State. Extension services are the
authoritative source for agricultural recommendations in the US:
publicly funded, peer-reviewed at the university level, written for
farmers. I avoided naming specific brand fungicides because brand
availability is state-regulated and changes with EPA rulings — the
recommendations specify chemical families like "copper-based fungicide"
or "broad-spectrum protectant." Every prediction surfaces an AI
disclaimer pointing the farmer to a local extension agent — that's the
appropriate professional boundary.

### Q9. "What if FastAPI crashes mid-presentation?"

[Switch to a backup terminal.] Static fallback at
`scripts/demo_static.sh` runs only the dashboard. The Analytics page
still works because it reads `web/public/training_data.json` — no
backend dependency. The Diagnose page surfaces the "Backend not
reachable" alert, which actually demonstrates one of the assignment's
required integration scenarios. Worst case I lose the live demo flow
but keep the entire analytics walkthrough and the architecture
animation.

### Q10. "PlantVillage is studio photos. Will this work on actual field photos?"

Honest answer: no, not at 99%. PlantVillage was captured under
controlled conditions — uniform lighting, leaves laid flat on neutral
backgrounds, single camera setup. The 99.73% number is in-distribution
accuracy. Field photos with variable lighting, complex backgrounds,
partial occlusion, motion blur, hand shadows — the literature on
cross-dataset generalization suggests a 30–50 percentage point drop.
The dashboard intentionally guides users to plain-background,
single-leaf, well-lit photos. The 39th `Background_without_leaves`
class lets us reject obviously off-distribution input. Long-term the
right move is field-photo augmentation training or a two-stage system
that detects whether the input is in-distribution before running the
disease classifier.

### Q11. "Why `eval()` AND `torch.no_grad()`? Aren't they doing the same thing?"

Different things and both needed. `model.eval()` flips two behaviors:
dropout stops randomly zeroing activations (so inference is
deterministic), and BatchNorm uses stored running statistics instead of
recomputing from the current batch (so single-image inference doesn't
get garbage statistics from `batch_size=1`). `torch.no_grad()` tells
autograd to skip building the computation graph, which saves memory
and is slightly faster. Without `eval()` predictions would be
non-deterministic and BN statistics would be wrong. Without
`no_grad()` inference would still be correct but slower and would
leak memory under sustained traffic.

### Q12. "How do you verify the same image produces the same prediction?"

Determinism falls out of the design. `eval_tfms` is non-stochastic by
construction (`Resize` → `CenterCrop` → `ToTensor` → `Normalize`),
`model.eval()` disables dropout and BatchNorm sampling, the model
weights don't move during inference, and the device (MPS or CPU) is
deterministic for forward passes. Demonstrable by uploading the same
image twice and getting identical responses — which is one of the
assignment's required integration scenarios. The test-set picker on
the demo page actually lets me click the same thumbnail twice and the
audience can see identical confidence values.

---

## Backup paths — if something breaks

| Failure | Plan B |
|---|---|
| FastAPI crashes | `./scripts/demo_static.sh`, do the Analytics walkthrough only — the Diagnose card already demonstrates the API-down handling |
| Test-set thumbnails 404 | Switch the upload card to the Upload tab and drag a file from `data/test_demo/` |
| Browser tab freezes in a 3D scene | Reload that tab — `demo.sh` keeps running |
| 3D Architecture animation fails | The System pipeline tab is the primary architecture answer; Model is optional dive |
| Whole laptop crashes | Repo is on GitHub — `git pull` on any Node + Python machine and `./scripts/demo.sh` in ~5 minutes |

---

## Dry-run checklist (do this tonight)

- [ ] Full beat sheet read aloud, timed end-to-end. Target: 10:00 ± 20s.
- [ ] Watch the recording at 1.5× — listen for filler words and pacing.
- [ ] Q&A: have someone push back on three random questions. Practice
  the *gist*, not the script.
- [ ] Test failure modes: stop FastAPI mid-demo (Ctrl+C), verify the
  "API not reachable" alert appears within 2 seconds, then restart.
- [ ] Re-pull from GitHub to confirm a fresh machine could run the
  demo: `git pull && ./scripts/demo.sh`.
- [ ] Push the final commit so the submitted URL matches the demo.
