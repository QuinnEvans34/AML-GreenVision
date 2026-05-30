# GreenVision · W10D3 Presentation Script

> 10 minutes + 2 minutes Q&A. Live demo from the dashboard, **not** slides.
> Everything is one terminal command (`./scripts/demo.sh`) plus a browser tab.

---

## Pre-demo checklist (5 minutes before)

In one terminal, at the repo root:

```bash
./scripts/demo.sh
```

Wait for all three "Server ready" messages, then open these three tabs in your browser **in this order** (so the demo flow doesn't have you fumbling for tabs):

1. **Tab 1 — http://localhost:3000** — the dashboard `/` (Diagnose)
2. **Tab 2 — http://localhost:3000/analytics** — the analytics with the 5 tabs
3. **Tab 3 — http://localhost:5001** — the MLflow UI for the "credibility tab"

Before you start speaking:

- Set browser zoom to **125%** (auditorium projector compensation)
- Switch to **dark mode** (the 3D scenes pop more against the dark background; also flatters slow projectors)
- Pre-load three test images on the desktop in a folder labeled `demo-images/`:
  - `healthy.jpg` — an `Apple___healthy` or `Tomato___healthy` from `data/raw/PlantVillage/`
  - `diseased.jpg` — a `Tomato___Late_blight` or `Potato___Late_blight` (high severity → dramatic UI)
  - `background.jpg` — any non-leaf photo (a desk, a face, a coffee cup) to demo the rejection state
- Have `./scripts/demo_reset.sh` and `./scripts/demo_static.sh` paths memorized — they're your emergency exits

---

## 10-minute beat sheet

### 0:00 – 1:00 — Problem + stakeholder

**On screen:** Tab 1 (`/`), top of page only.

**Script** (60 seconds):
> "If you're a smallholder farmer and your tomato leaves are spotting, you have a 24-hour decision window. Late blight will take your crop in a week. Early blight is annoying but you have time. The two diseases look almost identical to an untrained eye. The wrong fungicide costs you money and damages the plant. The right one saves your harvest.
>
> GreenVision is the front door: a farmer takes a phone photo of a single leaf, uploads it, and gets back the disease name, a confidence score, and a treatment recommendation cited to a US university extension publication. The stakeholder is the farmer, the decision is what to spray, and the constraint is that being confidently wrong is worse than being honestly uncertain."

### 1:00 – 2:30 — Architecture

**On screen:** Tab 2 (`/analytics`) → switch to the **Architecture** tab.

**Script** (90 seconds):
> "The system is four pieces in a line. [Hover the first block.] EfficientNet-B0, pretrained on ImageNet — that's our backbone. Nine convolutional blocks plus our classifier head, the last layer of which is `Linear(1280, 39)`. We fine-tuned it on PlantVillage — 54,306 leaf images across 38 disease and healthy classes for 14 crops, plus a 39th `Background_without_leaves` negative class that lets the model say 'this isn't a leaf' instead of confidently misclassifying a coffee cup."
>
> [Click **Play inference**.]
>
> "When the farmer uploads a photo, it gets resized, center-cropped, ImageNet-normalized, and pushed through the network. The classifier produces 39 logits, softmax gives us probabilities, and the top class plus its confidence is what we show. The treatment text comes from a static JSON knowledge base — 39 entries cited to UMN, Penn State, Cornell, UC IPM extension publications. The whole pipeline is reproducible from `seed=42`, every hyperparameter is logged in MLflow, and the final model is in the Model Registry at the Production stage."

### 2:30 – 4:30 — Training results

**On screen:** Tab 2 → switch to **Overview** tab. Then briefly **Per-class**, then **Confusion**.

**Script** (120 seconds):
> "Training was two phases. [Point to the Phase 1 shaded zone.] Phase 1 was three epochs of head-only training — backbone frozen — to get the random head into a sane region of parameter space before letting it touch the pretrained weights. Phase 2 was 20 epochs of gradual unfreezing — we unlocked the deepest blocks first because those are the most ImageNet-specific, then progressively the shallower ones.
>
> [Point to the convergence point.] Best validation accuracy was 99.73% at epoch 18, head LR 1e-3 and backbone LR 1e-4 — the standard 10× ratio for fine-tuning. The held-out test set hit the same number, 99.73%, with a zero-gap between val and test. We've intentionally avoided overfitting to val.
>
> [Switch to **Per-class** tab. Hover a couple of bars.] Per-class recall is uniformly high — every one of the 39 classes is above 99% F1. Bar height is recall, color gradient is precision rose to emerald. There are no truly weak classes.
>
> [Switch to **Confusion** tab.] Confusion matrix: 5,545 test images, 5,530 correct, 15 misclassifications across the diagonal. The top confusions are between visually-similar tomato pathologies — late blight ↔ early blight, septoria ↔ leaf mold — which is exactly what you'd expect from a model looking at leaf textures."

### 4:30 – 7:00 — Live demo

**On screen:** Tab 1 (`/`).

**Script + actions** (150 seconds):

1. **Healthy upload** (~30s):
   > "Let me show you the happy path." [Drag `healthy.jpg` onto the upload zone, click Diagnose.]
   >
   > "99% confidence, healthy state, maintenance tips, three cited sources. Inference round-trip was [point at the footer] ~150ms on this MacBook."

2. **Diseased upload** (~60s):
   > [Drag `diseased.jpg`.]
   >
   > "Late blight, high severity — see the rose-tinted card border? That's a visual cue for the farmer that the time-sensitivity matters. 'Act immediately, within 24 hours of first lesion' — that's the actionable language. Numbered treatment steps, three sources, every recommendation traces back to a real extension publication that I can read live. [Click one of the sources.] Penn State Extension on tomato late blight."

3. **Background upload — edge case** (~60s):
   > [Drag `background.jpg`.]
   >
   > "Non-leaf input. The model could have confidently misclassified this as Tomato — Spider mites, but it didn't. The rejection state is a completely different UI: no treatment, no severity, no top-k — just clear retake guidance for the farmer. This is what the 39th `Background_without_leaves` class buys us: graceful failure when the input is off-distribution. The boundary works."

### 7:00 – 8:30 — Reflection

**On screen:** Stay on `/`, scroll past the result so the AI disclaimer is visible.

**Script** (90 seconds):
> "Two things changed during implementation. First, the W8A1 design saved the model as a plain `.pt` artifact and the API did a `torch.load`. W9A1's rubric required the MLflow Model Registry approach, so I changed the design before training. Cleaner deployment story, slightly more dependency at the API boundary — fine trade.
>
> Second — and this is the surprising finding — val accuracy and test accuracy matched at 99.73% to four decimal places. Zero gap. My first instinct was 'great model.' The honest read is 'PlantVillage is too clean.' Every image is studio-lit, single leaf on neutral background, same camera. A field photo with variable lighting, shadows, dirt, hands holding the leaf? This model would drop to 50-70% accuracy. That's a documented result in the literature. So this dashboard intentionally guides users to plain-background, focused, single-leaf photos — keeping inference on the input distribution we trained on. The boundary between 'this works' and 'this fails silently' is a UI affordance, not a model property.
>
> If I had another week, I would add a second model trained on field-photo augmentation and have the API choose between them based on a blur/complexity score on the input. That's the production path."

### 8:30 – 10:00 — Buffer + transition to Q&A

> "Architecture, results, demo, reflection. Questions?"

---

## Q&A defense — 10 paragraphs, one per likely question

These aren't answers to memorize verbatim. They're the *gist* of what to say
when each question comes up. The actual answer should be conversational.

### Q1. "Why two-phase fine-tuning instead of just unfreezing everything from epoch 1?"

Catastrophic forgetting. The pretrained backbone is already in a useful region of parameter space — the ImageNet representations are good general visual features. The classifier head is randomly initialized. If we start training everything at the same LR from epoch 1, the random head's huge initial gradients flow backward through the network and trash the pretrained weights before they can adapt. Phase 1 protects the backbone absolutely (`requires_grad=False`) while the head settles. Phase 2 then lets the backbone fine-tune at a 10× smaller LR (`1e-4` vs head at `1e-3`) so it makes small, controlled adjustments to the pretrained features rather than overwriting them. The gradual unfreezing inside Phase 2 — deepest layers first — is because the deepest layers are the most ImageNet-specific and need the most adaptation to PlantVillage.

### Q2. "Your model is 99.73% accurate. What does the 0.27% it gets wrong look like?"

15 misclassifications out of 5,545 test images. [Switch to the Confusion tab.] The top confusion pair is `Tomato — Late blight` predicted as `Tomato — Early blight`, three instances. Visually these two diseases produce similar leaf lesions — both are foliar fungal infections producing dark spots with halos — and the differentiating signal is in fine texture and lesion edge sharpness. The next pair is `Tomato — Septoria leaf spot` ↔ `Tomato — Leaf mold`, two instances each direction. Same story — visually similar pathologies. None of the errors cross crop boundaries (no apple → tomato mistakes). The model has learned crop-level features reliably and is confused only between very-similar within-crop pathologies, which is the kind of error a human agronomist would also make from a photo alone.

### Q3. "Walk me through what happens from clicking Diagnose to seeing the result."

[Switch to the Architecture tab, click Play inference.] The browser takes the selected file, builds a `FormData`, and `POST`s to `/api/predict`. Next.js rewrites that to `localhost:8000/predict` so it's same-origin in dev. FastAPI's handler reads the file body, rejects if it's over 10MB, decodes the bytes as a PIL Image, runs a blur check (variance of Laplacian — low variance triggers a warning), applies the **exact same** `eval_tfms` from training — `Resize(256), CenterCrop(224), ToTensor, Normalize(ImageNet mean, ImageNet std)` — moves the tensor to MPS, runs a forward pass under `torch.no_grad()`, softmaxes the 39 logits, picks the top-3 by argsort. The top class index looks up `class_names[i]` from the JSON artifact training saved, and we use that as the key into `treatments.json` to fetch severity, action steps, and sources. The whole response goes back as JSON. The browser renders the result card with the confidence band coloring, severity chip, treatment steps, top-k alternatives, and the AI disclaimer. About 140 milliseconds end-to-end on MPS.

### Q4. "Why ImageNet normalization? What breaks if you use different values?"

EfficientNet-B0 was pretrained with images normalized by ImageNet's per-channel mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. The first convolutional layer's weights have been calibrated against inputs that have already been shifted and scaled to that distribution. If we change those values — say, use the per-channel mean of PlantVillage instead — the first conv layer sees inputs outside the distribution its weights were trained on. The network still produces outputs, but they're slightly off in a way that doesn't crash anything; it just degrades predictions silently. The architecture decision document calls this out as a non-negotiable lock — we use ImageNet normalization at both training and inference, and the API imports the exact same `eval_tfms` object from `greenvision.data.transforms` so there's no chance of drift.

### Q5. "How do you handle a prediction where the top two classes have nearly equal confidence?"

The UI degrades gracefully through three confidence bands. At ≥85%, the top class is shown prominently and alternatives are visible but de-emphasized. At 70-85%, the same UI but the badge color shifts to a more muted emerald. At 40-70%, we add an explicit warning above the treatment: "model is moderately confident, consider reviewing alternatives." Below 40%, the treatment is hidden behind a "Show anyway" disclosure — clicking it shows the treatment, but the default action is "Retake photo." That gating exists because the failure mode at low confidence is "the farmer applies the wrong fungicide based on a guess," and that's actively harmful. We make them sign off explicitly. The top-3 alternatives are always shown so the farmer can see what the model was almost confident about — useful when the two top classes are visually similar and a human can break the tie.

### Q6. "Why didn't you build the dashboard with Streamlit?"

The assignment mentioned Streamlit but Part 2 only required "a user interface that calls your FastAPI endpoint." Streamlit is excellent for an internal data tool in 30 minutes; it's not the right tool for a polished, animated, interactive surface for a non-technical stakeholder. The 3D visualization suite [gesture at the analytics tab] uses React Three Fiber, which gives me declarative scene graphs and proper React lifecycle. The form layout, accessibility, dark mode, and animation precision matter for the farmer-facing flow. Building this with Next.js cost me about 12 hours; rebuilding the same level of polish in Streamlit would have either not been possible or would have taken longer.

### Q7. "Where do those treatment recommendations come from?"

[Click a source link.] Every disease entry in `data/treatments.json` cites at least two US university extension publications — UMN, Penn State, Cornell, UC IPM, NC State, Iowa State, and so on. Extension services are the authoritative source for agricultural recommendations in the US because they're publicly funded, peer-reviewed at the university level, and explicitly written for farmers, not researchers. I avoided naming specific brand fungicides because brand availability is state-regulated and changes with EPA rulings — instead the recommendations specify chemical families like "copper-based fungicide" or "broad-spectrum protectant." Every prediction surfaces an AI disclaimer telling the farmer to consult their local extension agent for region-specific guidance — that's the appropriate professional boundary.

### Q8. "What if FastAPI crashes mid-presentation?"

[Switch to a backup terminal.] I have a static fallback at `scripts/demo_static.sh` that runs only the dashboard. The Analytics page still works because it reads `web/public/training_data.json` — no backend dependency. The Diagnose page surfaces the "Backend not reachable" alert, which actually demonstrates one of the assignment's integration requirements — graceful handling of API unavailability. Worst case I lose the live diagnose flow but I keep the entire analytics walkthrough and the architecture animation. For the demo today specifically, all three services start under `concurrently` in one command, so a crash in one is visible and recoverable in seconds.

### Q9. "PlantVillage is studio photos. Will this work on actual field photos?"

Honest answer: no, not at 99% accuracy. PlantVillage was captured under controlled conditions — uniform lighting, leaves laid flat on neutral backgrounds, single camera. The 99.73% number is in-distribution accuracy. Field photos introduce variable lighting, complex backgrounds, partial occlusion, motion blur, hand shadows. The literature on cross-dataset generalization for PlantVillage-trained models suggests a 30-50 percentage point drop. The dashboard intentionally guides users to plain-background, single-leaf, well-lit photos — keeping inference on the input distribution we trained on. The 39th `Background_without_leaves` class lets us reject obviously off-distribution input. Long-term, the right move is field-photo augmentation training, or a two-stage system that first detects whether the input is in-distribution and only then runs the disease classifier.

### Q10. "Why `eval()` mode AND `torch.no_grad()`? Aren't they doing the same thing?"

They do different things and we need both. `model.eval()` flips two behaviors in the network: dropout layers stop randomly zeroing activations (so inference is deterministic), and BatchNorm layers use their stored running statistics instead of recomputing from the current batch (so a single-image inference doesn't get garbage statistics from `batch_size=1`). `torch.no_grad()` is unrelated — it tells autograd to skip building the computation graph, which saves memory and is slightly faster. Without `eval()`, predictions would be non-deterministic (dropout) and wrong (BN statistics). Without `no_grad()`, inference would still be correct but slower and would leak memory if you ran a lot of predictions in sequence.

---

## Backup paths — if something breaks on stage

| Failure | Plan B |
|---|---|
| FastAPI crashes | Switch terminals, run `./scripts/demo_static.sh`, do the Analytics walkthrough only. Diagnose tab demonstrates graceful API-unavailable handling. |
| Browser tab freezes during a 3D scene | Reload that tab only (don't restart `demo.sh`). All scenes recover on remount. |
| `npm run dev` errors during the demo | The dashboard is already running and serving from a prior compile. Just don't edit code mid-demo. |
| MLflow UI port is taken | Skip Tab 3. You can show the Registry inside the Architecture tab instead — the model URI is in the dashboard footer. |
| The dataset isn't where it should be | The demo doesn't actually run training; it runs inference via the already-registered Production model. Dataset path issues won't affect the demo. |
| Whole laptop crashes | The repo is on GitHub — you can pull onto any machine with Node + Python in ~5 min if absolutely necessary. |

---

## Dry-run checklist (do this Tuesday night)

- [ ] Full beat sheet read aloud, timed end-to-end. Target: 10:00, ±20 seconds.
- [ ] Watch the recording at 1.5× and listen for filler words and pacing dead air.
- [ ] Q&A: have someone push back on three random questions from the list. Practice the *gist*, not the script.
- [ ] Test the failure modes: kill FastAPI mid-demo, verify the "API not reachable" alert appears within 2 seconds. Run `demo_static.sh` and verify the Analytics tab still works.
- [ ] Confirm the three test images are committed at `data/test_demo/` (or wherever you keep them) so they can't get lost between machines.
- [ ] Push everything to GitHub one final time so the URL submission and the live demo are reading the same code.
