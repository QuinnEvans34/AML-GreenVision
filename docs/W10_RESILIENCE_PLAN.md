# GreenVision · W10P1 Resilience Plan

> **Status:** active · started after professor flagged out-of-distribution
> failure during the W10P1 demo window. Two-track plan to address the
> background-bias problem before W10D3 presentation.
>
> **Branch:** active W10 branch (post-commit baseline)
> **Cutoff:** demo morning W10D3
> **Owner:** Quinn Evans · execution support: Claude (Cowork)

---

## Why this exists

During pre-demo testing, the GreenVision v3 model failed to generalize to
out-of-distribution images — specifically, leaves photographed on field
backgrounds (grass, dirt, complex scenes) rather than the studio-style
neutral backgrounds PlantVillage uses. This is the exact failure mode the
W9A1 Training Report flagged as the project's biggest limitation. The
professor wants outside-internet images in tomorrow's live demo.

**Root cause:** Classic background bias. The model learned both leaf
features *and* "leaf images live on neutral backgrounds." When the
background distribution shifts, accuracy collapses.

**Constraint:** ~12 hours from now to demo time. Cannot retrain from
scratch. Cannot collect a large labeled real-world dataset.

---

## Strategy — two tracks in parallel

| Track | What | Risk | When demoable |
|---|---|---|---|
| **1. rembg at inference** | Production-style background removal preprocessing before the model sees the image | Low — additive | After ~2.5h work |
| **2. Robust-augmentation fine-tune** | Fine-tune v3 with heavy color/spatial/erasing augmentation; promote to v4 if better | Medium — could degrade v3 numbers | Morning verification |
| **3. Narrative update** | Reframe presentation: "I'd improve" → "I identified and fixed" | None | Anytime tonight |

**Demo guarantee:** Track 1 alone is enough for a working, defensible demo.
Track 2 is the bonus that makes the demo *land harder* if it succeeds. If
v4 turns out worse than v3, we stay on v3 + rembg with no loss of
functionality.

---

## Track 1 — `rembg` at inference (the safety net)

### Phase R1 — Install + smoke test (15 min)

**Goal:** Confirm `rembg` works on Quinn's Python 3.14 + MPS setup,
including the ONNX model download.

**Actions:**

1. Add `rembg` to `requirements.txt` (and `pyproject.toml` if used)
2. Install: `pip install rembg`
3. Pre-cache the U²-Net model so the demo machine isn't downloading
   on classroom WiFi:
   ```bash
   python -c "
   from rembg import remove
   from PIL import Image
   img = Image.new('RGB', (256, 256), 'white')
   out = remove(img)
   print(f'rembg ready · output mode {out.mode} · size {out.size}')
   "
   ```
   First run downloads ~170 MB U²-Net model to `~/.u2net/`.

**Verification:**

```bash
ls -lh ~/.u2net/u2net.onnx   # expect ~176 MB
python -c "from rembg import remove; print('import ok')"
```

**Fallback if rembg fails to install:** drop to OpenCV color-thresholding
(simpler, faster, lower quality). Won't need this — rembg installs cleanly
on Apple Silicon.

---

### Phase R2 — FastAPI integration (45 min)

**Goal:** `/predict` optionally applies background removal before model
inference, controlled by a request parameter.

**Files to modify:**

| File | Change |
|---|---|
| `api/inference.py` | Add `remove_background(pil) -> PIL.Image` helper. Update `predict_image()` to accept a `remove_bg: bool` argument and call the helper when true. Track timing separately for the rembg call. |
| `api/routes/predict.py` | Accept `remove_bg` as a `Form()` parameter (multipart-form-data-compatible). Pass it through to `predict_image()`. |
| `api/schemas.py` | Add `background_removed: bool` and `preprocessing_time_ms: float` fields to `PredictionResponse`. |

**Implementation details:**

- rembg returns RGBA. Composite onto a white background before passing
  to the model (so the model sees a white-bg image like training data).
- Cache the rembg session at module level to avoid re-loading the ONNX
  model on every request.
- Skip rembg for input images already on a uniform background (optional
  optimization — skip for tonight).

**Verification:**

```bash
# Start the API
PYTHONPATH=src .venv/bin/uvicorn api.main:app --reload --port 8000

# In another terminal — without rembg (baseline)
curl -s -F "file=@data/test_outside/<some-field-photo>.jpg" http://localhost:8000/predict | jq '{class:.class_name, conf:.confidence, bg_removed:.background_removed}'

# With rembg
curl -s -F "file=@data/test_outside/<some-field-photo>.jpg" -F "remove_bg=true" http://localhost:8000/predict | jq '{class:.class_name, conf:.confidence, bg_removed:.background_removed, prep_ms:.preprocessing_time_ms}'
```

Expected difference: confidence on outside images materially higher when
`remove_bg=true`.

---

### Phase R3 — Dashboard toggle + before/after preview (60 min)

**Goal:** A visible toggle in the upload card so the audience can see the
fix being applied live. Cleaned image shown alongside the result.

**Files to modify:**

| File | Change |
|---|---|
| `web/lib/api-client.ts` | `predict(file, options?: { removeBg?: boolean })` — append `remove_bg` to the FormData when true. |
| `web/components/upload-card.tsx` | Add a `Switch` (shadcn) labeled **"Remove background (for outside photos)"** with a subtitle "Recommended for field photos with grass, dirt, or complex backgrounds." Default: **off** for test-set images, **on** for arbitrary uploads. |
| `web/components/prediction-result.tsx` | Add a "Background removed" badge below the disease name when `background_removed === true`. |
| `web/app/page.tsx` | Manage toggle state. Pass to `predict()` call. |
| `web/lib/types.ts` | Add `background_removed: boolean` and `preprocessing_time_ms: number` to `PredictionResponse`. |

**UI behavior:**

- Toggle is in the upload card header
- Bold when on, muted when off
- Stays toggled across consecutive predictions
- Auto-on when an arbitrary file is uploaded; auto-off when a test-set
  thumbnail is clicked (test set is studio-style — no rembg needed)

**Verification:**

```bash
cd web && npm run dev
# Upload an outside leaf photo with toggle OFF — observe low confidence + likely wrong class
# Toggle ON, re-upload — observe higher confidence + correct class
# Switch to a test-set thumbnail — toggle should auto-disable
```

---

### Phase R4 — Validate with real outside images (30 min)

**Goal:** Confirm rembg actually helps, document any failure cases.

**Test images Quinn collects** (3–5 of each):

| Category | Source suggestions |
|---|---|
| Healthy leaves on grass/dirt | Google Image: "tomato plant leaf", "apple tree leaf outdoor" |
| Diseased leaves (matching trained classes) | Google: "tomato late blight field", "potato early blight crop" |
| Close-up phone photos | Quinn's own garden if any; or close-ups from Google |
| Edge cases | Leaf held in hand, leaf on busy patterned background |

Save to `data/test_outside/<class_hint>/<filename>.jpg` (gitignored — local-only).

**Actions:**

1. For each image: predict with rembg OFF, then rembg ON
2. Record (in a quick markdown table or just terminal output):
   - Actual class (Quinn knows)
   - Predicted class WITHOUT rembg, confidence
   - Predicted class WITH rembg, confidence
3. Compute uplift: how often does rembg bring the correct class into top-1?
   Into top-3?

**Acceptable result for demo viability:** rembg restores correct top-1
prediction on ≥ 60% of outside images. Even 40% with high confidence is
defensible because we can demo the toggle + show the limitation honestly.

---

## Track 2 — Robust-augmentation fine-tune (the upside)

### Phase T1 — Augmentation pipeline design (15 min)

**Goal:** Add a `train_tfms_robust` to `src/greenvision/data/transforms.py`
that aggressively forces the model away from background reliance.

**The pipeline:**

```python
train_tfms_robust = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),    # wider zoom — exposes more background variation
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),                                # more aggressive than v3's 15°
    transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15    # stronger than v3
    ),
    transforms.RandomGrayscale(p=0.10),                           # forces texture, not color
    transforms.RandomApply([transforms.GaussianBlur(5)], p=0.25), # simulate phone-photo focus
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(
        p=0.40, scale=(0.02, 0.25), ratio=(0.3, 3.3), value="random"
    ),                                                            # CutOut — destroys patches incl. background
])
```

**Why this combination:**
- `RandomGrayscale(0.10)` — 1-in-10 batches see grayscale, forcing the
  model to learn from texture and shape, not just color
- `RandomErasing(0.40, scale up to 25%)` — frequently destroys large
  patches of background; teaches the model that "something else there"
  doesn't change the disease
- `ColorJitter(hue=0.15)` — simulates outdoor lighting hue shifts
- `RandomResizedCrop(scale=(0.5, 1.0))` — sometimes crops to just 50%
  of the image, sometimes the full leaf — varies background-to-leaf ratio

**Verification:**

```python
# Quick visual sanity check
from greenvision.data.transforms import train_tfms_robust
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("data/raw/PlantVillage/Tomato___healthy/<any>.JPG")
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax in axes:
    aug = train_tfms_robust(img)
    # Inverse normalize for display
    ax.imshow(aug.permute(1,2,0).numpy() * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
plt.savefig("docs/screenshots/augmentation_samples.png")
```

Eyeball: each output should look obviously different from the input.

---

### Phase T2 — Training script — fine-tune from v3 (30 min)

**Goal:** Add a `--fine-tune-from <version>` mode to `scripts/train.py`
that loads weights from a registered model instead of fresh ImageNet
init, and uses the robust augmentation pipeline.

**Files to modify:**

| File | Change |
|---|---|
| `scripts/train.py` | Add `--fine-tune-from <int>` (e.g., `--fine-tune-from 3`). When set: load model from `models:/GreenVision/<version>` instead of `build_model()`. Skip Phase 1. Use shorter Phase 2 (10-15 epochs). |
| `scripts/train.py` | Add `--robust-augmentation` flag. When set: use `train_tfms_robust` for training transforms (`eval_tfms` unchanged). |
| `src/greenvision/data/datasets.py` | `build_dataloaders` accepts an optional `train_transform` parameter to override the default. |
| `src/greenvision/training/registry.py` | (if not already present) add `load_model_from_registry(version)` helper. |

**Run command:**

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
    --attempt-id 003 \
    --fine-tune-from 3 \
    --robust-augmentation \
    --phase2-max-epochs 12 \
    --batch-size 64 \
    --num-workers 4
```

**Expected duration:** ~30-45 minutes for 10-12 epochs on M-series MPS
(faster than original 1-hour Phase 2 because fewer epochs and no Phase 1).

---

### Phase T3 — Launch detached training (15 min + 30-45 min training)

**Goal:** Start training as a background process. Quinn sleeps while it
runs.

**Launch:**

```bash
cd /Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/GreenVision
source .venv/bin/activate

mkdir -p logs

nohup env PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 \
    .venv/bin/python scripts/train.py \
    --attempt-id 003 \
    --fine-tune-from 3 \
    --robust-augmentation \
    --phase2-max-epochs 12 \
    > logs/train_attempt_003.log 2>&1 &

disown
echo "Training PID: $!"
```

**Monitor in another terminal:**

```bash
# Watch the log
tail -f logs/train_attempt_003.log

# Or use MLflow UI
mlflow ui --backend-store-uri file:./mlruns --port 5001
# Open http://localhost:5001 → click attempt_003 → phase2 child
```

**Sanity check after 2 epochs (run after 5 minutes):**

- Val accuracy after epoch 1 should be ≥ 95%. If it drops below 90%,
  something is wrong with the fine-tune setup. Cancel and investigate.
- Train accuracy should be lower than val accuracy initially (the augmentation
  makes training harder than validation) — this is good.

---

### Phase T4 — Morning verification + promote (30 min)

**Goal:** Decide between v3 + rembg vs v4 + rembg based on real-world test
performance.

**Morning actions:**

1. Check `logs/train_attempt_003.log` — did training finish?
2. Check MLflow UI — what was the best val accuracy?
3. Run validation suite on outside images using BOTH v3 and v4 (without
   rembg, then with rembg):

```bash
# Quick A/B comparison script
PYTHONPATH=src .venv/bin/python scripts/compare_v3_v4_on_outside.py
```

(If this script doesn't exist, write a quick ad-hoc test that loads both
versions from the registry, runs each outside image through both, and
prints a results table.)

4. **Decision matrix:**

| v4 PlantVillage val | v4 outside accuracy | Action |
|---|---|---|
| ≥ 98% | > v3 outside | Promote v4 to Production |
| ≥ 98% | ≈ v3 outside | Stay on v3 (no regression worth promoting) |
| 95-98% | > v3 outside | Promote v4 — small val regression OK |
| < 95% | any | Keep v3 (val regression too costly) |

5. If promoting v4:

```bash
PYTHONPATH=src .venv/bin/python scripts/promote.py --version 4
# Restart the API to load the new Production version
```

6. Re-bake `web/public/training_data.json` to reflect attempt_003:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_mlflow_for_dashboard.py --attempt 003
```

---

## Track 3 — Presentation narrative update

### Phase P1 — SPEAKER_NOTES update (15 min)

**Files:** `docs/SPEAKER_NOTES.md`

**Changes:**

- §7:30–9:00 (Reflection): change "what I'd improve" to "what I implemented"
  with the rembg + augmentation story. New beat order: (1) identified the
  bias via outside-image testing, (2) shipped rembg as a production
  technique, (3) (if v4 promoted) fine-tuned with robust augmentation.
- Q&A speed-cards: update Q10 (field photos) to say "yes, with rembg
  preprocessing and the augmentation-fine-tuned v4 model" instead of
  "no, not at 99%."
- Add a new Q-card: **"How does rembg work?"** — U²-Net salient object
  segmentation, masks the foreground, composes on white. ~150 ms per
  image on MPS. Show during the live-demo segment.

### Phase P2 — PRESENTATION_SCRIPT update (15 min)

**Files:** `docs/PRESENTATION_SCRIPT.md`

**Changes:**

- 5:30–7:30 (live demo): add a 4th demo step — Quinn picks an outside
  photo, toggles rembg ON, shows the cleaned image, shows the prediction.
- 7:30–9:00 (reflection): mirror the SPEAKER_NOTES changes.
- Q&A: extend Q10, add the new "How does rembg work" Q.

### Phase P3 — System pipeline diagram update (15 min)

**Files:** `web/components/viz/system-pipeline.tsx` and `IMPLEMENTATION_GUIDE.md`

**Changes:**

- Add an **"Inference preprocessing"** stage box between "User upload"
  and "FastAPI backend" in the serving row. Subtitle: "rembg
  (U²-Net) background removal · optional per-request".
- Color: emerald (serve phase).
- File path: `api/inference.py:remove_background()`.
- Rubric tag: tie to Part 1 design decision "How do you handle non-image
  uploads? Corrupted files?" extended to "How do you handle field
  photos?"
- Stub Decisions 14 and 15 in `IMPLEMENTATION_GUIDE.md` (filled out after
  T4 verification).

---

## Time budget

| Track | Phase | Active time | Background time |
|---|---|---|---|
| Track 1 | R1 — Install + smoke | 15 min | — |
| Track 1 | R2 — FastAPI integration | 45 min | — |
| Track 1 | R3 — Dashboard toggle + preview | 60 min | — |
| Track 1 | R4 — Validation | 30 min | — |
| **Track 1 subtotal** | | **2h 30m** | |
| Track 2 | T1 — Augmentation pipeline | 15 min | — |
| Track 2 | T2 — Training script update | 30 min | — |
| Track 2 | T3 — Launch | 15 min | 30-45 min train |
| Track 2 | T4 — Morning verify (Wed AM) | 30 min | — |
| **Track 2 subtotal** | | **1h 30m active** | **30-45 min train** |
| Track 3 | P1 — SPEAKER_NOTES | 15 min | — |
| Track 3 | P2 — PRESENTATION_SCRIPT | 15 min | — |
| Track 3 | P3 — Pipeline diagram + guide | 15 min | — |
| **Track 3 subtotal** | | **45 min** | |
| Buffer / final dry run | — | 30 min | — |
| **TOTAL** | | **~5h 15m active** | **+ overnight train** |

**Realistic schedule (assuming start NOW):**

```
NOW       — Phase R1 (15m)
+15 min   — Phase R2 (45m)
+1h       — Phase R3 (60m)
+2h       — Phase R4 (30m) ← demo working with rembg here
+2h 30m   — Phase T1 (15m)
+2h 45m   — Phase T2 (30m)
+3h 15m   — Phase T3 launch (15m), training starts in background
+3h 30m   — Phase P1 (15m)
+3h 45m   — Phase P2 (15m)
+4h       — Phase P3 (15m)
+4h 15m   — Final dry run with both v3+rembg and outside images (30m)
+4h 45m   — Bed. Training continues.
Morning   — Phase T4 (30m): check training, decide v3 vs v4, deploy
+30 min   — Final morning dry run
```

---

## Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | rembg fails to install (Python 3.14 incompatibility) | Low | High | Try Python 3.11 fallback; or OpenCV color-threshold backup |
| 2 | rembg slow at inference (> 2s per image) | Medium | Low | Show "Processing…" spinner in UI; pre-cache the ONNX model |
| 3 | rembg produces poor masks on hard images | Medium | Medium | Honest: show toggle off case as edge case; framework still defensible |
| 4 | Fine-tune destroys v3 PlantVillage accuracy | Medium | Medium | Run as v4 (separate version), only promote if PlantVillage val ≥ 95% |
| 5 | Training fails overnight (process killed, log error) | Medium | Low | Track 1 still works; come back next session if needed |
| 6 | Quinn collects only easy outside images | Low | High | Suggest specific sources; include genuinely hard examples |
| 7 | rembg model download fails on classroom WiFi | Medium | High | Pre-cache `~/.u2net/u2net.onnx` before leaving home |
| 8 | Dashboard toggle conflicts with test-set picker auto-predict | Low | Medium | Auto-disable toggle on test-set click |
| 9 | New Pydantic field breaks dashboard parsing | Low | Medium | TS types updated in same commit as API change |
| 10 | rembg adds enough latency to feel sluggish (>1s) | Medium | Low | UI already has loading state; latency acceptable for the demo |

---

## Decision points

Quinn has to make these calls during execution:

1. **After R4 (rembg validation):** Is rembg good enough to demo alone?
   → If yes, Track 2 becomes pure upside, no pressure.
   → If no, Track 2 is the real fix and we lean on it.

2. **Morning T4 (v4 evaluation):** Promote v4 or stay on v3?
   → Use the decision matrix above.

3. **Final dry run:** Reorganize live demo to lead with outside image
   (showing rembg fix) or with test-set image (showing in-distribution
   strength)?
   → Recommendation: lead with test-set (high-confidence baseline), then
   switch to outside photo (showcase the engineering).

---

## Files affected

### Track 1 — rembg

- `requirements.txt` or `pyproject.toml` — add `rembg`
- `api/inference.py` — `remove_background()` helper, updated `predict_image()`
- `api/routes/predict.py` — accept `remove_bg` Form param
- `api/schemas.py` — `background_removed`, `preprocessing_time_ms` fields
- `web/lib/api-client.ts` — pass `removeBg` option
- `web/lib/types.ts` — new fields on `PredictionResponse`
- `web/components/upload-card.tsx` — Switch toggle
- `web/components/prediction-result.tsx` — "Background removed" badge
- `web/app/page.tsx` — toggle state management

### Track 2 — fine-tune

- `src/greenvision/data/transforms.py` — `train_tfms_robust`
- `src/greenvision/data/datasets.py` — `train_transform` parameter
- `scripts/train.py` — `--fine-tune-from`, `--robust-augmentation` flags
- `src/greenvision/training/registry.py` — model-load helper (may exist)
- `logs/train_attempt_003.log` (gitignored — training output)
- (after T4) `mlruns/models/GreenVision/version-4/`

### Track 3 — presentation

- `docs/SPEAKER_NOTES.md` — Reflection rewrite + new Q-card
- `docs/PRESENTATION_SCRIPT.md` — Reflection + Q&A updates
- `web/components/viz/system-pipeline.tsx` — add preprocessing box
- `IMPLEMENTATION_GUIDE.md` — Decisions 14 + 15 (stub now, fill after T4)
- `docs/RUBRIC_COMPLIANCE.md` — update Part 1 design decisions

---

## Verification checklist (before sleeping tonight)

- [ ] `pip show rembg` returns a version
- [ ] `~/.u2net/u2net.onnx` exists (~176 MB)
- [ ] `curl /predict` with `remove_bg=true` returns `background_removed: true`
- [ ] Dashboard toggle is visible and labeled clearly
- [ ] Toggle ON → re-prediction returns different (hopefully correct) class
- [ ] At least 3 outside images tested, results documented
- [ ] System pipeline diagram shows preprocessing box
- [ ] SPEAKER_NOTES reflection section updated
- [ ] PRESENTATION_SCRIPT live-demo section updated
- [ ] `nohup` training process is alive (`ps aux | grep train.py`)
- [ ] Training log shows epoch 0–1 metrics looking reasonable
- [ ] All changes committed and pushed to the W10 branch

## Verification checklist (Wed morning before demo)

- [ ] Training process exited cleanly (`tail logs/train_attempt_003.log`)
- [ ] MLflow shows attempt_003 → phase2 with best_val_acc logged
- [ ] If v4 promoted: `models:/GreenVision/Production` loads v4
- [ ] If v4 promoted: `web/public/training_data.json` regenerated
- [ ] One full dry run of the new presentation flow including the rembg demo
- [ ] All emergency exits still work (`scripts/demo_static.sh`, etc.)
- [ ] Repo URL submitted to Canvas

---

## Reference

- Master script: [`docs/PRESENTATION_SCRIPT.md`](./PRESENTATION_SCRIPT.md)
- Speaker crib: [`docs/SPEAKER_NOTES.md`](./SPEAKER_NOTES.md)
- Rubric audit: [`docs/RUBRIC_COMPLIANCE.md`](./RUBRIC_COMPLIANCE.md)
- Architecture: dashboard `/analytics` → Architecture → System pipeline
- Design decisions: dashboard `/about` → Design decisions
