# GreenVision Training Report

> W9A1 deliverable. Numbers pulled from MLflow run `attempt_002` (parent + `phase1` and `phase2` children). Strategy details are summarized here; the full design rationale lives in [`IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md).

## Results

- **Final val accuracy: 99.73%** (best at Phase 2 epoch 15)
- **Test accuracy (held-out, never seen during training): 99.73%** (5,530 / 5,545 images correct)
- **Naive baseline (random):** `2.56%` (1/39 classes — 38 PlantVillage + 1 `Background_without_leaves`)
- **Improvement over baseline:** **+97.17 percentage points**
- **Val/test gap: 0.0000** — model generalizes perfectly within the PlantVillage distribution; no overfitting to the val set.

## Fine-tuning strategy

- **Approach used:** Two-phase fine-tuning with gradual Phase 2 unfreezing.
  - **Phase 1 — head warm-up (3 epochs):** Freeze entire backbone (including BatchNorm running stats via `.eval()`), train new classifier head only at LR `1e-3`. Reached ~93% val accuracy by epoch 3 with a frozen backbone alone.
  - **Phase 2 — gradual unfreezing (21 epochs total, three stages):**
    - **Phase 2a** (from_idx=6): unfreeze `features[6]`–`features[8]` + classifier
    - **Phase 2b** (from_idx=3): additionally unfreeze `features[3]`–`features[5]`
    - **Phase 2c** (from_idx=0): additionally unfreeze `features[0]`–`features[2]` (full unfreeze)
- **Why this strategy.** Pre-trained EfficientNet-B0 weights are already in a useful region of parameter space — large gradient steps would overwrite that ImageNet knowledge (catastrophic forgetting). Phase 1 protects the backbone absolutely while the random head settles into a sane region. Phase 2's gradual unfreezing lets the most ImageNet-specific layers (deepest) adapt first, while the universal-feature layers (earliest, learning edges and color blobs) wait their turn — they need the least change anyway. PlantVillage at ~55K images is large enough for fine-tuning to outperform pure feature extraction, but not so large that training-from-scratch is justified.
- **Learning rates:**
  - Phase 1: head `1e-3`, backbone frozen
  - Phase 2: head `1e-3`, backbone `1e-4` — **10× ratio**, the industry-standard sweet spot for fine-tuning
- **Optimizer:** `torch.optim.AdamW` (not `Adam` — decoupled weight decay is what makes regularization work uniformly across all weights, regardless of gradient history) with `weight_decay=1e-4` applied to **matrix weights only**. Bias terms and BatchNorm scale/shift parameters get `weight_decay=0`.
- **Scheduler:** `ReduceLROnPlateau` on val loss, `factor=0.5`, `patience=2`. Halves the LR if val loss plateaus for 2 epochs without improvement.
- **Loss function:** `nn.CrossEntropyLoss` (standard for multi-class classification; combines softmax + NLL with the log-sum-exp trick for numerical stability).
- **Total epochs:** 24 (Phase 1: 3 + Phase 2: 21)
- **Best epoch:** Phase 2 epoch 15 (val_acc 0.9973). Training continued through epoch 20 as Phase 2 ran its full schedule, with the best-checkpoint saved at epoch 15.
- **Training time:** ~ 1.1h wall-clock on Apple M5 Pro (MPS device, batch size 64, 4 DataLoader workers).

## What changed during implementation

- **`mlflow.log_artifact("best.pt")` → `mlflow.pytorch.log_model` + Model Registry.** The W8A1 design used `log_artifact` to keep FastAPI free of an MLflow runtime dependency. The W9A1 rubric explicitly requires the Registry approach (`models:/GreenVision/Production`) so that the W10A1 serving layer can load the model via the MLflow URI. We updated `IMPLEMENTATION_GUIDE.md` Decisions 7 and 8 to match before training, and the training script logs the model via `mlflow.pytorch.log_model(..., registered_model_name="GreenVision")` which both saves and registers a new version in one call.
- **`No_plant_detected` placeholder → `Background_without_leaves`.** The W8A1 design used a placeholder name for the 39th negative class. The professor's curated dataset uses `Background_without_leaves`, so all docs/code references were updated to match the canonical class name.
- **MLflow tracking URI: `./mlruns`** in the running code. The W8A1 design referenced `file:./artifacts/mlruns`; this was tightened during W9A1 to a simpler relative path so the path appears consistently in the file store, MLflow UI, and registry queries.
- **No hyperparameter tuning was needed during training.** The locked W8A1 design ran end-to-end cleanly on the first complete attempt. Phase 1 hit ~93% val accuracy in 3 epochs (the strategy's main risk — catastrophic forgetting — never materialized), and Phase 2 converged to 99.73% by epoch 15 with no instability at any unfreezing stage transition.
- **Test set evaluation confirmed zero overfitting.** Test acc = val acc = 99.73% with a gap of 0.0000. The stratified 80/10/10 split worked as designed: held-out test data never touched during training generalizes identically to val data used for model selection.

## Most surprising finding

**The test set accuracy exactly matched the val set accuracy at 99.73%, with a gap of 0.0000.** I expected at least 1-2 percentage points of "overfitting to the val set" — that's the typical penalty for using val to drive early stopping and best-checkpoint selection. Instead, val and test agreed exactly down to four decimal places.

That's *not* "wow, our model is amazing" — it's a clue about the dataset itself. PlantVillage is studio-captured: uniform lighting, leaves laid flat against neutral backgrounds, same camera setup across all 38 disease classes. The 39th `Background_without_leaves` class is similarly curated. Under those distributional conditions, a fine-tuned EfficientNet-B0 has so much capacity relative to the task that **every split essentially represents the same distribution**. Field photos taken with phones in real conditions (variable lighting, complex backgrounds, dirt, partial occlusion, hands holding the leaf) would almost certainly drop a PlantVillage-trained model to **50-70% accuracy** — a known result in the literature.

The implication for **W10A1**: the FastAPI serving layer needs explicit UI guidance ("photograph a single leaf against a plain background, in focus, well-lit") to keep users on the input distribution we actually trained on. Without it, we'd be shipping confidently wrong predictions in the field with a model that scores 99.7% on a benchmark it can't honestly deliver in production.

## Ready for Assignment W10A1

- ✅ **Model in Production stage:** `GreenVision` v3 @ Production (was the registered version produced by `attempt_002` → `phase2`)
- ✅ **Class names artifact saved** to `artifacts/checkpoints/class_names.json` AND attached to the phase2 child run in MLflow (39 classes, alphabetically sorted by folder name as `ImageFolder` produces them)
- ✅ `mlflow.pytorch.load_model("models:/GreenVision/Production")` **verified to load cleanly** — returns `<class 'torchvision.models.efficientnet.EfficientNet'>`
- ✅ Confusion matrix (39×39) + per-class precision/recall/F1 report logged to the parent run `attempt_002` for class-level diagnostics
- **Next (W10A1):** Build the FastAPI serving layer using the model URI pattern documented in `IMPLEMENTATION_GUIDE.md` Decision 8 — load via `mlflow.pytorch.load_model("models:/GreenVision/Production")` in a FastAPI `lifespan` context at server startup.

---

## Appendix — MLflow run reference

- **Best training attempt:** `attempt_002` (parent) → `phase2` child run `6fcedd73`
- **Run ID (full):** 6fcedd73ad6a4bd48b11154ff57cee7f
- **Registered version:** `GreenVision v3` @ stage `Production`
- **Tracking URI:** `./mlruns` (file-backed, repo-local)
- **Experiment name:** `greenvision`

## Appendix — Screenshots

| Screenshot | Path |
|---|---|
| Val accuracy curve (best run, `phase2` child) | [`docs/screenshots/Val_accuracy.png`](./screenshots/Val_accuracy.png) |
| GreenVision @ Production in Model Registry | [`docs/screenshots/model_registry.png`](./screenshots/model_registry.png) |
| Full Phase 2 training curves (train/val loss + accuracy per epoch) | [`docs/screenshots/training_curves.png`](./screenshots/training_curves.png) |
