# GreenVision Training Report

> Filled in after Phase 7 (real training run) completes. Pull numbers from MLflow run history. Strategy details come from `IMPLEMENTATION_GUIDE.md` — this file is the summary, not the source of truth.

## Results

- **Final val accuracy:** `X.XX%`
- **Naive baseline (random):** `2.56%` (1/39 classes — 38 PlantVillage + 1 Background_without_leaves)
- **Improvement over baseline:** `+X.XX` percentage points

## Fine-tuning strategy

- **Approach used:** Two-phase fine-tuning with gradual Phase 2 unfreezing.
  - Phase 1 — head warm-up (3 epochs): frozen backbone (including BN running stats), train classifier head only at LR `1e-3`.
  - Phase 2 — gradual unfreezing (~12-20 epochs, three stages):
    - Phase 2a (~4 epochs): unfreeze `features[6]`–`features[8]` + classifier
    - Phase 2b (~4 epochs): additionally unfreeze `features[3]`–`features[5]`
    - Phase 2c (~4 epochs): additionally unfreeze `features[0]`–`features[2]` (full)
- **Why this strategy:** Pre-trained EfficientNet-B0 weights are already in a useful region of parameter space — large gradient steps would overwrite that knowledge (catastrophic forgetting). Phase 1 protects the backbone absolutely while the random head settles. Phase 2's gradual unfreezing lets the most ImageNet-specific layers (deepest) adapt first, while the universal-feature layers (earliest, learning edges and color blobs) wait their turn — they need the least change anyway. PlantVillage at 54K+ images is large enough for fine-tuning to outperform pure feature extraction, but not so large that training-from-scratch is justified.
- **Learning rates:**
  - Phase 1: head `1e-3`, backbone frozen
  - Phase 2: head `1e-3`, backbone `1e-4` (**10× ratio** — industry-standard sweet spot for fine-tuning)
- **Optimizer:** `torch.optim.AdamW` (not `Adam` — decoupled weight decay) with `weight_decay=1e-4` applied to matrix weights only (bias and BatchNorm params get `weight_decay=0`)
- **Scheduler:** `ReduceLROnPlateau` on val loss, `factor=0.5`, `patience=2`
- **Total epochs:** `N` (Phase 1: `3` + Phase 2: `N-3` — fill in actual)
- **Early stopping:** triggered at epoch `M` on val-accuracy patience `5`
- **Training time:** ~`X` minutes wall-clock on Apple M5 Pro (MPS device, batch size 64, 4 workers)

## What changed during implementation

- **`mlflow.log_artifact("best.pt")` → `mlflow.pytorch.log_model` + Model Registry.** W8A1's design used `log_artifact` to keep FastAPI free of an MLflow dependency. The W9A1 rubric requires the Registry approach (`models:/GreenVision/Production`), which is actually cleaner for deployment. We updated `IMPLEMENTATION_GUIDE.md` Decisions 7 and 8 to match.
- *(Add any hyperparameters tuned during training — Phase 1 epoch count adjusted? Early-stopping patience changed? Weight decay tuned? Augmentation pulled or pushed?)*
- *(Add any architectural surprises — did the gradual unfreezing schedule transition cleanly, or did one stage cause instability that needed delaying?)*

## Most surprising finding

`[Fill in after training — one specific thing you didn't expect. Example candidates: "Phase 1 converged in 2 epochs, not 3 — would have benefited from extending Phase 2 instead." Or: "Val accuracy on the Background_without_leaves class was harder to push past 90% than disease classes — the variance in non-plant images makes it less learnable." Or: "ReduceLROnPlateau fired exactly once and that single LR halving was enough — early stopping never triggered, training just naturally tapered."]`

## Ready for Assignment W10A1

- ✅ Model in Production stage: GreenVision v`N` @ Production
- ✅ Class names artifact saved to `artifacts/checkpoints/class_names.json` AND attached to the phase2 child run in MLflow
- ✅ `mlflow.pytorch.load_model("models:/GreenVision/Production")` verified to load cleanly (output type: `EfficientNet`)
- ✅ Confusion matrix + per-class report logged to parent run for class-level diagnostics
- **Next:** Build the FastAPI serving layer (W10A1) using the model URI pattern documented in `IMPLEMENTATION_GUIDE.md` Decision 8.

---

## Appendix — MLflow run reference

- **Best run:** `attempt_NNN` parent → `phase2` child
- **Run ID:** `<paste from MLflow UI>`
- **Registered version:** `GreenVision v<N>`
- **Tracking URI:** `file:./artifacts/mlruns`

## Appendix — Screenshots

| Screenshot | Path |
|---|---|
| Val accuracy curve (best run, phase2 child) | `docs/screenshots/val_accuracy_curve.png` |
| GreenVision @ Production in Model Registry | `docs/screenshots/registry_production.png` |
