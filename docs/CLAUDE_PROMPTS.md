# Claude Code Prompts — GreenVision W9A1

> Copy each prompt below into Claude Code **one at a time**. Wait for completion, run the verification step from [`docs/IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md), then move to the next prompt.
>
> **Important:** every prompt assumes Claude Code has the GreenVision repo open with read/write access to the project root. Every prompt also tells Claude exactly which files to read for design context.

---

## How to use this file

1. Open this file in your editor (or just keep it in a browser tab).
2. Copy **one** prompt (the text inside the `Prompt` code block, not the surrounding headings) into Claude Code.
3. Let Claude Code finish.
4. Run the verification command listed for that phase (see `docs/IMPLEMENTATION_PLAN.md` for the exact script).
5. If verification passes → move to the next prompt. If it fails → iterate with Claude until it passes before moving on.

The prompts are written to be self-contained — you should not need to tell Claude additional context beyond what's in the prompt.

---

## Prompt 0 — Environment verification

```
Read docs/IMPLEMENTATION_PLAN.md (Phase 0 section). Do not write code yet. Instead, walk through the pre-flight checklist with me, run each verification command from Phase 0 in the terminal, and report the results back so I can confirm the environment is ready before we start coding.

Specifically check:
1. Python version (3.10+)
2. requirements.txt is installed
3. torch.backends.mps.is_available() returns True
4. mlflow and mlflow.pytorch import cleanly
5. data/raw/PlantVillage/ exists and contains 39 subfolders
6. We have at least 5 GB of free disk space in this folder

Tell me which checks pass and which fail. If anything fails, stop and surface the issue — don't try to fix it without my input.
```

---

## Prompt 1 — Data layer

```
Implement the GreenVision data layer.

Context to read first:
- IMPLEMENTATION_GUIDE.md → Decision 3 (data pipeline: splits, transforms, augmentation)
- IMPLEMENTATION_GUIDE.md → Decision 4 (normalization & class index handling)
- .github/copilot-instructions.md → for project conventions (Python 3.10+ typing, Google docstrings, imports order)
- .github/agent.md → for the must-not-touch list (do not change ImageNet normalization values, do not modify class_names artifact structure)
- docs/IMPLEMENTATION_PLAN.md → Phase 1 for the verification step

Create these files exactly:

1. src/greenvision/data/transforms.py
   - Module constants: IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE (= 224)
   - train_tfms: RandomResizedCrop(224, scale=(0.7, 1.0)) → RandomHorizontalFlip → RandomVerticalFlip → RandomRotation(15) → ColorJitter(0.2, 0.2, 0.1) → ToTensor → Normalize(IMAGENET_MEAN, IMAGENET_STD)
   - eval_tfms: Resize(256) → CenterCrop(224) → ToTensor → Normalize(IMAGENET_MEAN, IMAGENET_STD)

2. src/greenvision/data/splits.py
   - stratified_split(dataset, val_frac=0.1, test_frac=0.1, seed=42) using sklearn.model_selection.train_test_split, applied twice (peel off test first, then val from the remainder). Returns (train_idx, val_idx, test_idx).

3. src/greenvision/data/class_names.py
   - NUM_CLASSES = 39
   - CLASS_NAMES_PATH = Path("artifacts/checkpoints/class_names.json")
   - save_class_names(classes): validates len == 39, creates parent dirs, writes JSON
   - load_class_names(): raises FileNotFoundError or RuntimeError with clear messages if missing/malformed

4. src/greenvision/data/datasets.py (new wiring file)
   - build_dataloaders(root, batch_size=64, num_workers=4, seed=42) that:
     a. Calls set_seed first (imported from training/seed.py — we'll create that later, for now placeholder OK)
     b. Loads ImageFolder twice (once with train_tfms, once with eval_tfms) so train and val/test get the right transforms
     c. Calls stratified_split to get indices
     d. Wraps in Subset
     e. Returns (train_loader, val_loader, test_loader, class_names)
     f. DataLoader settings: pin_memory=False, persistent_workers=True, num_workers as passed in
     g. Calls save_class_names(class_names) before returning

All public functions get Google-style docstrings. All function signatures have type hints. Imports order: stdlib → third-party → local, separated by blank lines.

After all four files are created, run docs/IMPLEMENTATION_PLAN.md Phase 1 verification script and show me the output.
```

---

## Prompt 2 — Model layer

```
Implement the GreenVision model layer.

Context to read first:
- IMPLEMENTATION_GUIDE.md → Decision 1 (model architecture & head design)
- .github/copilot-instructions.md → critical constants section
- docs/IMPLEMENTATION_PLAN.md → Phase 2 verification

Create exactly one file:

src/greenvision/models/efficientnet_head.py
- Module constants: NUM_CLASSES = 39, FEATURE_DIM = 1280
- build_model(num_classes=NUM_CLASSES, dropout=0.3) that:
  - Loads torchvision.models.efficientnet_b0 with weights=EfficientNet_B0_Weights.IMAGENET1K_V1
  - Replaces model.classifier with nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(FEATURE_DIM, num_classes))
  - Returns the modified model

Use Google-style docstring on build_model. Include type hints.

After creating, run docs/IMPLEMENTATION_PLAN.md Phase 2 verification (forward pass on a [2, 3, 224, 224] random tensor, verify output shape is [2, 39], print trainable param count).
```

---

## Prompt 3 — Training infrastructure

```
Implement the GreenVision training infrastructure helpers.

Context to read first:
- IMPLEMENTATION_GUIDE.md → Decision 2 (transfer learning, freezing logic, gradual unfreezing schedule)
- IMPLEMENTATION_GUIDE.md → Decision 5 (optimizer, learning rates, bias/BN weight-decay exclusion, ReduceLROnPlateau)
- IMPLEMENTATION_GUIDE.md → Decision 6 (seed setup, MPS config)
- docs/IMPLEMENTATION_PLAN.md → Phase 3 verification

Create these files:

1. src/greenvision/training/phases.py
   - freeze_all_backbone(model): freezes every param not starting with "classifier"; sets BatchNorm2d modules to eval()
   - unfreeze_from_block(model, from_idx): unfreezes features[from_idx:] + classifier; sets BN to train() for unfrozen blocks and eval() for frozen blocks
   - The schedule docstring comment: from_idx=6 is Phase 2a, =3 is Phase 2b, =0 is Phase 2c

2. src/greenvision/training/optim.py
   - Module constants: WEIGHT_DECAY = 1e-4, HEAD_LR = 1e-3, BACKBONE_LR = 1e-4
   - _no_decay_param_names(model): returns the set of param names belonging to BatchNorm1d/2d/3d, LayerNorm, GroupNorm modules
   - build_optimizer(model, head_lr=HEAD_LR, backbone_lr=BACKBONE_LR, wd=WEIGHT_DECAY): builds AdamW with 4 param groups (backbone_decay, backbone_no_decay, head_decay, head_no_decay). Bias terms (param names ending in ".bias") and BN params get weight_decay=0. Only includes parameters where requires_grad=True.

3. src/greenvision/training/schedulers.py
   - build_scheduler(optimizer) returning ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

4. src/greenvision/training/seed.py
   - set_seed(seed=42) that seeds random, numpy.random, torch.manual_seed, torch.cuda.manual_seed_all (if cuda available); sets torch.backends.cudnn.deterministic = True, benchmark = False

Use Google-style docstrings, full type hints. Imports order stdlib → third-party → local.

After creating, run docs/IMPLEMENTATION_PLAN.md Phase 3 verification: build a model, freeze backbone, count trainable params, build optimizer, print param group sizes; then unfreeze from_idx=6 and repeat.
```

---

## Prompt 4 — Training loop

```
Implement the GreenVision training loop.

Context to read first:
- IMPLEMENTATION_GUIDE.md → Decision 6 (training loop, gradual unfreezing schedule, early stopping, overfitting detection) — the code sketch under "Where it shows up in code" is the starting point
- IMPLEMENTATION_GUIDE.md → Decision 2 (Phase 1 = 3 epochs, Phase 2 = up to 20 epochs with PHASE2_SCHEDULE)
- IMPLEMENTATION_GUIDE.md → Decision 5 (ReduceLROnPlateau steps on val_loss)
- docs/IMPLEMENTATION_PLAN.md → Phase 4 verification

Create:

src/greenvision/training/loop.py
- Module constants: MAX_EPOCHS = 20, EARLY_STOP_PATIENCE = 5, PHASE2_SCHEDULE = [(0, 6), (4, 3), (8, 0)]
- from_idx_for(epoch): returns the from_idx for the given Phase 2 epoch based on PHASE2_SCHEDULE
- train_one_epoch(model, loader, optimizer, criterion, device) → dict with keys train_loss, train_acc
- evaluate(model, loader, criterion, device) → dict with keys val_loss, val_acc
- save_checkpoint(model, path, epoch, val_acc) → saves a dict checkpoint with model_state_dict, epoch, val_acc, num_classes, arch
- run_phase1(model, train_loader, val_loader, criterion, device, epochs=3, head_lr=1e-3): builds optimizer for head-only (since backbone is frozen), runs `epochs` epochs, logs MLflow metrics per epoch (assume `import mlflow` and current run is already active). Returns the history dict.
- run_phase2(model, train_loader, val_loader, criterion, device): rebuilds optimizer when from_idx changes (because new params come online), runs ReduceLROnPlateau on val_loss, saves best checkpoint by val_acc, early-stops on val_acc patience 5. Returns (history, best_val_acc, best_epoch).

Use the train_one_epoch/evaluate pattern from any standard PyTorch training tutorial: model.train(), iterate batches, forward/loss/backward/step, accumulate loss and predictions, return metrics.

For the val_loop: model.eval(), with torch.no_grad(), iterate, accumulate, return metrics.

CrossEntropyLoss is the loss function (passed in as `criterion`).

After creating, run docs/IMPLEMENTATION_PLAN.md Phase 4 verification: 1-epoch smoke test on a 200-image subset.
```

---

## Prompt 5 — MLflow integration

```
Implement the GreenVision MLflow integration.

Context to read first:
- IMPLEMENTATION_GUIDE.md → Decision 7 (experiment tracking with MLflow, including the W9A1 update for mlflow.pytorch.log_model and Model Registry promotion)
- IMPLEMENTATION_GUIDE.md → Decision 4 (class_names.json artifact logging)
- docs/IMPLEMENTATION_PLAN.md → Phase 5 verification

Create:

1. src/greenvision/training/mlflow_utils.py
   - Module constants: TRACKING_URI = "file:./artifacts/mlruns", EXPERIMENT = "greenvision"
   - init_mlflow(): sets tracking URI and experiment
   - @contextmanager parent_run(attempt_id, params): yields run, logs params at start
   - @contextmanager phase_run(phase, params): nested=True child run, logs params at start
   - log_training_curves(history, out_path="artifacts/training_curves.png"): matplotlib plot of train_loss/val_loss (left subplot) and train_acc/val_acc (right subplot) over epochs, save PNG, log as artifact
   - log_test_artifacts(y_true, y_pred, class_names): generates 39×39 confusion matrix heatmap via matplotlib + sklearn.metrics.confusion_matrix, saves PNG to artifacts/confusion_matrix.png and logs; generates sklearn.metrics.classification_report, writes to artifacts/per_class_report.txt and logs.
   - log_model_to_registry(model): wraps mlflow.pytorch.log_model(model, "model", registered_model_name="GreenVision") so the run-time call site is one line.

2. src/greenvision/training/registry.py — exactly as specified in IMPLEMENTATION_GUIDE.md Decision 7's code section:
   - MODEL_NAME = "GreenVision"
   - list_versions(client=None) → returns versions sorted newest-first
   - promote_to_production(version, client=None) → uses transition_model_version_stage with archive_existing_versions=True
   - verify_production_loads() → loads via mlflow.pytorch.load_model("models:/GreenVision/Production"), prints model type

Use Google-style docstrings, full type hints.

After creating, write a small script scripts/_phase5_verify.py that:
- Calls init_mlflow
- Opens a parent_run
- Opens a phase_run("phase1", {...}) and logs a single metric
- Opens a phase_run("phase2", {...}) and logs metrics + a fake model via log_model_to_registry
- Then run it and confirm via "mlflow ui --backend-store-uri file:./artifacts/mlruns" that nested runs appear, the model is logged, and the GreenVision model shows up in the Models tab.
```

---

## Prompt 6 — Orchestration

```
Implement the top-level training orchestration.

Context to read first:
- IMPLEMENTATION_GUIDE.md → Decision 7 (orchestration code under "Where it shows up in code")
- All previous Phase files I created (data, models, training/* helpers, mlflow_utils, registry)
- docs/IMPLEMENTATION_PLAN.md → Phase 6 verification

Create:

1. scripts/train.py — top-level CLI
   - argparse: --data-root (default data/raw/PlantVillage), --batch-size (default 64), --phase1-epochs (default 3), --phase2-max-epochs (default 20), --seed (default 42), --attempt-id (required), --num-workers (default 4)
   - main() that:
     a. set_seed
     b. init_mlflow
     c. build_dataloaders → train/val/test loaders + class_names
     d. build_model
     e. move model to device (mps if available else cpu)
     f. criterion = nn.CrossEntropyLoss()
     g. parent_run context with high-level params (seed, batch_size, num_classes, image_size, imagenet_mean, imagenet_std, attempt_id)
     h. inside parent: phase_run("phase1", {...}) → freeze_all_backbone → run_phase1
     i. inside parent: phase_run("phase2", {...}) → run_phase2 (which manages its own unfreezing schedule via from_idx_for)
     j. inside phase2 run: log_training_curves, log_model_to_registry, mlflow.log_artifact("artifacts/checkpoints/class_names.json"), mlflow.log_artifact("artifacts/checkpoints/best.pt")
     k. inside parent (NOT phase2): evaluate on test set, log_test_artifacts
   - if __name__ == "__main__": main()

2. scripts/promote.py — exactly as specified in IMPLEMENTATION_GUIDE.md Decision 7 ("scripts/promote.py — CLI to promote a specific run's model")

After creating, smoke-test scripts/train.py with --phase1-epochs 1 --phase2-max-epochs 2 (a 3-epoch total run) and confirm:
- It runs end-to-end without errors
- MLflow UI shows the new parent run with two children
- The phase2 child has a "model" artifact and the GreenVision Model Registry has a new version

Then PAUSE and tell me the smoke test result. Don't run a full training run — that's Phase 7 (separate).
```

---

## Prompt 7 — Real training run

```
Time to train for real. This is Phase 7 from docs/IMPLEMENTATION_PLAN.md.

Before kicking off, walk through the IMPLEMENTATION_PLAN.md Phase 7 section with me — confirm:
1. MPS fallback env var is set (PYTORCH_ENABLE_MPS_FALLBACK=1)
2. We have enough disk space
3. The smoke test from Phase 6 passed cleanly
4. MLflow UI is reachable so I can watch progress

If everything is green, kick off:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py --attempt-id 001

While it runs, in a separate terminal, start `mlflow ui --backend-store-uri file:./artifacts/mlruns` so I can monitor.

If anything goes wrong during training (NaN loss, accuracy stalls, crashes), apply the relevant contingency from IMPLEMENTATION_PLAN.md Phase 7 troubleshooting section — but check with me before changing hyperparameters.

When training completes, report the final val accuracy, the epoch it converged at, and what version number GreenVision got in the registry.
```

---

## Prompt 8 — Promotion + screenshots

```
This is Phase 8 from docs/IMPLEMENTATION_PLAN.md.

Steps:
1. Look at the MLflow UI and identify the GreenVision Model Registry version that corresponds to my best training run.
2. Run: python scripts/promote.py --version N (where N is that version)
3. Verify: python -c "import mlflow.pytorch; mlflow.set_tracking_uri('file:./artifacts/mlruns'); m = mlflow.pytorch.load_model('models:/GreenVision/Production'); print(type(m))"
   Expected output includes 'EfficientNet' in the type name.

4. Then walk me through capturing the two required screenshots:
   - docs/screenshots/val_accuracy_curve.png — from the best run's phase2 child, the val_acc metric chart
   - docs/screenshots/registry_production.png — from the Models tab, GreenVision @ Production stage

Tell me the macOS shortcut sequence for area screenshots so I can capture them quickly (Cmd+Shift+4 etc.) and where on disk to save them.
```

---

## Prompt 9 — Deliverables wrap-up

```
This is Phase 9 from docs/IMPLEMENTATION_PLAN.md — the final deliverables pass.

Do these in order:

1. Fill in docs/TRAINING_REPORT.md using my actual numbers. Pull from the training log:
   - Final val accuracy
   - Total epochs run
   - Training time (wall-clock)
   - Strategy (already locked in IMPLEMENTATION_GUIDE.md — summarize, don't re-decide)
   - Any hyperparameter that got tuned mid-training
   - "Most surprising finding" — ask me for input on this one
   - Improvement over baseline (final_val_acc - 1/39)

2. Update IMPLEMENTATION_GUIDE.md "Open questions to resolve during implementation" section near the bottom — strike-through (~~text~~) or move to a "Resolved during W9A1" sub-section anything we now have an empirical answer for.

3. Update .github/copilot-instructions.md "Architecture notes" if anything ended up differing from the design (e.g., if we ended up using a different scheduler, different LR, different unfreezing schedule timing).

4. Run final commit:
   git add .
   git commit -m "W9A1 complete: training pipeline + GreenVision @ Production, val accuracy X.X%"
   git push origin main

Then verify on github.com that the new commit is there with my email attribution, and tell me the URL to submit.
```

---

## Bonus prompts (use if something breaks)

### If the verification step for any phase fails

```
The Phase N verification step failed with this error:

[paste the error here]

Read the relevant section of docs/IMPLEMENTATION_PLAN.md Phase N, look at the files we just created, and figure out what's wrong. Don't make assumptions — show me your diagnosis first, then propose a fix, then apply it. After fixing, re-run the verification.
```

### If MLflow UI shows the model but Production-stage load fails

```
mlflow.pytorch.load_model("models:/GreenVision/Production") is erroring with:

[paste the error here]

Check:
1. Was the model actually promoted to Production? (mlflow.tracking.MlflowClient().get_latest_versions("GreenVision", stages=["Production"]) should return one entry)
2. Did the original log_model call succeed without errors? Look at the relevant MLflow run's artifacts directory on disk.
3. Are we calling load_model with the right tracking URI?

Report what you find before changing anything.
```

### If training accuracy stalls below 80%

```
Training finished with val accuracy X.X% — below the 80% W9A1 minimum. Don't change the architecture (that's locked) — instead, read docs/IMPLEMENTATION_PLAN.md Phase 7 "if accuracy stalls" section, pick the most likely contingency given the MLflow run's training curves, and propose a SPECIFIC change before applying it. Then we re-run.
```

---

## Quick reference — file paths Claude needs to know

When Claude asks "where does this go?", here are the canonical paths:

```
src/greenvision/data/transforms.py      ← transforms + ImageNet constants
src/greenvision/data/splits.py          ← stratified_split
src/greenvision/data/class_names.py     ← save/load class_names.json
src/greenvision/data/datasets.py        ← build_dataloaders (wiring)

src/greenvision/models/efficientnet_head.py   ← build_model

src/greenvision/training/phases.py      ← freeze/unfreeze
src/greenvision/training/optim.py       ← 4-group AdamW
src/greenvision/training/schedulers.py  ← ReduceLROnPlateau
src/greenvision/training/seed.py        ← set_seed
src/greenvision/training/loop.py        ← train_one_epoch, evaluate, run_phase1/2
src/greenvision/training/mlflow_utils.py    ← run context managers, plotting, log_model_to_registry
src/greenvision/training/registry.py    ← list/promote/verify Production

scripts/train.py                        ← top-level CLI
scripts/promote.py                      ← promote a model version

artifacts/checkpoints/best.pt           ← best checkpoint (dev-only)
artifacts/checkpoints/class_names.json  ← canonical class names
artifacts/mlruns/                       ← MLflow tracking store
docs/screenshots/                       ← W9A1 deliverable screenshots
```
