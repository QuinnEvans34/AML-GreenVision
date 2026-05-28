# GreenVision Training Pipeline — Implementation Plan (W9A1)

> Build plan that translates the design in [`IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md) into ordered phases. Each phase is independently verifiable so problems surface early instead of compounding into a 4-hour debug session at 11pm Sunday.
>
> The actual Claude Code prompts to run for each phase live in [`docs/CLAUDE_PROMPTS.md`](./CLAUDE_PROMPTS.md). The final training report template is at [`docs/TRAINING_REPORT.md`](./TRAINING_REPORT.md), to be filled in after Phase 7 completes.

---

## Pre-flight checklist (do this BEFORE Phase 0)

- [ ] PlantVillage dataset placed at `data/raw/PlantVillage/` with 38 disease folders + 1 `No_plant_detected/` folder = **39 subfolders total**.
- [ ] `requirements.txt` installed in the active Python 3.10+ environment.
- [ ] MLflow can write to `artifacts/mlruns/` (no permission errors).
- [ ] Apple Silicon device available: `python -c "import torch; print(torch.backends.mps.is_available())"` returns `True`.
- [ ] Cursor / VS Code / Claude Code extension open with the project root.

---

## Build phases — overview

| # | Phase | Goal | New files | Time | Depends on |
|---|---|---|---|---|---|
| 0 | Environment | Verify everything works before writing code | (none) | 10 min | — |
| 1 | Data layer | DataLoaders produce `[64, 3, 224, 224]` batches with ImageNet norm | `data/transforms.py`, `data/splits.py`, `data/class_names.py`, `data/datasets.py` | 30-45 min | 0 |
| 2 | Model layer | Build EfficientNet-B0 + 39-class head; forward returns `[B, 39]` | `models/efficientnet_head.py` | 15 min | 0 |
| 3 | Training infra | Freeze/unfreeze, optimizer, scheduler, seed helpers | `training/phases.py`, `training/optim.py`, `training/schedulers.py`, `training/seed.py` | 30 min | 2 |
| 4 | Training loop | `train_one_epoch`, `evaluate`, `run_phase1`, `run_phase2` | `training/loop.py` | 45-60 min | 1, 3 |
| 5 | MLflow integration | Nested runs, artifact logging, plots, model registry helpers | `training/mlflow_utils.py`, `training/registry.py` | 30-45 min | 4 |
| 6 | Orchestration | Top-level training script | `scripts/train.py`, `scripts/promote.py` | 30 min | 1, 2, 3, 4, 5 |
| 7 | Real training | Train for real until >80% val accuracy | (none — runs over weekend) | 2-6 hours wall-clock | 6 |
| 8 | Promotion + screenshots | Register Production, capture MLflow UI screenshots | (none — manual steps) | 30 min | 7 |
| 9 | Deliverables | Update IMPLEMENTATION_GUIDE with changes; fill TRAINING_REPORT.md | edits to existing files | 30 min | 8 |

**Total active development time: ~3-4 hours** (excluding the actual training wall-clock).
**Total wall-clock until submission: 2 days of focused work + 1 weekend training run.**

---

## Phase 0 — Environment verification

**Goal.** Catch environment issues now so they don't surface mid-training.

**Steps:**

```bash
# 1. Confirm Python version
python --version    # expect 3.10+

# 2. Install requirements
pip install -r requirements.txt

# 3. Confirm MPS available
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# expect: MPS: True

# 4. Confirm MLflow installed and works
python -c "import mlflow, mlflow.pytorch; print('MLflow:', mlflow.__version__)"

# 5. Confirm dataset layout (substitute your actual path if different)
ls data/raw/PlantVillage/ | wc -l
# expect: 39

# 6. Confirm enough disk space for MLflow runs + checkpoints (~2-5 GB headroom)
df -h .
```

**Stop here** if any of these fail. Don't proceed to Phase 1.

---

## Phase 1 — Data layer

**Goal.** A `DataLoader` that yields batches of shape `[64, 3, 224, 224]` already normalized to ImageNet stats. A separate validation `DataLoader` with deterministic transforms. A `class_names.json` artifact written to `artifacts/checkpoints/`.

**Files to create:**

| File | Contains |
|---|---|
| `src/greenvision/data/transforms.py` | `IMAGENET_MEAN`, `IMAGENET_STD`, `IMG_SIZE`, `train_tfms`, `eval_tfms` |
| `src/greenvision/data/splits.py` | `stratified_split(dataset, val_frac, test_frac, seed)` returning `(train_idx, val_idx, test_idx)` |
| `src/greenvision/data/class_names.py` | `save_class_names(classes)`, `load_class_names()` with validation against `NUM_CLASSES=39` |
| `src/greenvision/data/datasets.py` | `build_dataloaders(root, batch_size)` that wires ImageFolder + transforms + splits + DataLoaders together |

**Design source of truth:** Decisions 3 and 4 of `IMPLEMENTATION_GUIDE.md`.

**Verification:**

```python
# scripts/_phase1_verify.py
from src.greenvision.data.datasets import build_dataloaders
train_loader, val_loader, test_loader, class_names = build_dataloaders(
    "data/raw/PlantVillage", batch_size=64
)
batch_x, batch_y = next(iter(train_loader))
assert batch_x.shape == (64, 3, 224, 224), batch_x.shape
assert batch_x.dtype.is_floating_point, batch_x.dtype
print(f"Classes: {len(class_names)} (expect 39)")
print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
print(f"Per-channel mean: {batch_x.mean(dim=[0,2,3]).tolist()}")  # near 0
print(f"Per-channel std:  {batch_x.std(dim=[0,2,3]).tolist()}")   # near 1
```

Expected: 39 classes, train batches > 0, per-channel mean near 0, std near 1.

---

## Phase 2 — Model layer

**Goal.** `build_model(num_classes=39)` returns an `nn.Module` whose `forward(x)` on `[B, 3, 224, 224]` returns `[B, 39]`.

**Files to create:**

| File | Contains |
|---|---|
| `src/greenvision/models/efficientnet_head.py` | `build_model(num_classes=39, dropout=0.3)`, plus `NUM_CLASSES` and `FEATURE_DIM` constants |

**Design source of truth:** Decision 1 of `IMPLEMENTATION_GUIDE.md`.

**Verification:**

```python
# scripts/_phase2_verify.py
import torch
from src.greenvision.models.efficientnet_head import build_model
model = build_model(num_classes=39)
model.eval()
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    out = model(x)
assert out.shape == (2, 39), out.shape
print(f"✓ Forward pass works, output shape: {out.shape}")
print(f"✓ Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

---

## Phase 3 — Training infrastructure

**Goal.** Freeze/unfreeze backbone in stages. Build a 4-group AdamW optimizer. Build a `ReduceLROnPlateau` scheduler. Set seeds.

**Files to create:**

| File | Contains |
|---|---|
| `src/greenvision/training/phases.py` | `freeze_all_backbone(model)`, `unfreeze_from_block(model, from_idx)` |
| `src/greenvision/training/optim.py` | `build_optimizer(model, head_lr=1e-3, backbone_lr=1e-4, wd=1e-4)` with 4 param groups (backbone-decay, backbone-no-decay, head-decay, head-no-decay) |
| `src/greenvision/training/schedulers.py` | `build_scheduler(optimizer)` returning `ReduceLROnPlateau(mode="min", factor=0.5, patience=2)` |
| `src/greenvision/training/seed.py` | `set_seed(seed=42)` for Python/NumPy/PyTorch/cuDNN |

**Design source of truth:** Decisions 2 and 5 of `IMPLEMENTATION_GUIDE.md`.

**Verification:**

```python
# scripts/_phase3_verify.py
from src.greenvision.models.efficientnet_head import build_model
from src.greenvision.training.phases import freeze_all_backbone, unfreeze_from_block
from src.greenvision.training.optim import build_optimizer

model = build_model()
# Phase 1 freeze
freeze_all_backbone(model)
trainable = sum(1 for p in model.parameters() if p.requires_grad)
all_p = sum(1 for _ in model.parameters())
print(f"Phase 1 — trainable: {trainable}/{all_p} (expect head-only, small number)")
opt = build_optimizer(model)
print(f"Phase 1 — optimizer param groups: {[len(g['params']) for g in opt.param_groups]}")

# Phase 2a unfreeze
unfreeze_from_block(model, from_idx=6)
trainable_2a = sum(1 for p in model.parameters() if p.requires_grad)
print(f"Phase 2a — trainable: {trainable_2a}/{all_p} (expect more than phase 1)")
```

---

## Phase 4 — Training loop

**Goal.** `train_one_epoch`, `evaluate`, plus the orchestrators `run_phase1` and `run_phase2` that combine the unfreezing schedule + LR scheduler + early stopping + checkpoint saving.

**Files to create:**

| File | Contains |
|---|---|
| `src/greenvision/training/loop.py` | `train_one_epoch`, `evaluate`, `save_checkpoint`, `run_phase1`, `run_phase2`, `PHASE2_SCHEDULE`, `MAX_EPOCHS`, `EARLY_STOP_PATIENCE` constants |

**Design source of truth:** Decision 6 of `IMPLEMENTATION_GUIDE.md` has a full sketch — use it as the starting point.

**Verification:** Smoke-test on a tiny subset (don't train the full model yet).

```python
# scripts/_phase4_verify.py — runs 1 epoch on a 200-image subset
from torch.utils.data import Subset, DataLoader
from src.greenvision.data.datasets import build_dataloaders
from src.greenvision.models.efficientnet_head import build_model
from src.greenvision.training.loop import train_one_epoch, evaluate
import torch.nn as nn, torch

train_loader, val_loader, _, _ = build_dataloaders("data/raw/PlantVillage", batch_size=32)
# Use a tiny subset for the smoke test
train_loader = DataLoader(Subset(train_loader.dataset, range(200)), batch_size=32, shuffle=True)
val_loader   = DataLoader(Subset(val_loader.dataset,   range(50)),  batch_size=32)

device = torch.device("mps")
model = build_model().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

m_train = train_one_epoch(model, train_loader, opt, crit, device)
m_val   = evaluate(model, val_loader, crit, device)
print(f"Smoke test: train_loss={m_train['train_loss']:.3f}, val_acc={m_val['val_acc']:.3f}")
```

Expected: loss is a finite positive number, val_acc is a number in [0, 1]. Don't expect good accuracy — this is just verifying the loop runs.

---

## Phase 5 — MLflow integration

**Goal.** Nested runs, per-epoch metric logging, end-of-training artifact attachment, `mlflow.pytorch.log_model` + Model Registry registration.

**Files to create:**

| File | Contains |
|---|---|
| `src/greenvision/training/mlflow_utils.py` | `init_mlflow`, `parent_run`, `phase_run` (context managers), `log_training_curves`, `log_test_artifacts`, `log_model_to_registry` |
| `src/greenvision/training/registry.py` | `list_versions`, `promote_to_production`, `verify_production_loads` |

**Design source of truth:** Decision 7 of `IMPLEMENTATION_GUIDE.md` (including the W9A1 updates for `mlflow.pytorch.log_model` and Model Registry promotion).

**Verification:**

```bash
# Run a minimal MLflow experiment
python scripts/_phase5_verify.py
# Then look at the UI:
mlflow ui --backend-store-uri file:./artifacts/mlruns
# In your browser, confirm:
#   - experiment "greenvision" exists
#   - one parent run "attempt_test"
#   - two nested children "phase1" and "phase2"
#   - phase2 child has a "model" artifact and metrics logged per step
```

---

## Phase 6 — Orchestration

**Goal.** A single CLI script that runs the full pipeline: load data, build model, run Phase 1, run Phase 2, log everything to MLflow, save best checkpoint, log + register the model.

**Files to create:**

| File | Contains |
|---|---|
| `scripts/train.py` | `main()` that parses CLI args, calls everything in order |
| `scripts/promote.py` | CLI to promote a model version to Production |

**Suggested CLI:**

```bash
python scripts/train.py \
    --data-root data/raw/PlantVillage \
    --batch-size 64 \
    --phase1-epochs 3 \
    --phase2-max-epochs 20 \
    --seed 42 \
    --attempt-id 001
```

**Verification (smoke test):** Add `--dry-run` flag (or `--phase1-epochs 1 --phase2-max-epochs 1`) and confirm the whole pipeline runs end-to-end without errors. Don't expect good accuracy — just confirm the pipeline runs.

---

## Phase 7 — Real training run

**Goal.** Train to >80% val accuracy (the W9A1 minimum).

```bash
# Set MPS fallback for any unsupported ops, then run
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py --attempt-id 001
```

**Monitor in MLflow UI:** `mlflow ui --backend-store-uri file:./artifacts/mlruns` in another terminal.

**Expected timing on M5 Pro:**

- Phase 1 (3 epochs, head only on frozen backbone): ~10-15 min
- Phase 2 (12-20 epochs, gradual unfreezing): ~1-3 hours depending on how many epochs before early stopping fires

**If accuracy stalls below 80%:** see the contingencies in `IMPLEMENTATION_GUIDE.md` "Open questions" section. Common fixes:

- Extend Phase 1 from 3 → 5 epochs if val loss was still trending down
- Loosen early-stopping patience to 7
- Tighten `weight_decay` from `1e-4` → `5e-5` if regularization is suppressing learning
- Switch to weighted sampling if MLflow's per-class report shows rare classes underperforming

---

## Phase 8 — Promotion + screenshots

**Goal.** Best model live in MLflow Model Registry @ Production, deliverable screenshots captured.

**Steps:**

```bash
# 1. Identify the best version (look at MLflow UI)
# 2. Promote it
python scripts/promote.py --version N

# 3. Verify the URI loads
python -c "import mlflow.pytorch; mlflow.set_tracking_uri('file:./artifacts/mlruns'); print(type(mlflow.pytorch.load_model('models:/GreenVision/Production')))"
# expect: <class 'torchvision.models.efficientnet.EfficientNet'>
```

**Screenshots to capture** (open MLflow UI, take screenshots, save to `docs/screenshots/`):

1. `docs/screenshots/val_accuracy_curve.png` — the val accuracy curve from the best run's `phase2` child
2. `docs/screenshots/registry_production.png` — Models tab → GreenVision → showing version N @ Production

---

## Phase 9 — Deliverables wrap-up

**Goal.** All W9A1 deliverables present, IMPLEMENTATION_GUIDE.md reflects any changes, TRAINING_REPORT.md is filled in.

**Steps:**

1. Fill in [`docs/TRAINING_REPORT.md`](./TRAINING_REPORT.md) using the template — final val accuracy, strategy summary, training time, surprising finding.
2. Update [`IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md) "Open questions to resolve during implementation" section with what was tuned during training (e.g., "Phase 1 was 3 epochs and converged at epoch 3 — no extension needed").
3. Commit everything: `git add . && git commit -m "W9A1: training pipeline + GreenVision @ Production" && git push origin main`.
4. Submit the GitHub URL.

---

## MPS-specific notes (M5 Pro)

- **Set the env var** before any heavy training run: `export PYTORCH_ENABLE_MPS_FALLBACK=1`. Some torchvision ops still fall back to CPU on MPS — this lets them silently fall back rather than erroring out.
- **DataLoader settings:** `pin_memory=False` (CUDA-only), `num_workers=4` (M5 Pro has 18 CPU cores, but `mmap` overhead caps useful worker count for image data), `persistent_workers=True` (avoid worker startup overhead each epoch).
- **MLflow tensor logging:** if a `.cpu()` conversion is needed before logging a metric (e.g., for stats computed on MPS tensors), do it explicitly — auto-conversion isn't always reliable on MPS.
- **Memory monitoring:** `python -c "import torch; print(torch.mps.current_allocated_memory() / 1e9)"` if you suspect memory pressure. Unified memory means you can keep big tensors around for free, but unused ones still take RAM.

---

## Things that are easy to get wrong (anti-checklist)

- Forgetting `model.eval()` before validation/inference — dropout fires during val, val accuracy is suppressed and looks wrong.
- Forgetting `with torch.no_grad():` around validation — slow + memory waste.
- Forgetting to call `set_seed(42)` before building the model + loaders — non-reproducible runs.
- Saving `class_names.json` AFTER promoting — class index can get out of sync with the model.
- Running `mlflow.pytorch.log_model` outside an active run — silently no-ops or errors.
- Logging metrics without `step=epoch` — MLflow's epoch-over-epoch curves are broken.
- Not setting `PYTORCH_ENABLE_MPS_FALLBACK=1` — some EfficientNet ops will hard-fail on MPS without it.

---

## If something goes wrong during Phase 7 training

- **Loss is NaN.** LR too high. Check Phase 2 backbone LR (should be `1e-4`). Reduce by 5×.
- **Val accuracy stuck around 1/39 (~2.5%).** Model isn't learning. Verify normalization is correct (per-channel mean ≈ 0, std ≈ 1 on training batches). Check Phase 1 actually unfroze the head.
- **Val accuracy plateaus around 50-60%.** Probably converged too early. Extend Phase 2 epochs or unfreeze deeper layers earlier.
- **Training is incredibly slow (> 30 min per epoch).** MPS fallback is hitting too many ops. Check terminal warnings; consider switching specific layers to CPU.
- **`mlflow.pytorch.log_model` errors.** Usually missing dependency or in-active run. Confirm `mlflow.start_run()` context is open when called.
