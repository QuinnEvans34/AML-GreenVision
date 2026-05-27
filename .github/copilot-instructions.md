# Copilot Instructions — GreenVision

> Context for GitHub Copilot (and future-me) when working in this repo. If you're an AI assistant editing this codebase, read this file first and let it constrain your suggestions.

## Project description

**GreenVision** is a plant disease classifier. Given a single leaf image, it predicts one of 38 PlantVillage classes (disease state × crop, plus healthy classes) covering 14 crops. The model is **EfficientNet-B0**, pre-trained on ImageNet, fine-tuned on PlantVillage. Training runs are tracked with **MLflow**. The trained model is served behind a **FastAPI** endpoint.

**Users:** growers, agronomists, ag-tech app developers. They upload a leaf photo and expect back a class label + confidence. Mispredictions caused by silent preprocessing drift (wrong normalization, swapped class indices) are the single worst failure mode for these users — they look like working predictions and aren't.

## Dataset details

- **Source:** PlantVillage — 54,306 RGB leaf images.
- **Classes:** 38 total. Each class encodes a (crop, condition) pair; "healthy" variants exist for each of the 14 crops.
- **Layout (on disk):** `torchvision.datasets.ImageFolder` format — `data/raw/PlantVillage/<class_name>/<image>.jpg`.
- **Class index assignment:** `ImageFolder` sorts class folder names **alphabetically** and assigns indices in that order. This ordering must be captured as an artifact (`artifacts/checkpoints/class_names.json`) at training time and reloaded at inference time. **Never re-derive class names at inference by scanning the filesystem.**
- **Class naming convention:** PlantVillage folder names look like `Tomato___Late_blight` or `Apple___healthy` — crop and condition separated by triple underscores. Preserve them as-is.

## Critical constants

These constants are not freely tunable — they're locked to choices the rest of the system depends on:

```python
# ImageNet normalization — required because the backbone was pretrained with these
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Image input
IMG_SIZE = 224                # EfficientNet-B0 default input

# Output
NUM_CLASSES = 38              # PlantVillage class count

# Backbone
FEATURE_DIM = 1280   # output of EfficientNet-B0 `features` block (consumed by the classifier head)
```

If you change any of these, the model breaks silently. Don't.

## Code conventions

- **Python version:** 3.10+ (use modern typing: `list[str]`, `dict[str, int]`, `X | None`).
- **Imports:** stdlib → third-party → local, separated by blank lines. Absolute imports rooted at `src/greenvision/`.
- **Docstrings:** every public function and class gets a docstring. Single-line if the function is trivial; full Google-style (Args / Returns / Raises) for anything non-trivial.
- **Type hints:** all function signatures (params and return). Tensors annotated as `torch.Tensor`.
- **Error handling:**
  - API endpoints catch image-decode failures and return HTTP 400 with a clear message — never leak a raw stack trace.
  - File I/O around checkpoints and class-name artifacts raises `FileNotFoundError` early with a helpful message ("did you run training first?") rather than failing deep in a tensor op.
  - Never swallow exceptions silently.
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for module-level constants.
- **Formatting:** ruff (`ruff format`) with defaults. Line length 100.
- **No magic numbers** in training code — pull learning rates, batch sizes, epoch counts from a config file or function arguments.

## Architecture notes

### Two-phase fine-tuning strategy

GreenVision uses a **two-phase** training procedure. Both phases must run in order; don't collapse them into one.

**Phase 1 — Head warm-up**

- Backbone frozen (`requires_grad = False` on every param whose name does not start with `classifier`).
- Train only the new classifier head.
- Higher learning rate on the head (`~1e-3`).
- Purpose: get the randomly-initialized head into a sane region before any gradient touches the pretrained features.

**Phase 2 — Gradual fine-tuning** (3 stages, ~4 epochs each)

- Backbone unfrozen in three stages, deepest layers first:
  - **Phase 2a**: unfreeze `features[6]`–`features[8]` (last two MBConv stages + final 1×1 conv) + classifier head.
  - **Phase 2b**: additionally unfreeze `features[3]`–`features[5]` (middle MBConv stages).
  - **Phase 2c**: additionally unfreeze `features[0]`–`features[2]` (stem + earliest stages — full unfreeze).
- Optimizer is `torch.optim.AdamW` with four parameter groups:
  - Backbone matrix weights: low LR (`~1e-4`), `weight_decay=1e-4`.
  - Backbone bias/BN params: low LR (`~1e-4`), `weight_decay=0`.
  - Head matrix weights: higher LR (`~1e-3`), `weight_decay=1e-4`.
  - Head bias: higher LR (`~1e-3`), `weight_decay=0`.
- BatchNorm runs in default `train()` mode for unfrozen blocks; frozen blocks keep BN in `eval()` mode.
- Purpose: adapt ImageNet features to leaf-disease specifics without erasing them. Gradual unfreezing preserves the universal-feature layers (edges, color blobs) while letting higher-level features specialize.

**Why backbone gets the lower LR.** The pretrained weights are already useful; large gradient steps would overwrite that knowledge ("catastrophic forgetting"). The head is randomly initialized and needs bigger steps to learn anything. The 10× ratio (`1e-3` head / `1e-4` backbone) is the industry-standard sweet spot.

**Why bias and BN params are excluded from weight decay.** Their purpose isn't to learn feature detectors — they're calibration parameters. Applying weight decay disrupts training dynamics with no regularization benefit. This is convention across HuggingFace, fastai, and timm.

### Train/inference parity (non-negotiable)

The validation transforms (`eval_tfms`) defined in `src/greenvision/data/transforms.py` are **the same object** imported by the FastAPI inference path. Do not re-implement preprocessing in the API. If the resize, crop, or normalization at inference drifts from what was used at validation, predictions silently degrade.

### Class index → name lookup

Class names are persisted **once** to `artifacts/checkpoints/class_names.json` at training time (from `ImageFolder.classes`) and **loaded** at inference time. The order of this list is the contract between the model's output indices and human-readable labels.

### Tracking

Every training attempt is an MLflow run logged to `file:./artifacts/mlruns`. Per-epoch metrics (`train_loss`, `train_acc`, `val_loss`, `val_acc`) and the best checkpoint are attached. If you're tempted to "just train it quickly without logging", don't — losing a run's hyperparameters means losing the run.
