# GreenVision — Implementation Guide

> Design document for the GreenVision plant disease classifier (EfficientNet-B0 fine-tuned on PlantVillage, tracked with MLflow, served with FastAPI). This is the build roadmap for Weeks 9–10.
>
> **This is a living document.** Sections marked `TBD` or `Uncertain` will be filled in or revised as research and implementation progress. Honest uncertainty beats confident guessing.

## Table of contents

1. [Project summary](#1-project-summary)
2. [Decision 1 — Model architecture & head design](#decision-1--model-architecture--head-design)
3. [Decision 2 — Transfer learning strategy (two-phase fine-tuning)](#decision-2--transfer-learning-strategy-two-phase-fine-tuning)
4. [Decision 3 — Data pipeline: splits, transforms, augmentation](#decision-3--data-pipeline-splits-transforms-augmentation)
5. [Decision 4 — Normalization & class index handling](#decision-4--normalization--class-index-handling)
6. [Decision 5 — Optimizer, learning rates, schedulers, regularization](#decision-5--optimizer-learning-rates-schedulers-regularization)
7. [Decision 6 — Training loop: batch size, epochs, early stopping, overfitting detection](#decision-6--training-loop-batch-size-epochs-early-stopping-overfitting-detection)
8. [Decision 7 — Experiment tracking with MLflow](#decision-7--experiment-tracking-with-mlflow)
9. [Decision 8 — Serving with FastAPI: I/O contract & inference preprocessing](#decision-8--serving-with-fastapi-io-contract--inference-preprocessing)
10. [Open questions to resolve during implementation](#open-questions-to-resolve-during-implementation)
11. [Sources](#sources)

---

## Rubric cross-reference — the 7 outline questions → our 8 decisions

The course outline asks 7 specific design questions. This guide answers all of them and adds three additional decisions (optimizer, training loop, MLflow) that aren't enumerated by the outline but inform the W9–10 implementation. Quick map for the grader:

| # | Outline question | Answered in |
|---|---|---|
| 1 | Why EfficientNet-B0? Fine-tuning strategy (freeze what, train what, for how many epochs)? | Decisions **1** + **2** |
| 2 | Data pipeline: train/val split ratio and strategy. Augmentation list with justification for each. | Decision **3** |
| 3 | Serving: how will you load the model in FastAPI? How will you handle class names? | Decisions **8** + **4** |
| 4 | Normalization: what values, applied where (training, val, inference)? | Decision **4** |
| 5 | Class name persistence: how do you ensure model's output index → correct disease name? | Decision **4** |
| 6 | Low-confidence predictions: what does the API return when confidence < 50%? What does the dashboard show? | Decision **8** (`warnings` + dashboard sub-decision) |
| 7 | Error handling: what happens on a non-leaf image? A corrupted file? | Decision **8** (non-leaf via 39th class; corrupted via HTTP 400) |

**Additional design content** (not on the rubric, but locked):

- **Decision 5** — Optimizer (AdamW), learning rates, scheduler, regularization
- **Decision 6** — Training loop (batch size, epochs, early stopping, overfitting detection)
- **Decision 7** — Experiment tracking with MLflow (nested runs, artifacts, end-of-training reports)

---

## 1. Project summary

**What it is.** GreenVision classifies an uploaded leaf image into one of **39** classes: **38 PlantVillage classes** (disease state × crop, plus healthy classes) across **14** crops, plus a **39th `Background_without_leaves` negative class** for non-leaf input rejection. Built on **EfficientNet-B0** pre-trained on ImageNet and fine-tuned on **54,306** PlantVillage images + a background folder of non-plant photos.

**Who uses it.** Growers, agronomists, and ag-tech apps that need a quick triage signal — "what's likely wrong with this leaf" — before reaching for a more expensive diagnostic.

**What it predicts.** A single class label and a confidence score. (Top-k probabilities are returned by the API for downstream UX.)

**Stack.**

- Model: EfficientNet-B0 (`torchvision.models.efficientnet_b0`)
- Training: PyTorch
- Tracking: MLflow
- Serving: FastAPI

---

## Decision 1 — Model architecture & head design

**What I'm deciding.** What to keep from EfficientNet-B0, what to replace, and what the new classifier head looks like.

**Working choice.** ✅ Locked.

- Keep EfficientNet-B0's feature extractor (ImageNet-pretrained).
- Replace the final classifier with a head that maps the **1280-dim** feature vector to **39** class logits (38 PlantVillage classes + 1 `Background_without_leaves` negative class).
- Locked structure: `Dropout(p=0.3) → Linear(1280, 39)`.

**Reasoning.** EfficientNet-B0 is the right base model for GreenVision for three reasons.

First, **learnable downsampling via strided convolutions.** EfficientNet downsamples at every spatial reduction using stride-2 convs, not max pooling. The difference matters for fine-tuning: max pool hard-codes "take the maximum value in each region" and discards the other values; a strided conv learns a weighted combination of all values in each region. With max pool, the network would be permanently stuck with "max wins" downsampling regardless of dataset. With strided convs, every downsampling step can re-learn its weights during Phase 2 to suit PlantVillage's disease patterns — and disease signal often isn't "where is the brightest pixel" but "what's the spatial pattern of dark spots," which learnable downsampling preserves where max pool would flatten it. This is the reason an older max-pool architecture (e.g., VGG) would be a worse choice even if the parameter count were similar.

Second, **strong, transferable pretrained features.** Trained on ImageNet (1.2M images, 1000 classes). Early-layer filters are battle-tested edge/texture detectors that transfer cleanly to leaf images. The 1280-dim feature vector at the head encodes high-level visual content in a way that linear classification can effectively consume.

Third, **good accuracy/compute trade-off for our scale.** EfficientNet-B0 is the smallest of the EfficientNet family (~5.3M parameters), which keeps iteration fast during development and deployment lightweight. A larger variant (B1–B7) could yield a small accuracy bump at significant compute cost — premature optimization for a 39-class problem with 54K+ training images.

For the **head**, keeping it minimal (`Dropout → Linear(1280, 39)`) is the right starting point. The 1280-dim feature space is already well-organized by the pretrained backbone, so a single linear projection is usually sufficient. A hidden layer between pooled features and class logits adds parameters and risks overfitting without obvious benefit; I'll leave that as an explicit "test during implementation" item rather than committing now.

**Where it shows up in code.**

```python
# src/greenvision/models/efficientnet_head.py
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

NUM_CLASSES = 39   # 38 PlantVillage classes + 1 Background_without_leaves (negative) class
FEATURE_DIM = 1280  # EfficientNet-B0 output of features

def build_model(num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> nn.Module:
    """EfficientNet-B0 with a new classification head sized for PlantVillage + Background_without_leaves."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(FEATURE_DIM, num_classes),
    )
    return model
```

**Uncertain about.** Nothing — locked. May revisit the hidden-layer variant (`1280 → 512 → 39`) as a follow-up experiment if Phase 2 plateaus and we want to test additional head capacity. Initial dropout rate of 0.3 is slightly above EfficientNet's default of 0.2 — adjustable if MLflow shows under/over-regularization. Final folder name for the `Background_without_leaves` class is a placeholder until the dataset is provided.

---

## Decision 2 — Transfer learning strategy (two-phase fine-tuning)

**What I'm deciding.** Feature extraction vs. fine-tuning vs. train-from-scratch — and if fine-tuning, what gets frozen when.

**Working choice — two phases with gradual Phase 2 unfreezing.** ✅ Locked.

- **Phase 1 — Head warm-up (3 epochs).** Freeze the entire backbone (including BN running stats via `.eval()`). Train only the new classifier head at LR `1e-3`.
- **Phase 2 — Gradual fine-tuning (~12 epochs total, 3 stages).**
  - **Phase 2a** (~4 epochs): Unfreeze `features[6]`–`features[8]` (last 2 MBConv stages + final 1×1 conv) + the classifier head.
  - **Phase 2b** (~4 epochs): Additionally unfreeze `features[3]`–`features[5]` (middle MBConv stages).
  - **Phase 2c** (~4 epochs): Additionally unfreeze `features[0]`–`features[2]` (stem + earliest MBConv stages).
- Throughout Phase 2: backbone LR `1e-4`, head LR `1e-3` (**10× ratio**). BatchNorm runs in default `train()` mode for unfrozen blocks; frozen blocks keep BN in `eval()` mode.

**Reasoning.** The two-phase strategy comes from Session 1.2: with 54K images, training from scratch is wasteful, and pure feature extraction leaves accuracy on the table. Fine-tuning is the right move — but doing it naively risks **catastrophic forgetting** (Session 1.4): when the classifier head is freshly initialized, gradients flowing into the network are loud, and large updates can overwrite ImageNet features that took millions of images to learn.

**Phase 1 — head warm-up.** Freezing the entire backbone protects pretrained features absolutely while the random head settles. A linear classifier on top of well-organized pretrained features converges fast; 3 epochs is the sweet spot. Contingency: if val loss in MLflow is still trending down at epoch 3, extend to 5.

**Phase 2 — gradual unfreezing.** Instead of opening every backbone parameter at once, we unfreeze in stages from the deepest layers backward. This preserves the universal-feature layers (edges, color blobs, simple textures — Session 1.1) while letting higher-level features specialize for plant disease. The deepest layers learned the most ImageNet-specific features, so they need adaptation first. The earliest layers (universal primitives) are the least worth changing, so they wait until the rest of the network has settled.

**LR ratio.** The 10× ratio (head `1e-3`, backbone `1e-4`) is the standard fine-tuning sweet spot. Pretrained weights are already in a good neighborhood of parameter space and want gentle steps; the head was random-init at the start and still needs bigger steps. 5× would risk catastrophic forgetting; 100× would prevent the backbone from adapting at all.

**BatchNorm handling.** From Session 1.3 we know BN has separate state (running stats) that updates regardless of `requires_grad`. In Phase 1 we explicitly `.eval()` the backbone's BN layers so their running stats stay locked to ImageNet values while the head trains. In Phase 2, as each block becomes unfrozen, we set its BN layers back to `train()` mode so they calibrate to leaf data. Frozen blocks keep their BN in eval mode. Default behavior is fine for batch size 32; if Phase 2 destabilizes, the known mitigation is to keep BN frozen throughout.

**Where it shows up in code.**

```python
# src/greenvision/training/phases.py
import torch.nn as nn

# EfficientNet-B0's backbone is model.features — a Sequential of 9 modules:
#   features[0]    = stem 3×3 conv
#   features[1..7] = MBConv stages 1..7
#   features[8]    = final 1×1 conv (expands to 1280 channels)

def freeze_all_backbone(model: nn.Module) -> None:
    """Phase 1: freeze every backbone param. BN stats stay locked too."""
    for name, p in model.named_parameters():
        if not name.startswith("classifier"):
            p.requires_grad = False
    for m in model.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def unfreeze_from_block(model: nn.Module, from_idx: int) -> None:
    """Phase 2 gradual: unfreeze features[from_idx:] + classifier.

    Schedule:
      Phase 2a → from_idx = 6  (last 2 MBConv stages + final 1×1 conv unfrozen)
      Phase 2b → from_idx = 3  (middle MBConv stages also unfrozen)
      Phase 2c → from_idx = 0  (everything unfrozen)
    """
    for i, block in enumerate(model.features):
        unfreeze = (i >= from_idx)
        for p in block.parameters():
            p.requires_grad = unfreeze
        # BN: train mode if block is unfrozen, eval if still frozen
        for m in block.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train() if unfreeze else m.eval()
    for p in model.classifier.parameters():
        p.requires_grad = True
```

```python
# src/greenvision/training/loop.py — Phase 2 unfreezing schedule
PHASE2_SCHEDULE = [
    (0, 6),  # epoch 0-3:  from_idx=6 → Phase 2a
    (4, 3),  # epoch 4-7:  from_idx=3 → Phase 2b
    (8, 0),  # epoch 8+:   from_idx=0 → Phase 2c (full unfreeze)
]

def from_idx_for(epoch: int) -> int:
    for start, idx in reversed(PHASE2_SCHEDULE):
        if epoch >= start:
            return idx
    return 6  # default to Phase 2a

# Inside the Phase 2 epoch loop:
#   unfreeze_from_block(model, from_idx_for(epoch))
```

**Uncertain about.** Mostly locked. Open items to validate during W9–10:

- Exact timing of Phase 2a → 2b → 2c transitions (currently 4 epochs each; may tighten/loosen based on Phase 2a val curves).
- Contingency for catastrophic forgetting: if val accuracy drops sharply at any stage transition, delay the next unfreeze step or switch BN to eval mode.
- Phase 1 length: extend from 3 to 5 epochs if val loss is still trending down at epoch 3.

---

## Decision 3 — Data pipeline: splits, transforms, augmentation

**What I'm deciding.** How to split PlantVillage; which augmentations preserve labels for leaves; how the `Dataset`/`DataLoader` are wired.

**Working choice.** ✅ Locked.

- Loader: `torchvision.datasets.ImageFolder`, with PlantVillage laid out as `<class_name>/<image>.jpg`.
- Split: **stratified 80/10/10** (train/val/test) via `sklearn.model_selection.train_test_split`, applied twice (peel off test first, then val from the remainder).
- **Train transforms** — five label-preserving augmentations applied stochastically per image:
  - `RandomResizedCrop(224, scale=(0.7, 1.0))`
  - `RandomHorizontalFlip()`
  - `RandomVerticalFlip()`
  - `RandomRotation(15)`
  - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)`
- **Val/test transforms** — deterministic, matching EfficientNet-B0's original ImageNet evaluation pipeline:
  - `Resize(256) → CenterCrop(224) → ToTensor → Normalize(IMAGENET_MEAN, IMAGENET_STD)`
- **Sampling**: uniform (`shuffle=True` on the train DataLoader). No `WeightedRandomSampler`.

**Reasoning.**

**Split strategy — stratified 80/10/10.** PlantVillage has real class imbalance (some diseases have ~5,000 images, some have <200 — roughly a 25:1 ratio). A random split would risk under-representing rare classes in val and test, making metrics noisy and unrepresentative. Stratified preserves the natural per-class distribution in every split — every class shows up in train, val, and test in roughly its true ratio. The 80/10/10 split keeps a held-out test set for honest final reporting; touching test for tuning would leak information. Val drives model selection (early stopping, best-checkpoint saving); test reports the final number only.

**Augmentation philosophy.** From Session 3.1: every augmentation must preserve the label. Each of the five we picked passes that test for leaf disease imagery:

- `RandomResizedCrop` — zoom variations don't change which disease is present.
- `RandomHorizontalFlip` — leaves don't have left/right canonical orientation.
- `RandomVerticalFlip` — leaves photographed on a surface don't have a true "up." (PlantVillage's photos happen to have consistent orientation in the dataset, but real-world phone photos won't — vertical flip helps the model generalize.)
- `RandomRotation(15)` — camera-tilt variation mimics real handheld photography.
- `ColorJitter(0.2, 0.2, 0.1)` — mild lighting variation. Crucially, the brightness/contrast/saturation deltas are **mild** — aggressive color shifts could mask discoloration symptoms (turning yellow leaves green would destroy the `Yellow_Leaf_Curl_Virus` signal).

**Skipped augmentations.** No `GaussianBlur` — disease signal lives in fine texture (lesion borders, stippling), and blur erases it. We treat "non-blurry photo" as a *user-facing requirement* at inference time (UI prompt: "Ensure the photo is in focus") rather than something we train robustness against. Trains a sharper model and ships better user guidance. No `RandomErasing` — could erase the lesion that defines the label.

**Val/test transforms.** The `Resize(256) → CenterCrop(224)` pipeline is exactly what EfficientNet-B0 was evaluated with on ImageNet. Keeping it identical at val time ensures the pretrained backbone sees the input distribution it expects. The exact same `eval_tfms` object is imported by `api/main.py` at inference — this is the train/inference parity rule from Session 3.3. No re-implementation in the API.

**Uniform sampling, not weighted.** Weighted sampling forces rare-class images to be seen multiple times per epoch, which can backfire when those classes have few representative images — the model memorizes the same handful of pictures instead of learning generalizable features. With strong augmentation and dropout, class imbalance often resolves on its own. We start with uniform sampling and monitor per-class precision/recall in MLflow after Phase 2. If rare classes underperform (e.g., recall < 0.7 on classes with <500 images), the mitigation is to switch to `WeightedRandomSampler` *or* class-weighted cross-entropy loss for the next run — but not preemptively.

**The 39th class — `Background_without_leaves` (negative class).** Alongside the 38 PlantVillage disease classes, the training set includes a folder of non-plant photos (vehicles, indoor scenes, random subjects). This "negative class" teaches the model to *reject* obviously non-leaf inputs at inference instead of confidently classifying a car photo as a tomato disease. It's trained the same way as the other 38 classes — same `train_tfms`, same uniform sampling, same `CrossEntropyLoss` — and the API surfaces a specific warning when this class wins (see Decision 8). The folder name `Background_without_leaves` matches the professor's curated dataset; in `ImageFolder`'s alphabetical sort it lands at class index 4 (after the four `Apple___*` classes).

**Where it shows up in code.**

```python
# src/greenvision/data/transforms.py
from torchvision import transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

```python
# src/greenvision/data/splits.py
from sklearn.model_selection import train_test_split

def stratified_split(dataset, val_frac: float = 0.1, test_frac: float = 0.1,
                     seed: int = 42):
    """Stratified train/val/test split of an ImageFolder by class label.

    Returns three index lists that can wrap the dataset in Subset.
    """
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))

    # First: peel off test, stratified
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_frac, stratify=labels, random_state=seed,
    )
    # Then: peel off val from the remainder, stratified
    tv_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_frac / (1 - test_frac),
        stratify=tv_labels,
        random_state=seed,
    )
    return train_idx, val_idx, test_idx
```

**Uncertain about.** Mostly locked. Open items:

- **Weighted sampling**: skip for the baseline. Revisit if per-class recall in MLflow shows rare classes (<500 training images) underperforming after Phase 2. Mitigation toolbox: `WeightedRandomSampler` or class-weighted `CrossEntropyLoss`.
- **Vertical flip**: added on the assumption that real-world phone photos will arrive in arbitrary orientations. If per-class accuracy on PlantVillage's well-oriented training data suffers, drop it.
- **User-facing data quality**: addressed in Decision 8 (FastAPI handler) via a `/predict` response that warns on suspected low-quality input (low contrast, very small image), plus a UI prompt for non-blurry photos.

---

## Decision 4 — Normalization & class index handling

**What I'm deciding.** Which normalization stats to use, how the 39 class names are pinned to indices, and how to keep train and inference in sync.

**Working choice.** ✅ Locked.

- **Normalization stats**: ImageNet `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`. Hardcoded as constants in `src/greenvision/data/transforms.py`. Used identically in `train_tfms` and `eval_tfms`.
- **Stats logged to MLflow** as run params (`imagenet_mean`, `imagenet_std`) at the start of every run. Makes the run self-describing for reproduction.
- **`class_names.json`** persisted to **two locations** at the end of training:
  - **Canonical**: `artifacts/checkpoints/class_names.json` — what the FastAPI handler loads at inference.
  - **MLflow artifact**: attached to the run via `mlflow.log_artifact(...)` — run-level audit trail pairing the class list with the specific checkpoint that produced it.
- **Validation at load time**: `load_class_names()` raises with a clear message if the file is missing or has the wrong number of entries. Catches stale/corrupted artifacts before they silently corrupt predictions.

**Reasoning.**

**Why ImageNet stats, not PlantVillage's own.** From Session 3.2, every layer in the pretrained EfficientNet-B0 (weights, biases, BatchNorm running mean and variance) was tuned assuming inputs in the ImageNet distribution. Computing PlantVillage-specific stats would describe our data better in an absolute sense, but they'd be *wrong relative to the pretrained backbone's expectations* — every input would arrive at the first conv layer in a different distribution than its weights were calibrated for. The cascading effect through the network quietly costs accuracy. The rule: pretrained weights → use the same normalization the model was trained with. Dataset-specific stats are only correct when training from scratch. (See `docs/RESEARCH_LOG.md` §3.2 for the full derivation.)

**Why save class names to both disk and MLflow.** The canonical file gives the FastAPI handler one stable place to look. The MLflow artifact makes each run self-contained: you can pull `class_names.json` *from the run that produced the checkpoint you're serving* and verify it matches what's on disk. Belt and suspenders — for an artifact whose corruption silently desyncs every prediction the model makes, the redundancy is cheap.

**Why validation on load.** From Session 3.4: filesystem sort order varies across operating systems, adding/removing classes shifts every index after it, and the index → name mapping is the contract between the model's output and human-readable labels. If anything tampers with the file (or it's stale), the API would silently produce wrong predictions. The five lines of validation catch that immediately with a clear error message instead of letting hours of mispredictions accumulate.

**Why log normalization stats to MLflow.** They're constants and won't change between runs — but logging them as `mlflow.log_params(...)` makes every run fully self-describing. Anyone (including future-me) could reproduce the exact preprocessing pipeline from the run record alone, without needing access to the code. Documentation-as-code for ~free.

**Where it shows up in code.**

```python
# src/greenvision/data/class_names.py
import json
from pathlib import Path

CLASS_NAMES_PATH = Path("artifacts/checkpoints/class_names.json")
NUM_CLASSES = 39   # 38 PlantVillage classes + 1 Background_without_leaves (negative) class

def save_class_names(classes: list[str]) -> None:
    """Persist the alphabetical class list to the canonical path.

    Raises ValueError if `classes` doesn't match the expected NUM_CLASSES.
    """
    if len(classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, got {len(classes)}: {classes!r}"
        )
    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(json.dumps(classes, indent=2))

def load_class_names() -> list[str]:
    """Load the class list. Fails loudly if missing or malformed."""
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"{CLASS_NAMES_PATH} missing — did you run training first?"
        )
    classes = json.loads(CLASS_NAMES_PATH.read_text())
    if not isinstance(classes, list) or len(classes) != NUM_CLASSES:
        raise RuntimeError(
            f"{CLASS_NAMES_PATH} is malformed: expected list of "
            f"{NUM_CLASSES} strings, got {classes!r}"
        )
    return classes
```

```python
# How this slots into MLflow.
# The full run structure (parent + nested phase children + helper context managers)
# lives in Decision 7 — see `src/greenvision/training/mlflow_utils.py` there.
# At run start, normalization stats are logged as params alongside the rest:

mlflow.log_params({
    "imagenet_mean": IMAGENET_MEAN,
    "imagenet_std":  IMAGENET_STD,
    # ... other run-level params
})

# After training, the canonical artifacts are attached to the run:
mlflow.log_artifact("artifacts/checkpoints/best.pt")
mlflow.log_artifact("artifacts/checkpoints/class_names.json")
```

**Uncertain about.** Nothing — locked. Listed in `.github/agent.md` as a **must-not-modify** category for AI assistants: the normalization constants and the class-names artifact structure are exactly the kind of changes that would silently corrupt every prediction.

---

## Decision 5 — Optimizer, learning rates, schedulers, regularization

**What I'm deciding.** Optimizer family, per-phase learning rates, weight decay, and whether to use a scheduler.

**Working choice.** ✅ Locked.

- **Loss function**: `nn.CrossEntropyLoss` — standard for multi-class classification. Combines log-softmax + NLL loss in one numerically stable step, so the model's `forward` returns raw logits (no softmax inside the model).
- **Optimizer**: `torch.optim.AdamW` (not `Adam`). Decoupled weight decay — Session 4.2.
- **Phase 1 LR**: `1e-3` on head; backbone frozen so its LR is irrelevant.
- **Phase 2 LRs**: head `1e-3`, backbone `1e-4` (10× ratio). Different per-group via AdamW param groups.
- **Weight decay**: `1e-4`, applied **only to matrix weights**. Bias terms and BatchNorm scale/shift parameters get `weight_decay=0`.
- **Scheduler**: `torch.optim.lr_scheduler.ReduceLROnPlateau`, monitoring **val loss**, `factor=0.5`, `patience=2`.
- **No LR warmup**. The two-phase strategy serves the same purpose.
- **Regularization stack** (three independent levers): dropout `p=0.3` in classifier head + decoupled weight decay `1e-4` on matrix weights + augmentation pipeline from Decision 3.

**Reasoning.**

**Why AdamW, not Adam.** From Session 4.2 (the W9D2 teaching topic): plain Adam's L2 regularization is mathematically equivalent to adding `λw` to the gradient — which then gets passed through Adam's adaptive denominator `√v̂ + ε`. Result: weights with active gradient histories get *less* regularization than dormant ones, exactly the opposite of what we want for preventing overfitting. AdamW decouples weight decay from the gradient pipeline so every weight gets uniform shrinkage proportional to its own magnitude. The one-letter difference (`torch.optim.AdamW` vs `torch.optim.Adam`) is the difference between the bug and the fix.

**Why the 10× LR ratio.** From Session 1.4: pretrained backbone weights are already in a "good neighborhood" of parameter space — they want small steps to gently adapt to leaf disease without overwriting ImageNet features (catastrophic forgetting). The classifier head, on the other hand, started random and even after Phase 1 has some adaptation left to do. A single shared LR would force a bad tradeoff: too high and the backbone forgets; too low and the head learns at a crawl. 10× is the industry-standard sweet spot — 5× would risk catastrophic forgetting, 100× would prevent the backbone from adapting meaningfully.

**Why exclude bias and BN from weight decay.** This is convention across HuggingFace's `Trainer`, fastai, timm, and most modern fine-tuning recipes. The case: weight decay's purpose is to keep the *feature-detector weights* (the conv kernels, the linear layer matrices) from growing too large — those are the parameters that can memorize quirky patterns. Bias terms are scalar offsets that have no "magnitude problem"; BatchNorm's learned scale (`γ`) and shift (`β`) are calibration parameters that BN itself manages. Applying weight decay to them can disrupt training dynamics with no regularization benefit. We build the AdamW param groups to put bias/BN params in groups with `weight_decay=0`.

**Why `ReduceLROnPlateau` instead of cosine.** Our gradual unfreezing schedule (Phase 2a → 2b → 2c) creates *natural* training phase transitions when new parameters come online. A fixed cosine schedule would have already wound the LR down by the time Phase 2c unfreezes the early stages — exactly when those layers actually need some LR to adapt. `ReduceLROnPlateau` reacts to what's actually happening: if val loss is making progress, LR stays put; if it plateaus, LR halves. Pairs naturally with early stopping (both react to val signal) and gives us a "warning shot" before stopping — at least one LR reduction before the early-stop trigger.

**Why no LR warmup.** Phase 1 *is* the warmup. It runs at low effective complexity (just the head trains, on top of a frozen backbone), settles the head into a sane region of parameter space, and only then unleashes gradients on the backbone. Adding a linear warmup on top would be over-engineered for this problem — the two-phase design already handles the "noisy gradients at start" issue differently.

**Where it shows up in code.**

```python
# src/greenvision/training/optim.py
import torch
import torch.nn as nn

WEIGHT_DECAY = 1e-4
HEAD_LR      = 1e-3
BACKBONE_LR  = 1e-4

def _no_decay_param_names(model: nn.Module) -> set[str]:
    """Parameter names that should NOT receive weight decay:
    biases + BatchNorm/LayerNorm/GroupNorm scale and shift.
    """
    no_decay: set[str] = set()
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                  nn.LayerNorm, nn.GroupNorm)
    for module_name, module in model.named_modules():
        if isinstance(module, norm_types):
            for pname, _ in module.named_parameters(prefix=module_name, recurse=False):
                no_decay.add(pname)
    return no_decay

def build_optimizer(
    model: nn.Module,
    head_lr: float = HEAD_LR,
    backbone_lr: float = BACKBONE_LR,
    wd: float = WEIGHT_DECAY,
) -> torch.optim.AdamW:
    """AdamW with four parameter groups:
        1. backbone matrix weights   → backbone_lr, weight_decay=wd
        2. backbone bias/BN params   → backbone_lr, weight_decay=0
        3. head matrix weights       → head_lr, weight_decay=wd
        4. head bias                 → head_lr, weight_decay=0
    Only includes parameters with requires_grad=True (respects freezing).
    """
    no_decay_names = _no_decay_param_names(model)

    groups: dict[str, list] = {
        "backbone_decay":    [],
        "backbone_no_decay": [],
        "head_decay":        [],
        "head_no_decay":     [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_head = name.startswith("classifier")
        is_no_decay = name.endswith(".bias") or name in no_decay_names
        key = ("head" if in_head else "backbone") + (
            "_no_decay" if is_no_decay else "_decay"
        )
        groups[key].append(p)

    return torch.optim.AdamW([
        {"params": groups["backbone_decay"],    "lr": backbone_lr, "weight_decay": wd},
        {"params": groups["backbone_no_decay"], "lr": backbone_lr, "weight_decay": 0.0},
        {"params": groups["head_decay"],        "lr": head_lr,     "weight_decay": wd},
        {"params": groups["head_no_decay"],     "lr": head_lr,     "weight_decay": 0.0},
    ])
```

```python
# src/greenvision/training/schedulers.py
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

def build_scheduler(optimizer: Optimizer) -> ReduceLROnPlateau:
    """Halve LR when val loss hasn't improved for 2 epochs.

    Call scheduler.step(val_loss) after each epoch's validation pass.
    Pairs with early stopping (patience > 2) in the training loop.
    """
    return ReduceLROnPlateau(
        optimizer,
        mode="min",        # monitoring val loss (lower is better)
        factor=0.5,         # halve the LR
        patience=2,         # tolerate 2 epochs without improvement
    )
```

**Uncertain about.** Locked. Open items to validate during W9–10:

- Exact `factor` and `patience` for `ReduceLROnPlateau` — `0.5/2` is a sensible default. If we see LR drops happening too aggressively (and val never recovers), bump patience to 3.
- The `1e-4` weight decay magnitude — standard for fine-tuning. Tune up if overfitting persists, down if regularization is suppressing learning.

---

## Decision 6 — Training loop: batch size, epochs, early stopping, overfitting detection

**What I'm deciding.** Batch size, total epochs per phase, when to stop, and how I'll know overfitting is starting.

**Working choice.** ✅ Locked.

- **Batch size: 64.** Tuned for the development hardware (Apple M5 Pro, 64 GB unified memory, 20-core GPU). LRs are *not* scaled — fine-tuning is forgiving of batch-size changes.
- **MPS-specific config**: `device = torch.device("mps")`, DataLoader with `pin_memory=False` (CUDA-only optimization), `num_workers=4` (well within the 18 CPU cores), `persistent_workers=True`.
- **Phase 1**: 3 epochs (locked in Decision 2; extend to 5 if val loss is still trending down).
- **Phase 2**: minimum 12 epochs (to complete the gradual unfreezing schedule), maximum 20.
- **Early stopping**: monitor **val accuracy**, patience **5**. With `ReduceLROnPlateau` firing on val-loss patience 2 (Decision 5), we get up to two LR halvings before early stopping triggers.
- **Best checkpoint**: saved whenever val accuracy improves. Path: `artifacts/checkpoints/best.pt`.
- **Overfitting detection**: log `train_val_acc_gap = train_acc − val_acc` to MLflow every epoch. Manual inspection after each run; no automated action.

**Reasoning.**

**Why batch 64 specifically (and why no LR scaling).** Standard fine-tuning recipes recommend batch 32 because that fits on typical 8–12 GB consumer GPUs — the M5 Pro's 64 GB unified memory has ~14× the headroom needed at batch 64, and the 20-core GPU benefits from the larger batch because it has more parallelism to keep busy (~30-50% wall-clock speedup per epoch). The "noise helps generalization" concern from Session 4.3 is most pronounced at batch 256+; the gap between 32 and 64 is small enough not to measurably hurt a 54K-image fine-tune. No linear-scaling-rule LR bump because fine-tuning is making small adjustments to pretrained weights, not exploring large parameter regions — doubling LRs would risk catastrophic forgetting.

**Why max 20 epochs.** The gradual unfreezing schedule requires 12 epochs minimum to complete all three stages. The extra 8 epochs are insurance — early stopping should fire well before 20, but if training is genuinely making progress at epoch 19, we let it keep going. Cost of "too long" is just compute time; cost of "too short" is missing the converged peak.

**Why val accuracy stops, val loss reduces LR.** From Session 4.4: val loss is smoother and more sensitive — better suited to adaptive decisions like LR reduction where we want to catch plateaus early. Val accuracy is what we ultimately report and aligns with the user-facing metric — better suited to terminal decisions like "stop training" and "save best." Using both signals (val loss for LR, val accuracy for stop and checkpoint) gets the right behavior for each job. Patience 5 on val accuracy paired with patience 2 on val loss means we get ~2 LR halvings before stopping outright.

**Overfitting detection — track the gap.** From Session 4.1: the diagnostic for overfitting in CNNs is the train-vs-val gap. Logging both `train_acc` and `val_acc` per epoch lets MLflow plot them together; explicit `train_val_acc_gap` makes the trend easy to query. Healthy gap is < 5-8%; > 10% is overfitting in progress; > 20% is serious. No automated action — overfitting mitigations (bump weight decay, more aggressive augmentation, fewer Phase 2 epochs) are decisions for the *next* run, informed by this signal.

**Where it shows up in code.**

```python
# src/greenvision/training/loop.py — Phase 2 sketch
import mlflow
import torch
from greenvision.training.phases import unfreeze_from_block
from greenvision.training.optim import build_optimizer
from greenvision.training.schedulers import build_scheduler

MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 5
PHASE2_SCHEDULE = [(0, 6), (4, 3), (8, 0)]

def from_idx_for(epoch: int) -> int:
    for start, idx in reversed(PHASE2_SCHEDULE):
        if epoch >= start:
            return idx
    return 6

def run_phase2(model, train_loader, val_loader, criterion, device):
    best_val_acc, bad_epochs = 0.0, 0
    current_from_idx = -1  # force rebuild on first iteration

    for epoch in range(MAX_EPOCHS):
        # Gradual unfreezing: rebuild optimizer when new blocks come online
        new_from_idx = from_idx_for(epoch)
        if new_from_idx != current_from_idx:
            current_from_idx = new_from_idx
            unfreeze_from_block(model, current_from_idx)
            optimizer = build_optimizer(model)         # picks up newly trainable params
            scheduler = build_scheduler(optimizer)
            mlflow.log_metric("phase2_from_idx", current_from_idx, step=epoch)

        # Train + validate
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = evaluate(model, val_loader, criterion, device)

        # MLflow: per-epoch metrics + overfitting gap + current LRs
        gap = train_m["train_acc"] - val_m["val_acc"]
        mlflow.log_metrics({
            **train_m, **val_m,
            "train_val_acc_gap": gap,
            "lr_head":     optimizer.param_groups[2]["lr"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        # LR reduction on val loss (more sensitive)
        scheduler.step(val_m["val_loss"])

        # Best checkpoint + early stopping on val accuracy (what we report)
        if val_m["val_acc"] > best_val_acc:
            best_val_acc = val_m["val_acc"]
            bad_epochs = 0
            save_checkpoint(model, "artifacts/checkpoints/best.pt",
                            epoch=epoch, val_acc=val_m["val_acc"])
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                mlflow.log_metric("early_stopped_epoch", epoch)
                break

    return best_val_acc
```

```python
# DataLoader setup elsewhere — note the MPS-specific config:
train_loader = DataLoader(
    train_subset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=False,         # CUDA-only optimization, off on MPS
    persistent_workers=True,
)
```

```python
# src/greenvision/training/seed.py — call once at the start of every run
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Make a training run reproducible. Log the seed as an MLflow param."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Uncertain about.** Mostly locked. Open items:

- If MPS performance turns out spotty (some ops silently falling back to CPU), set `PYTORCH_ENABLE_MPS_FALLBACK=1` and watch for warnings; some EfficientNet operations may have edge cases.
- Patience tuning: if early stopping never fires before epoch 20, that suggests val accuracy is still improving and we should raise `MAX_EPOCHS`. Conversely if it fires at epoch 6 or 7 repeatedly, we're stopping too early — bump patience to 7.

---

## Decision 7 — Experiment tracking with MLflow

**What I'm deciding.** What to log, where to log it, and which artifacts to attach to runs.

**Working choice.** ✅ Locked.

- **Tracking URI**: `file:./mlruns`. Portable in the repo, no external server needed. The same file store backs the Model Registry (`mlruns/models/GreenVision/`).
- **Experiment name**: `greenvision`.
- **Run structure — nested**: each full training attempt is a **parent run** containing two child runs (`phase1` and `phase2`).
  - **Parent run** name: `attempt_NNN`. Holds experiment-level params (seed, dataset version, image size, num classes, batch size, ImageNet mean/std).
  - **Child run `phase1`**: 3 epochs, head-only training. Holds Phase 1 params (head LR, Phase 1 epochs).
  - **Child run `phase2`**: up to 20 epochs, gradual unfreezing. Holds Phase 2 params (head LR, backbone LR, weight decay, scheduler config, max epochs, early stop patience, unfreezing schedule).
- **Per-epoch metrics** (logged inside each child run, with `step=epoch`): `train_loss`, `train_acc`, `val_loss`, `val_acc`, `train_val_acc_gap`, `lr_head`, `lr_backbone`, `phase2_from_idx` (Phase 2 only).
- **Model logging — `mlflow.pytorch.log_model`** *(updated for W9A1)*: the trained model is logged with `mlflow.pytorch.log_model(model, "model", registered_model_name="GreenVision")` on the **phase2** child run. This registers a new version in the MLflow Model Registry under the name `GreenVision`.
- **Model Registry promotion** *(new for W9A1)*: after training completes and the best run is identified, a separate `scripts/promote.py` step uses `MlflowClient().transition_model_version_stage(name="GreenVision", version=N, stage="Production", archive_existing_versions=True)` to move the new version to Production and archive whatever was there before. The W10A1 serving layer (Decision 8) loads via `mlflow.pytorch.load_model("models:/GreenVision/Production")`.
- **End-of-training artifacts**:
  - **`class_names.json`** — logged to the phase2 child run via `mlflow.log_artifact(...)` so it travels in the same run as the registered model (W9A1 requirement). Also saved to the canonical `artifacts/checkpoints/class_names.json` from Decision 4.
  - **Training curves PNG** (`train/val loss` and `train/val accuracy` over epochs) — logged to the `phase2` child run.
  - **Confusion matrix PNG** (39×39 heatmap on the test set) — logged to the **parent** run.
  - **Per-class precision/recall/F1 report** (sklearn's `classification_report`, written to a `.txt`) — logged to the **parent** run.
  - The `best.pt` checkpoint is still saved locally to `artifacts/checkpoints/best.pt` for dev-time convenience but is no longer the canonical artifact for serving.

**Reasoning.**

**Why nested runs.** A full GreenVision training attempt is one experiment that happens to have two phases — the parent run captures that, while each child gets only the params that apply to its phase (Phase 1 has no backbone LR; Phase 2's params are quite different). When comparing across attempts later, you compare parents; when drilling into "what happened in Phase 2 of attempt 5," you drill into the child. MLflow's UI supports this natively. Flat runs would either mix incompatible params (one big run with phase-switching) or lose the linkage entirely (two unrelated runs).

**Why `mlflow.pytorch.log_model` + Model Registry (W9A1 revision).** The original W8A1 design chose `mlflow.log_artifact("best.pt")` because it kept the FastAPI serving layer free of the MLflow dependency and let the API do a plain `torch.load`. The W9A1 rubric explicitly requires the opposite: `mlflow.pytorch.log_model` + Model Registry registration + Production-stage promotion + `mlflow.pytorch.load_model("models:/GreenVision/Production")` at serving time. So we're flipping. The new approach is actually better for the long run — URI-based loading (`models:/GreenVision/Production`) means we can roll models forward and back without filesystem hacks, and the Registry gives us version history and stage transitions for free. The original "fewer dependencies at the API boundary" concern is moot in practice because we already require MLflow at training time, and adding it at inference is a small cost for cleaner deployment semantics. The training code itself barely changes — just one `log_artifact("best.pt")` line becomes `mlflow.pytorch.log_model(...)`.

**Why these three end-of-training artifacts.** Each one answers a different question after the run:
- **Training curves** answer "did this run converge cleanly?" — phase-specific, so logged under the phase2 child.
- **Confusion matrix** answers "which classes is the model confusing?" — computed on the test set after both phases complete, so logged under the parent.
- **Per-class report** answers "which specific classes underperform?" — same scope as confusion matrix, also on the parent.

All three are cheap to generate (a few seconds of matplotlib + sklearn) and serve as the artifact trail when reviewing whether to tune hyperparameters for the next attempt.

**Where it shows up in code.**

```python
# src/greenvision/training/mlflow_utils.py
from contextlib import contextmanager
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import confusion_matrix, classification_report

TRACKING_URI = "file:./mlruns"
EXPERIMENT = "greenvision"

def init_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

@contextmanager
def parent_run(attempt_id: str, params: dict):
    """Outer run: one per full training attempt."""
    with mlflow.start_run(run_name=f"attempt_{attempt_id}") as run:
        mlflow.log_params(params)
        yield run

@contextmanager
def phase_run(phase: str, params: dict):
    """Nested child run for one phase (phase1 or phase2)."""
    with mlflow.start_run(run_name=phase, nested=True) as run:
        mlflow.log_params(params)
        yield run

def log_training_curves(history: dict, out_path: str = "artifacts/training_curves.png"):
    """history has lists for train_loss, val_loss, train_acc, val_acc."""
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
    ax_l.plot(history["train_loss"], label="train"); ax_l.plot(history["val_loss"], label="val")
    ax_l.set_title("Loss"); ax_l.legend(); ax_l.set_xlabel("epoch")
    ax_a.plot(history["train_acc"], label="train"); ax_a.plot(history["val_acc"], label="val")
    ax_a.set_title("Accuracy"); ax_a.legend(); ax_a.set_xlabel("epoch")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(out_path)

def log_test_artifacts(y_true, y_pred, class_names):
    """Confusion matrix + per-class report on the test set. Call inside the parent run."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout()
    cm_path = "artifacts/confusion_matrix.png"
    fig.savefig(cm_path, dpi=120); plt.close(fig)
    mlflow.log_artifact(cm_path)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    report_path = "artifacts/per_class_report.txt"
    Path(report_path).write_text(report)
    mlflow.log_artifact(report_path)
```

```python
# src/greenvision/training/run.py — top-level orchestration (W9A1)
import mlflow
import mlflow.pytorch

def train(attempt_id: str, ...):
    init_mlflow()
    with parent_run(attempt_id, parent_params):
        with phase_run("phase1", phase1_params):
            run_phase1(...)
        with phase_run("phase2", phase2_params):
            history, best_model = run_phase2(...)
            log_training_curves(history)

            # W9A1: log the trained model via the pytorch flavor AND register it.
            # `registered_model_name` creates/appends a new version to the registry.
            mlflow.pytorch.log_model(
                best_model,
                "model",
                registered_model_name="GreenVision",
            )
            # Class names travel in the SAME run as the model (W9A1 requirement).
            mlflow.log_artifact("artifacts/checkpoints/class_names.json")
        # Final test-set evaluation logged under the parent run
        y_true, y_pred = evaluate_on_test(...)
        log_test_artifacts(y_true, y_pred, class_names)
```

```python
# src/greenvision/training/registry.py — Model Registry helpers (new for W9A1)
"""MLflow Model Registry helpers for promoting GreenVision to Production."""
from __future__ import annotations
from mlflow.tracking import MlflowClient

MODEL_NAME = "GreenVision"

def list_versions(client: MlflowClient | None = None):
    """Every registered version of GreenVision, newest first."""
    client = client or MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    return sorted(versions, key=lambda v: int(v.version), reverse=True)

def promote_to_production(version: int, client: MlflowClient | None = None) -> None:
    """Move the given version of GreenVision to Production.

    Existing Production versions are automatically archived so we always have
    exactly one Production model — what the FastAPI lifespan loads.
    """
    client = client or MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(version),
        stage="Production",
        archive_existing_versions=True,
    )

def verify_production_loads() -> None:
    """Sanity check that `mlflow.pytorch.load_model` works on Production.

    The W10A1 serving layer depends on this; if it fails here, it'll fail there too.
    """
    import mlflow.pytorch
    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/Production")
    print(f"✓ Production model loaded successfully — type: {type(model).__name__}")
```

```python
# scripts/promote.py — CLI to promote a specific run's model
"""Promote a trained model version to Production.

Usage:
    python scripts/promote.py --version 3
"""
import argparse
from greenvision.training.mlflow_utils import init_mlflow
from greenvision.training.registry import (
    promote_to_production,
    verify_production_loads,
    list_versions,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True,
                        help="GreenVision model version to promote")
    args = parser.parse_args()

    init_mlflow()
    print("Current versions:", [(v.version, v.current_stage) for v in list_versions()])
    promote_to_production(args.version)
    verify_production_loads()
    print(f"✓ GreenVision v{args.version} → Production")

if __name__ == "__main__":
    main()
```

**Uncertain about.** Locked. Open implementation items:

- Exact `attempt_id` strategy — probably `datetime.now().strftime("%Y%m%d_%H%M%S")` for sortability, or sequential `001`/`002` from a counter file. Either works.
- Whether to log per-class precision/recall as MLflow metrics (in addition to the text report). Would give us per-class trend lines across attempts. Probably worth adding once we have ≥ 3 attempts to compare.

---

## Decision 8 — Serving with FastAPI: I/O contract & inference preprocessing

**What I'm deciding.** Endpoint shape, upload format (multipart vs. base64), and how inference preprocessing matches the validation pipeline.

**Working choice.** ✅ Locked.

**Endpoints**

- `POST /predict` — **multipart/form-data**, field `file`. Primary upload path.
- `POST /predict_b64` — **JSON body** with `image_b64` (base64-encoded image). Secondary path for browser/mobile clients.
- `GET /health` — liveness probe. Reports whether the model is loaded.
- `GET /classes` — returns the 39 class names in model-output order.
- `GET /docs` — FastAPI auto-generates Swagger UI from the Pydantic schemas. Free.

**Response shape** (Pydantic-validated, identical for both prediction endpoints):

```json
{
  "label": "Tomato___Late_blight",
  "confidence": 0.92,
  "top_k": [
    {"label": "Tomato___Late_blight",        "probability": 0.92},
    {"label": "Tomato___Early_blight",       "probability": 0.05},
    {"label": "Tomato___Septoria_leaf_spot", "probability": 0.02}
  ],
  "model_version": "attempt_001",
  "warnings": ["Image appears blurry — please retake in better focus"]
}
```

**Model loading**: FastAPI **lifespan** context. Load checkpoint + class names + device at startup; store on `app.state`. Every request reads from there. Loaded once, used forever.

**Data-quality warnings** (computed per-request, non-blocking):

| Check | Threshold | Warning |
|---|---|---|
| Very small image | `min(w, h) < 64 px` | `"Image is very small ({w}x{h}) — accuracy may be reduced"` |
| Likely blurry | Variance of Laplacian < 100 | `"Image appears blurry — please retake in better focus"` |
| Low confidence | Top-1 probability < 0.5 | `"Model is uncertain ({p:.0%}) — please retake or seek expert opinion"` |

**Error handling**:

| Code | When |
|---|---|
| `400` | Image couldn't be decoded, base64 payload malformed |
| `413` | Upload exceeds 10 MB |
| `422` | Pydantic schema violation (FastAPI automatic) |
| `500` | Unexpected server error — logged, no stack trace leaked |
| `503` | Model not yet loaded (brief startup window) |

**Behavior on non-leaf / low-confidence input** (the two scenarios the rubric explicitly asks about):

| Scenario | What the API returns | What the dashboard renders *(future weeks)* |
|---|---|---|
| **Non-leaf image** (random photo of a car, indoors, etc.) | Model classifies as `Background_without_leaves` (the 39th negative class) → HTTP 200, `label: "Background_without_leaves"`, normal confidence, `top_k` still surfaces alternatives, `warnings: ["No plant detected — please upload a photo of a leaf"]` | Renders a clear "No plant detected" message, hides the class name, shows the upload prompt and a "Retake photo" CTA. `top_k` is not shown because the prediction isn't useful. |
| **Low confidence** (top-1 probability < 0.5 on any class) | HTTP 200 with the predicted label, `top_k` array of alternatives, `warnings: ["Model is uncertain (X%) — please retake or seek expert opinion"]` | Predicted label is rendered in a **muted/grayed** style instead of normal emphasis. The confidence percentage is shown prominently with an explicit warning chip. The `top_k` alternatives are surfaced side-by-side so the user can review them. "Retake photo" becomes the primary CTA; "Use this prediction" is secondary. |
| **Corrupted file** (PIL can't decode) | HTTP 400 with `"Could not decode image"` — no model inference happens | Inline error banner: "We couldn't read that image — try a different file." |

The distinction matters: **non-leaf is a valid prediction result** (the model says "this isn't a leaf"); **corrupted file is a protocol error** (we couldn't even feed it to the model). Different HTTP codes reflect that. Clients that ignore `warnings` still work — the API contract doesn't change based on confidence level.

**Reasoning.**

**Why multipart as primary, base64 as secondary.** Multipart is the native HTML/browser upload format — binary data sent without encoding overhead, supported by curl, `requests`, and any HTML form out of the box. The right default. Base64 adds ~33% payload size from encoding but is the right tool when clients can only send JSON (mobile apps, browsers with data URLs already in memory, middleware that strips binary bodies). Offering both costs ~15 lines and broadens compatibility — also useful for the W9 demo if we hit the API from a simple HTML/JS page.

**Why Pydantic response models.** FastAPI generates OpenAPI/Swagger docs automatically from Pydantic schemas — anyone using the API can open `/docs` and see the exact response shape. The response is validated automatically (we can't accidentally ship malformed JSON), and client SDKs generated from the OpenAPI spec get real types. Three benefits, one annotation.

**Why server-side data-quality warnings.** From Decision 3, we deliberately did not train robustness against blurry photos — the disease signal lives in fine texture, and training on blur would erase it. Easier to require non-blurry input at the boundary. The API enforces this contract by detecting suspicious input (size, blur via variance-of-Laplacian, model uncertainty) and surfacing `warnings`. **The request is not rejected** — the user still gets a prediction — but the UI can show "the photo was blurry, please retake" alongside the result. Clients that ignore `warnings` keep working, so adding/removing checks later isn't a breaking change.

**Why lifespan for model loading.** Module-level loading makes imports slow and errors awkward to handle. Per-request loading would be disastrous (a 100ms model load on every prediction). The lifespan pattern loads once at startup, fails cleanly if the checkpoint is missing or `class_names.json` is malformed, and stores everything on `app.state` for fast per-request access. `/health` can also report "model not loaded" during the brief startup window. Modern FastAPI default.

**Why the 400/413/422/500/503 mapping.** Each status code has one well-understood meaning. 400 covers client-supplied bad data (couldn't decode image, malformed base64). 413 covers oversized uploads — explicit check *before* decoding so we fail fast. 422 is FastAPI's automatic Pydantic-validation error (free with response models). 500 is reserved for true unexpected errors — we wrap the prediction handler to avoid leaking stack traces. 503 covers the rare startup window where the model isn't ready yet.

**Where it shows up in code.**

```python
# api/main.py — sketch
import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from scipy import ndimage

from greenvision.data.transforms import eval_tfms
from greenvision.data.class_names import load_class_names
from greenvision.models.efficientnet_head import build_model

# --- config ---
CHECKPOINT_PATH = Path("artifacts/checkpoints/best.pt")
MAX_FILE_SIZE   = 10 * 1024 * 1024   # 10 MB
TOP_K           = 3
MIN_IMAGE_DIM   = 64
BLUR_THRESHOLD  = 100.0
LOW_CONFIDENCE  = 0.5

# --- Pydantic response models ---
class TopKPrediction(BaseModel):
    label: str
    probability: float = Field(..., ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_k: list[TopKPrediction]
    model_version: str
    warnings: list[str]

class PredictB64Request(BaseModel):
    image_b64: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None

class ClassesResponse(BaseModel):
    classes: list[str]
    count: int

# --- lifespan: load model from MLflow Model Registry at startup (W9A1 revision) ---
import mlflow
import mlflow.pytorch

MLFLOW_TRACKING_URI = "file:./mlruns"
MODEL_URI = "models:/GreenVision/Production"

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    class_names = load_class_names()    # raises if missing or wrong count

    # Load model from the MLflow Model Registry — this is the canonical
    # serving artifact (the local best.pt file is dev-only).
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pytorch.load_model(MODEL_URI)
    model.to(device).eval()

    # Get the version number we actually loaded for the model_version field
    from mlflow.tracking import MlflowClient
    versions = MlflowClient().get_latest_versions("GreenVision", stages=["Production"])
    model_version = f"v{versions[0].version}" if versions else "unknown"

    app.state.device         = device
    app.state.model          = model
    app.state.class_names    = class_names
    app.state.model_version  = model_version
    yield

app = FastAPI(title="GreenVision", lifespan=lifespan)

# --- quality checks ---
def quality_warnings(pil_img: Image.Image, top_prob: float) -> list[str]:
    w, h = pil_img.size
    warnings = []
    if min(w, h) < MIN_IMAGE_DIM:
        warnings.append(f"Image is very small ({w}x{h}) — accuracy may be reduced")
    gray = np.array(pil_img.convert("L"), dtype=np.float64)
    if float(ndimage.laplace(gray).var()) < BLUR_THRESHOLD:
        warnings.append("Image appears blurry — please retake in better focus")
    if top_prob < LOW_CONFIDENCE:
        warnings.append(
            f"Model is uncertain ({top_prob:.0%}) — please retake or seek expert opinion"
        )
    return warnings

# --- shared prediction core ---
def _predict(pil_img: Image.Image, st) -> PredictionResponse:
    x = eval_tfms(pil_img).unsqueeze(0).to(st.device)
    with torch.no_grad():
        probs = F.softmax(st.model(x), dim=1)[0]
    top_p, top_i = probs.topk(TOP_K)
    top_p, top_i = top_p.cpu().tolist(), top_i.cpu().tolist()
    return PredictionResponse(
        label=st.class_names[top_i[0]],
        confidence=top_p[0],
        top_k=[TopKPrediction(label=st.class_names[i], probability=p)
               for i, p in zip(top_i, top_p)],
        model_version=st.model_version,
        warnings=quality_warnings(pil_img, top_p[0]),
    )

# --- endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health(req: Request):
    loaded = hasattr(req.app.state, "model")
    return HealthResponse(
        status="ok" if loaded else "loading",
        model_loaded=loaded,
        model_version=getattr(req.app.state, "model_version", None),
    )

@app.get("/classes", response_model=ClassesResponse)
async def classes(req: Request):
    return ClassesResponse(
        classes=req.app.state.class_names,
        count=len(req.app.state.class_names),
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: Request, file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")
    return _predict(img, req.app.state)

@app.post("/predict_b64", response_model=PredictionResponse)
async def predict_b64(req: Request, body: PredictB64Request):
    try:
        data = base64.b64decode(body.image_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 payload")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, "Decoded image too large")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")
    return _predict(img, req.app.state)
```

**Uncertain about.** Locked. Implementation-time tunings:

- `BLUR_THRESHOLD = 100.0` — tune empirically against PlantVillage's "obviously blurry" samples once we have them.
- `MAX_FILE_SIZE = 10 MB` — raise if real user photos start exceeding it.
- `LOW_CONFIDENCE = 0.5` — tune up if too many predictions get flagged; down if too few warnings reach legitimately uncertain cases.

---

## Resolved during W9A1 implementation

The W9A1 training run resolved most of the "open questions" from the W8A1 design phase. Here's what got empirically validated and what remains open for future runs.

### Resolved (no change needed)

- **Phase 1 epoch count** — locked at **3 epochs**. Val accuracy reached ~93% in Phase 1 alone with no signal that more head-only epochs were warranted. The extension contingency (3 → 5 if val loss still trending down) never fired.
- **Phase 2 unfreezing schedule** — the `(0, 6) → (4, 3) → (8, 0)` schedule (4 epochs per stage) ran cleanly. No instability at any stage transition; no catastrophic forgetting observed.
- **BatchNorm in Phase 2** — default `.train()` mode for unfrozen blocks worked fine. Never needed the "freeze BN" mitigation.
- **`ReduceLROnPlateau` factor/patience (0.5 / 2)** — fired the right number of times without aggressively suppressing learning.
- **Weight decay magnitude (1e-4)** — held val_acc and train_acc within 0.15 pp at every epoch; no overfitting signal that would warrant changing this.
- **Early stopping patience (5)** — Phase 2 ran 21 epochs (5 epochs past the best at epoch 15), triggered early stopping as designed.
- **Vertical flip in augmentation** — kept; per-class recall stayed high across all 39 classes.
- **Uniform sampling vs weighted** — uniform worked; even the smallest class (`Potato___healthy`, 152 images) reached >99% recall. No weighted sampling needed.

### Still open (revisit if conditions change)

- **API-side `BLUR_THRESHOLD` and `LOW_CONFIDENCE` thresholds** — to be tuned during W10A1 against real user photos.
- **File size limit (10 MB)** — to be tuned during W10A1.
- **Out-of-distribution behavior** — the 99.73% test accuracy is *in-distribution* (PlantVillage's studio photos). Field photos would likely drop significantly. The W10A1 serving layer needs UI guidance to keep users on the trained distribution.

## Open questions to resolve during W10A1

All eight decisions are locked and W9A1 training completed cleanly. These are *implementation-time tuning items* — knobs to adjust during W10A1 based on real serving traffic:

- Exact epoch timing for Phase 2a → 2b → 2c transitions (default 4 epochs each).
- Phase 1 length contingency: extend from 3 to 5 if val loss still trending down.
- BatchNorm freeze contingency if Phase 2 destabilizes at a stage transition.
- Weighted sampling fallback if rare-class recall underperforms.
- Vertical-flip toggle if per-class accuracy suffers.
- `ReduceLROnPlateau` factor/patience tuning if LR drops fire too aggressively.
- Weight decay magnitude — tune up if overfitting, down if under-learning.
- Early-stopping patience — tighten if it never fires before MAX_EPOCHS; loosen if it fires too early.
- API-side `BLUR_THRESHOLD` and `LOW_CONFIDENCE` thresholds — tune against real user input.
- File size limit — 10 MB default, raise if needed.

## Sources

References that informed the decisions in this guide. Per-session sources are also tracked in `docs/RESEARCH_LOG.md` under each session's own "Sources" block.

### Papers

- **EfficientNet** (the base model — Decision 1): Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- **AdamW** (Decision 5, W9D2 teaching topic): Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- **PlantVillage dataset** (the data): Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* [arXiv:1511.08060](https://arxiv.org/abs/1511.08060)
- **Effective receptive field** (Session 2.3): Luo, W. et al. (2017). *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks.* [arXiv:1701.04128](https://arxiv.org/abs/1701.04128)

### Documentation

- **PyTorch**: <https://pytorch.org/docs/>
- **torchvision** (`efficientnet_b0`, `ImageFolder`, `transforms`): <https://pytorch.org/vision/>
- **MLflow** (tracking, artifacts, nested runs): <https://mlflow.org/docs/latest/>
- **FastAPI** (serving, lifespan, Pydantic schemas): <https://fastapi.tiangolo.com/>
- **scikit-learn** (`train_test_split`, `confusion_matrix`, `classification_report`): <https://scikit-learn.org/stable/>

### Supplementary explanations

- Cornell course notes on AdamW: <https://optimization.cbe.cornell.edu/index.php?title=AdamW>
- "Recent Improvements to the Adam Optimizer" (iprally): <https://www.iprally.com/news/recent-improvements-to-the-adam-optimizer>
- YouTube walkthrough on AdamW: <https://www.youtube.com/watch?v=PTRk4vNcM-g>

### AI tools used during research

- **Claude** (Opus 4.7) via Cowork — primary research conversation that produced the five sessions in `docs/RESEARCH_LOG.md` and informed every decision in this guide.
- **ChatGPT** — supplementary AdamW deep-dive conversation: <https://chatgpt.com/share/6a14c602-41c8-83e8-a942-7ac00223ccb5>
