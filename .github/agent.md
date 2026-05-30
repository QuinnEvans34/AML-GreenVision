# Copilot Agent Guardrails — GreenVision

> Rules of the road for Copilot Agent (and any autonomous AI assistant) operating in this repo. Read this before doing anything that touches multiple files or runs commands.
>
> When in doubt: stop and ask. A confirmed change is worth ten "I think this is what you wanted" changes.

## What Agent CAN do autonomously

Agent may go ahead with these without checking in:

- **Generate boilerplate**
  - `DataLoader` setup (instantiating `ImageFolder`, wrapping with `DataLoader`, choosing `num_workers`).
  - Standard training-loop skeletons (forward pass, loss, backward, optimizer step, metric accumulation).
  - Evaluation/validation loops that mirror the training loop's structure.
- **Write docstrings** in the established style (Google-style; Args / Returns / Raises for non-trivial functions).
- **Suggest augmentation transforms** that are label-preserving for leaves (random crops, horizontal flips, mild color jitter, small rotations). Risky transforms should be proposed, not silently committed.
- **Scaffold FastAPI endpoints** following the existing `/predict`, `/health`, `/classes` pattern — handler signature, dependency injection, request/response models with Pydantic.
- **Add type hints** to under-typed functions.
- **Fix obvious bugs** like off-by-one errors, missing `model.eval()` calls before validation, or forgetting `with torch.no_grad():` around inference.
- **Add tests** under `tests/` that exercise existing public functions.
- **Update `requirements.txt`** to add a needed dependency *if* the dependency is already imported in code that won't run without it (i.e., closing a gap, not adding new functionality).

## What Agent MUST NOT do

These are hard stops. If a task seems to require any of them, **pause and ask the user first**, even if the change looks tiny.

- **Do not change the ImageNet normalization values.** `IMAGENET_MEAN = [0.485, 0.456, 0.406]` and `IMAGENET_STD = [0.229, 0.224, 0.225]` are locked. They match the values EfficientNet-B0 was pretrained with. Changing them silently corrupts every prediction the model will ever make.
- **Do not modify the class names artifact (`artifacts/checkpoints/class_names.json`) structure or generation logic.** That file is the contract between the model's output index and the human-readable label. Reordering it, regenerating it from a different source, or "cleaning up" the names will desync predictions from labels in a way that won't throw an error.
- **Do not alter the two-phase training sequence.** Phase 1 (head-only, backbone frozen) must run before Phase 2 (full fine-tune with parameter-group LRs). Don't merge them into a single phase, don't reorder them, don't add a third phase without confirmation.
- **Do not change the freeze/unfreeze logic** (which params get `requires_grad = False`) without confirmation.
- **Do not change learning rates, batch size, or epoch counts** without confirmation. These are decisions the user is actively reasoning about and tracking.
- **Do not remove error handling** — `try/except` blocks around image decoding, file I/O around checkpoints, HTTP 400/404/500 responses in API endpoints. If you think an error handler is dead code, ask before deleting.
- **Do not skip MLflow logging** to "make the code cleaner." Every training run gets logged.
- **Do not promote, archive, transition, or delete versions of the `GreenVision` registered model** in the MLflow Registry. The Production stage is the contract the W10A1 serving layer reads from. Stage transitions are deliberate human decisions made via `scripts/promote.py` — not Agent calls.
- **Do not delete `mlruns/` or any sub-tree of it.** The file store backs both the experiment tracking AND the model registry. Deleting `mlruns/models/GreenVision/` removes the Production model the serving layer depends on.
- **Do not change the registered model name** (`GreenVision`). It is hardcoded into `mlflow.pytorch.log_model(..., registered_model_name="GreenVision")` and the load URI `models:/GreenVision/Production`. Renaming silently desyncs every deployed serving instance.
- **Do not re-implement preprocessing inside the API.** The FastAPI handler imports `eval_tfms` from `src/greenvision/data/transforms.py`. If you find yourself writing `transforms.Resize(...)` inside `api/`, you're doing the wrong thing.
- **Do not commit data, model checkpoints, or MLflow runs** — `.gitignore` excludes them; don't override it.
- **Do not introduce new dependencies** for functionality that could reasonably be written with what's already in `requirements.txt`.

## Files Agent should not modify without confirmation

Even for "small" changes, ask first before editing:

- `.github/copilot-instructions.md` — the project's source of truth for AI behavior. Edits here change how every other AI interaction in this repo behaves.
- `.github/agent.md` — this file. Self-modifying guardrails defeat the point.
- `IMPLEMENTATION_GUIDE.md` — the user's design document. Agent can read it to understand decisions, but should not edit it.
- `ASSIGNMENT.md` — assignment brief; immutable reference.
- `artifacts/checkpoints/**` — trained model weights and the `class_names.json` artifact.
- `mlruns/**` — MLflow run history AND model registry (file-backed). Includes `mlruns/models/GreenVision/` which holds the registered model and its stage assignments.
- `scripts/promote.py` — the only sanctioned path for stage transitions. Don't change the script's behavior to auto-promote on training completion.
- `data/raw/**` and `data/processed/**` — the dataset.
- `requirements.txt` — beyond the narrow "close a gap" case above.

## When in doubt

Surface the proposed change as a diff or a summary and wait for a yes before applying it. The user would rather review a one-line proposal than discover a silent change to a normalization constant three weeks later.
