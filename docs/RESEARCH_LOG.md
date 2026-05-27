# Research Log — GreenVision

> Working log of research sessions for the W8A1 design phase. The goal isn't to master every topic — it's to build enough understanding to make informed design decisions and to identify what I'll need to learn during implementation.
>
> Rubric expects **3+ sessions**, follow-up questions, and evidence of wrestling with concepts (not first-response copy-paste).

## How to use this file

Each session below uses the same template:

- **Question** — the focused question driving the session
- **Tool used** — Claude / Copilot / docs / blog / paper / etc.
- **Prompt / query** — the exact prompt or query that started the session
- **What I learned** — concrete takeaways in my own words
- **Follow-up questions I asked** — the second/third-order questions, showing I went deeper
- **What's still unclear** — open threads I'll resolve through implementation or further research

---

## Topic coverage tracker

These are the five topic groups from the assignment. I'll check them off as each is addressed by a session below.

- [x] Transfer learning mechanics  *(Session 1 — complete)*
- [x] CNN architectures for image classification  *(Session 2 — complete)*
- [x] Image data pipelines  *(Session 3 — complete)*
- [x] Deep learning training practices  *(Session 4 — complete)*
- [x] Experiment tracking and model serving  *(Session 5 — complete)*

---

## Session 1 — Transfer learning mechanics

**Date:** 2026-05-25
**Duration:** in progress
**Tool used:** Claude (Opus 4.7) via Cowork

### Question

What do each of these four transfer-learning-mechanics sub-topics actually mean, and which ones do I need to go deeper on?

1. What layers of EfficientNet learned what during ImageNet training?
2. Fine-tuning vs. feature extraction vs. training from scratch — when to use each.
3. What "freezing" means in code and what happens during backpropagation when layers are frozen.
4. Why pre-trained layers need lower learning rates than new layers.

### Prompt / query

> Yes the implementation guide looks great. I think we should start from the top. Lets go through, and explain each of the research points, and then when a few stand out, then we can nail into those. start by explaining each of these topics:
>
> **Transfer learning mechanics**
> - What layers of EfficientNet learned what during ImageNet training?
> - Fine-tuning vs. feature extraction vs. training from scratch — when to use each
> - What "freezing" means in code and what happens during backpropagation when layers are frozen
> - Why pre-trained layers need lower learning rates than new layers
>
> I would like you to break down and explain these topics, and also put this outline into the research MD as my first prompt to you. We will outline all of the topics one section at a time, where you explain all of them, and when I want to know more about a given topic (I already understand a few of them) then we can document those as well. This should be a thorough documentation, and I want to spend as much time on the research as possible. I want to really go into this assignment with a deep understanding of the topics. So, document this prompt, and then we can research everything before moving on. I will also watch a few youtube videos on the subjects before class so I feel really confident in my understanding.

### Approach

1. Claude provides a thorough overview of all four sub-topics in chat.
2. I (Quinn) pick the ones I want to go deeper on — the rest get a short summary captured here.
3. Each deep-dive is appended below as its own sub-section with the follow-up prompt, the explanation, code/examples, and what's still unclear.

### Overview — what I learned (filled in as we go)

#### 1.1 What EfficientNet's layers learned during ImageNet training

*Quinn's note:* understood from chat overview, no deep dive needed.

**Key takeaways:**
- CNNs learn a feature **hierarchy** as a consequence of stacking convs: lines → shapes → parts → objects.
- EfficientNet-B0 breaks roughly into three regions:
  - **Early** (stem + first 2 MBConv stages): Gabor-like edges, color blobs, gradients — universal visual primitives, transfer to any image task.
  - **Middle** (MBConv 3–4): textures, corners, repeating patterns — still pretty general.
  - **Late** (MBConv 5–7 + final 1×1 conv expanding to 1280 channels): high-level features specialized to ImageNet classes (object parts).
- The final `Linear(1280, 1000)` is ImageNet-specific and is always thrown away in transfer learning; for GreenVision we replace it with `Linear(1280, 39)` (38 PlantVillage classes + 1 `No_plant_detected` negative class).
- Implication for GreenVision: early/mid layers transfer beautifully to leaves; late layers benefit from fine-tuning to adapt away from ImageNet specifics.

#### 1.2 Fine-tuning vs. feature extraction vs. training from scratch

*Quinn's note:* understood from chat overview, no deep dive needed.

**Key takeaways:**
- A spectrum of "how much pretrained knowledge to keep":
  - **From scratch:** random init, train everything. Needs huge data (think ImageNet's 1.2M images). Use only when domain is so different that ImageNet features don't help (raw medical signals, satellite radar, etc.).
  - **Feature extraction:** freeze entire backbone, train only a new head. Fast, safe for small datasets, can't adapt to domain.
  - **Fine-tuning:** replace head + unfreeze some/all backbone. Best accuracy when you have enough data, but risks catastrophic forgetting if mismanaged.
- GreenVision (54K images, leaves not too far from ImageNet's natural images) → fine-tuning is the right call.
- Our two-phase strategy combines both: Phase 1 *is* feature extraction; Phase 2 *is* fine-tuning. Best of both worlds.

#### 1.3 What "frozen" means in code and what happens during backprop

**Deep dive — this is the one I wanted to crack open.**

##### Follow-up prompt verbatim

> That is great, we are on the same page, I want to drill deeper into 3 for sure. The rest made total sense. I think there is more to learn about 3. First, I am curious why tensor flow would still compute the math for each of the layers if the layer is frozen, and has the requires_grad false? I know we can fix that, but it makes no sense to me why that would be something that was integrated into the code. Is this just a catch, or is there a reason that this was included? Also, is the only metric to claim that a layer is frozen just that we are not performing back propagation on it? So when we say the layer is "frozen" that just means it is already accurate, and does not need to be re assessed the the LR should not be changed? explain both of these concepts in deeper granularity, then we can move onto the next topic.

##### Correction to my earlier explanation

In the chat overview I said "PyTorch still does the math" during backward for frozen layers. That was imprecise and made the design sound worse than it is. **PyTorch's autograd is efficient by default** when `requires_grad=False` is set — it does not do wasted backward work on frozen layers.

##### How autograd actually decides what to track

Autograd is **dynamic and lazy**. It builds the computation graph as the forward pass runs, using a single propagation rule:

> If any input to an operation has `requires_grad=True`, the output also has `requires_grad=True`, and a graph node is created. Otherwise no node is created and the output has `requires_grad=False`.

From the PyTorch docs: *"Backward computation is never performed in the subgraphs where all Tensors didn't require gradients."*

For the typical transfer-learning case (frozen backbone, trainable head, image input that doesn't require grad):

```
x  (image)                requires_grad = False
backbone params           requires_grad = False
↓ backbone forward (each operation has all-False inputs)
each activation           requires_grad = False  → no graph node, activation freed after use
backbone output           requires_grad = False
↓ head forward (head params have requires_grad=True)
each head activation      requires_grad = True   → graph node created, activation retained
loss                      requires_grad = True
loss.backward()           → traverses only the head's portion of the graph
```

So the only "math that still runs" through the frozen backbone is the **forward pass** itself — which is unavoidable, because the head needs the features. There's no wasted backward work, no retained activations, no graph nodes.

##### Why `torch.no_grad()` still exists

Three legitimate use cases — none of them is "fix wasted compute on frozen layers" because that's already not a problem:

1. **Defensive coding.** If the input tensor has `requires_grad=True` (e.g., adversarial training, saliency-map visualization), the propagation rule would extend the graph through the frozen backbone. `no_grad()` blocks that regardless of inputs.
2. **Pre-cached feature extraction.** Run the dataset through the backbone once under `no_grad()`, save the 1280-d feature vectors to disk, then train the head on cached features. Much faster than recomputing every epoch.
3. **Inference.** No `loss.backward()` will happen, so wrap the prediction path in `with torch.no_grad():` to make intent explicit and prevent incidental graph-building.

##### Why `requires_grad` is a tensor-level flag, not a module-level flag

PyTorch's design treats gradients as a property of **tensors**, not modules. A module (`nn.Linear`, `nn.Conv2d`) is just a container of tensors plus a forward function. Putting the flag on the tensor gives you composability — you can freeze a single weight matrix inside a module, or freeze only the bias but not the weight, or freeze every other layer. A "freeze this module" flag would be strictly less expressive and would duplicate machinery that already exists.

##### What "frozen" actually means semantically

"Frozen" is informal terminology. The operational meaning: **these parameters do not change during this phase of training**. The mechanism is `requires_grad=False` → no gradient accumulation → optimizer can't update them.

The connotation in transfer learning is "these are pretrained and useful, leave them alone." But strictly, the freezing mechanism doesn't know or care whether the weights are "good" — you could freeze a randomly initialized layer for an ablation experiment, and PyTorch would oblige.

##### Frozen ≠ low LR — they're different tools

This is the key distinction my prompt was circling:

| | **Frozen** | **Low LR** |
|---|---|---|
| What it is | Binary state: locked | Continuous knob: slow movement |
| Effective LR | 0 | Small but nonzero |
| Over many epochs | Params don't move at all | Params drift gradually |
| Mechanism | `requires_grad = False` | Param group with smaller `lr` |

GreenVision's two-phase strategy uses **both** tools deliberately:

```
Phase 1 — FROZEN backbone
  Backbone: requires_grad = False   → effective LR = 0 (locked)
  Head:     requires_grad = True    → LR = 1e-3

Phase 2 — UNFROZEN backbone with LOW LR
  Backbone: requires_grad = True    → LR = 1e-4
  Head:     requires_grad = True    → LR = 1e-3
```

Phase 1 protects the backbone **absolutely** while the freshly-random head goes through its loud, high-loss settling phase. Phase 2 protects it **proportionally** — now that the head produces reasonable losses, the backbone can gently adapt without being overwritten.

##### Full freeze recipe (the BatchNorm gotcha)

`requires_grad=False` doesn't stop BatchNorm's `running_mean` / `running_var` from updating on each forward pass when the module is in `.train()` mode. EfficientNet uses BN throughout, so this matters. The complete freeze:

```python
def fully_freeze_backbone(model):
    # 1) Stop gradients on backbone params
    for name, p in model.named_parameters():
        if not name.startswith("classifier"):
            p.requires_grad = False
    # 2) Stop BN running stats and dropout from updating
    for name, m in model.named_modules():
        if not name.startswith("classifier") and isinstance(
            m, (torch.nn.BatchNorm2d, torch.nn.Dropout)
        ):
            m.eval()
```

##### Follow-up questions I asked

- Why would PyTorch still compute the math for frozen layers? *(Answer: it doesn't — autograd skips graph construction when no input requires grad. My earlier wording was wrong.)*
- Is "frozen" just "no backprop"? *(Answer: yes for the gradient mechanism, but full freezing also requires `.eval()` on BN/Dropout modules. And "frozen" should not be conflated with "low LR" — they're different tools.)*
- Are "frozen" and "very low LR" interchangeable? *(Answer: no. Frozen = zero movement. Low LR = slow movement that still accumulates over epochs.)*

##### Still unclear / to validate during implementation

- Whether to unfreeze the *entire* backbone in Phase 2 or only the last N MBConv blocks (gradual unfreezing). This is partly empirical — will test during W9–10.
- Exact ratio between head LR and backbone LR in Phase 2 — going to start with 10× (1e-3 vs. 1e-4) and tune.
- Whether to keep BatchNorm in `eval()` mode throughout Phase 2 even after unfreezing the conv weights (some recipes do this; it can help stability on small datasets).

#### 1.4 Why pre-trained layers need lower learning rates than new layers

*Quinn's note:* understood from chat overview, no deep dive needed. The frozen-vs-low-LR distinction in 1.3 also covers most of this.

**Key takeaways:**
- The two parts of the network are in **different regions of parameter space** with different gradient magnitudes.
- Pretrained backbone: already in a "good neighborhood." Same gradient step → useful weights get overwritten ("catastrophic forgetting"). Wants small steps.
- Randomly initialized head: gradients are *large* because loss is high; head needs to move far quickly to find a sane region. Wants big steps.
- A single LR forces a bad tradeoff: too high → backbone forgets; too low → head learns at a crawl.
- Solution: param groups with different LRs. Typical ratio is 10×–100× lower on backbone. Sample code:

```python
optimizer = torch.optim.AdamW([
    {"params": backbone.parameters(), "lr": 1e-4},
    {"params": head.parameters(),     "lr": 1e-3},
], weight_decay=1e-4)
```

- This is why Phase 1 even exists: in the early epochs when head gradients are loudest, *any* LR on the backbone is too high, so we freeze it completely.

### Sources

- Claude (Opus 4.7), conversation 2026-05-25 — initial overview of transfer-learning mechanics.
- (More to add as deep dives happen — torchvision docs, EfficientNet paper, PyTorch autograd docs.)

---

## Session 2 — CNN architectures for image classification

**Date:** 2026-05-25
**Duration:** in progress
**Tool used:** Claude (Opus 4.7) via Cowork

### Question

What do each of these four CNN-architecture sub-topics actually mean, and which ones do I need to go deeper on?

1. How convolutional layers extract spatial features from images.
2. What pooling does and why it's used between conv layers.
3. Receptive field — what it means and why deeper networks "see" larger patterns.
4. How to calculate output dimensions after conv and pool operations.

### Prompt / query

> Yes, great, lets follow the same format as the past two prompts. I would like you to break down these concepts: CNN architectures for image classification
>
> * How convolutional layers extract spatial features from images
> * What pooling does and why it's used between conv layers
> * Receptive field — what it means and why deeper networks "see" larger patterns
> * How to calculate output dimensions after conv and pool operations
>
> I would also like you to log this prompt inside the research log. I will follow up with any deeper questions, and we can continue to research all of these topics.

### Approach

Same as Session 1: Claude provides an overview of all four sub-topics in chat, I pick whichever ones I want to go deeper on, and each deep-dive gets appended below as its own sub-section.

### Overview — what I learned (filled in as we go)

#### 2.1 How convolutional layers extract spatial features

*Quinn's note:* covered in class + chat overview, no deep dive needed.

**Key takeaways:**
- Convolutions slide a small kernel over the input, computing dot products. Two foundational ideas: **parameter sharing** (same weights everywhere — one filter learns one feature that works in any location) and **local connectivity** (each output only sees a small spatial neighborhood).
- Multi-channel input matters: a 3×3 kernel applied to RGB has shape `(3, 3, 3)` — 27 weights per filter — because it integrates across input channels. A conv layer's weight tensor has shape `(out_channels, in_channels, k, k)`.
- Filters aren't programmed; they emerge from backprop. Early-layer filters consistently learn edge detectors, color blobs, Gabor patterns because those are the most useful primitives for any natural-image task.
- Parameter efficiency vs. FC layer: a 3×3 conv with 64 output channels has under 2K weights; a fully-connected layer doing the same job on a 224×224×3 image would need ~9.6M.

#### 2.2 What pooling does and why it's used between conv layers — and a deep dive on downsampling

**Deep dive prompted by my confusion over what "downsampling" actually means in practice.**

##### Follow-up prompt verbatim

> Ok, we went over most of these topics in class already, so I have a general understanding of each of these topics. Your break down helped a lot as well. I have on thing I want to break down before we move onto the next topics. what is the downsampling actually doing in the pooling layer? This did not make sense to me? Break this down, and explain how this is used. this is the section I am referring to: A modern twist worth knowing. EfficientNet doesn't use traditional max pooling between layers. It downsamples with strided convolutions (conv with stride 2) instead. A strided conv does the downsampling and learns useful weights at the same time, where a max pool just throws information away. Max pool is still common in older architectures (ResNet's first layer, VGG throughout), and the conceptual role — downsampling between feature-extraction stages — is identical.

##### What "downsampling" actually means

After a downsampling step you literally have **fewer numbers**, and each remaining number summarizes a region of the input. Analogy: shrinking a 1000×1000 photo to 500×500 — each pixel in the smaller image stands in for a 2×2 block of the original. CNNs do the same thing to feature maps.

##### Worked numerical example — 4×4 → 2×2

Suppose this is a single channel of a feature map. Each number is how strongly that location detected some feature (say "vertical edge"):

```
[1  3  2  4]
[5  8  1  2]
[7  6  9  3]
[2  4  5  1]
```

To downsample to 2×2, the 4×4 is split into four non-overlapping 2×2 windows. Each window collapses to one output number. The *choice* of how to collapse is what distinguishes the two methods.

##### Method A — Max pool (2×2, stride 2)

For each window, take the maximum:

```
[1  3 │ 2  4]                [8  4]
[5  8 │ 1  2]      →
─────────────
[7  6 │ 9  3]                [7  9]
[2  4 │ 5  1]
```

The other three values in each window are discarded — that's what "throws information away" means literally. If the top-left window had values [0.99, 0.95, 0.90, 0.10], max pool outputs 0.99 and the fact that three of four values were high (vs. just one) is lost.

Max pool's real virtue: strong **translation invariance**. If the feature shifts by one pixel inside the window, the max in that window is unchanged. For "did this feature appear *anywhere* in this region?" type questions, that's the right answer.

##### Method B — Strided convolution (stride 2)

A normal conv with stride 1 slides one pixel at a time. A strided conv with stride 2 jumps two pixels each step. The output naturally ends up half-sized — but each output value is a *learned* weighted combination of the values in its window, not just the max:

```
output[i, j] = bias + Σ_{di, dj}  w_{di, dj} × input[2i + di, 2j + dj]
```

Concretely, with learned 2×2 weights `w = [[0.1, 0.4], [0.2, 0.3]]`:

```
top-left  window → 0.1·1 + 0.4·3 + 0.2·5 + 0.3·8 = 4.7
top-right window → 0.1·2 + 0.4·4 + 0.2·1 + 0.3·2 = 2.6
...
```

Every input value contributes (weighted by something the network learned). Nothing is discarded — everything is blended together according to the optimization-discovered combination.

##### The fundamental comparison

|                            | **Max pool**                          | **Strided conv**                              |
|----------------------------|----------------------------------------|-----------------------------------------------|
| How outputs are computed   | Hard-coded: take max                  | Learned: weighted blend                       |
| Information lost?          | Yes — keeps 1 of 4 values             | No — all values contribute                    |
| Learnable parameters       | 0                                     | `k × k × in_channels × out_channels`          |
| Translation invariance     | Strong (by construction)              | Weak (learnable; the network can choose)      |
| Backward pass               | Gradient only to the max position    | Gradient distributed across the window        |
| Can adapt to dataset?      | No                                    | Yes — re-learned during fine-tuning           |

Modern wisdom: when data and compute allow, strided convs almost always outperform max pool. Max pool persists in older architectures or in places where coarse downsampling is "good enough" (e.g., ResNet's initial 3×3 maxpool, where the input is still huge and learnable downsampling at that resolution would cost more than it's worth).

##### Where EfficientNet downsamples (and how)

Every "stride 2" line in this trajectory is a strided conv, *not* a pool:

```
stem 3×3 stride 2:      112 × 112 × 32      ← strided conv
stage 2 (stride 2):      56 ×  56 × 24      ← strided depthwise conv in MBConv
stage 3 (stride 2):      28 ×  28 × 40
stage 4 (stride 2):      14 ×  14 × 80
stage 6 (stride 2):       7 ×   7 × 192
```

Spatial dims halve five times (224 → 112 → 56 → 28 → 14 → 7); channel count grows from 3 → 1280; the final 7×7×1280 collapses to 1×1×1280 via **global average pooling**, which is just "downsample all the way to a single value per channel by averaging." That 1280-d vector is what `Linear(1280, 39)` consumes.

##### Why downsampling helps the rest of the network

- **Compute & memory.** Each conv's cost scales with H×W. Halving spatial dims = 4× cheaper downstream layers. Without downsampling, stacking lots of convs at 224×224 would be prohibitive.
- **Receptive field growth.** Every downsampling step doubles the input region each subsequent unit covers. The deepest layer ends up with an RF that spans the whole 224×224 image — that's what lets it see global patterns like overall leaf shape and color distribution.
- **Feature abstraction.** Some loss of spatial precision is appropriate as you climb the hierarchy. We *want* the deepest layers to think in coarser units ("a leaf-shape exists somewhere here") rather than per-pixel detail.

##### Why this matters for GreenVision

When the model fine-tunes on PlantVillage, EfficientNet's strided convs *re-learn* their downsampling weights to be optimal for distinguishing disease classes. With max pool, the network would be stuck with "max wins" forever — and disease signal often isn't just "where is the brightest pixel" but "what's the spatial pattern of dark spots, and how does that pattern transform when downsampled." Strided convs let the network keep the pattern; max pool would flatten it.

##### Follow-up questions I asked

- What is downsampling actually doing? *(Answer: literally producing a smaller feature map where each value summarizes a region of the input.)*
- How does max pool perform downsampling? *(Answer: takes the max of each non-overlapping window; throws away the other values.)*
- How does a strided conv perform it instead? *(Answer: every output is a learned weighted combination of all values in its window; nothing is discarded; the network learns the best downsampling for the dataset.)*

##### Still unclear / to validate during implementation

- Whether intermediate max-pool layers ever help when fine-tuning a pretrained EfficientNet — probably not (we don't change the architecture), but worth noting that "where to downsample and how" is locked by the base model.
- How global average pooling compares to global max pooling at the very end of the network — average is the modern default, but worth understanding when max would be preferable.

#### 2.3 Receptive field

*Quinn's note:* covered in class + chat overview, no deep dive needed.

**Key takeaways:**
- Receptive field = region of the *input image* that influences a given unit's value in a downstream feature map.
- Stacking 3×3 convs grows RF additively (each adds 2 pixels in each direction); striding/pooling grows it multiplicatively (the "jump" doubles after a stride-2 layer).
- Formulas:
  - `RF_n   = RF_{n-1} + (k_n − 1) × jump_{n-1}`
  - `jump_n = jump_{n-1} × s_n`
- **Theoretical vs. effective RF**: the *theoretical* RF says what input region *could* influence the output; empirically, the *actual* influence is concentrated near the center in a roughly Gaussian shape. The effective RF is usually smaller than the theoretical one — this is why some modern architectures add explicit global-context mechanisms (attention, GAP).
- EfficientNet-B0's deepest layers have theoretical RF that covers the full 224×224 input — that's how it integrates global context to classify whole-leaf patterns.

#### 2.4 Calculating output dimensions after conv and pool

*Quinn's note:* covered in class + chat overview, no deep dive needed.

**Key takeaways:**
- One formula for both conv and pool:
  `output_size = floor((input_size + 2·padding − kernel_size) / stride) + 1`
- `padding = (kernel_size − 1) / 2` with `stride = 1` preserves spatial size ("same" padding).
- Channels: conv freely changes channel count (set by number of filters); pool preserves channel count.
- General CNN pattern: spatial dims shrink as you go deeper, channels grow. Total tensor volume (H × W × C) trends downward overall.
- EfficientNet-B0 spatial trajectory: `224 → 112 → 56 → 28 → 14 → 7`, channels `3 → 32 → 16 → 24 → 40 → 80 → 112 → 192 → 320 → 1280`, then GAP to `1280-d`, then `Linear(1280, 39)`.

### Sources

- Claude (Opus 4.7), conversation 2026-05-25 — initial overview of CNN architectures.
- *(More to add as deep dives happen — EfficientNet paper, PyTorch `nn.Conv2d` / `nn.MaxPool2d` docs, CS231n notes on convolutional networks.)*

---

## Session 3 — Image data pipelines

**Date:** 2026-05-25
**Duration:** in progress
**Tool used:** Claude (Opus 4.7) via Cowork

### Question

What do each of these four image-data-pipeline sub-topics actually mean, and which ones do I need to go deeper on?

1. Data augmentation strategies — which transforms preserve labels, which destroy them.
2. Why normalization matters and what breaks when normalization values are wrong.
3. Train vs. validation transforms — why they must be different.
4. How `ImageFolder` assigns class indices and why that order matters at inference.

### Prompt / query

> Great, yes update the implementation guide to follow downsampling rather than max pooling. That is great, when we were discussing this in class that was something I was really concerned about. I felt like max pooling lost a lot of information. So I am happy there is a better way to handle that. That makes me really happy haha. So thank you for breaking that down. Lets move onto the next section for research: Image data pipelines
>
> * Data augmentation strategies — which transforms preserve labels, which destroy them
> * Why normalization matters and what breaks when normalization values are wrong
> * Train vs. validation transforms — why they must be different
> * How ImageFolder assigns class indices and why that order matters at inference

### Approach

Same pattern as Sessions 1–2: Claude provides an overview of all four sub-topics in chat, I pick whichever I want to go deeper on, deep dives get appended below.

### Overview — what I learned (filled in as we go)

#### 3.1 Data augmentation — which transforms preserve labels, which destroy them

*Quinn's note:* covered in chat overview, no deep dive needed.

**Key takeaways:**
- Fundamental constraint: augmentation must not change the correct label. The transform has to preserve whatever signal the label depends on.
- Label-preserving for leaves: `RandomHorizontalFlip`, `RandomResizedCrop`, small rotations (≤15°), mild `ColorJitter` on brightness/contrast/saturation. Vertical flip is probably safe (leaves don't have a canonical "up").
- Label-destroying: heavy color shifts (can mask discoloration symptoms), extreme blur (destroys fine-texture disease signal), large random erasing (can erase the lesion that the label depends on).
- Baseline augmentation suite for GreenVision is conservative — pull harder on it during implementation if overfitting shows up.
- Modern recipes to know about but not necessarily use: MixUp/CutMix, RandAugment/TrivialAugment, test-time augmentation.

#### 3.2 Why normalization matters and what breaks when it's wrong — deep dive on the math

**Deep dive prompted by wanting the actual derivation, not just the formula.**

##### Follow-up prompt verbatim

> Ok, I want to nail deeper into the averages and standard deviation values. That does not make sense to me from a mathematical standpoint. when you say: normalized_pixel = (pixel - mean) / std
>
> tell me exactly how we are finding these values, and how we could derive this from images in the real dataset. I need to understand this from a pure mathematical standpoint, because the concept makes sense, but I want to know how we gather this.

##### What a pixel is, numerically

A digital RGB image stores each pixel as three integers in `[0, 255]` — one per channel. After `transforms.ToTensor()`, these are divided by 255 to become floats in `[0, 1]`. A whole image is a tensor of shape `(3, H, W)`.

`IMAGENET_MEAN` and `IMAGENET_STD` each have **three values** because they are per-channel statistics — not per-image, not per-pixel — computed across the entire dataset.

##### Per-channel mean formula

For channel `c`:

```
mean_c  =  (1 / N)  ×  Σ pixel_c
```

where `N = M × H × W` (M images, each H × W in size, so N is the total count of pixel positions across the dataset).

For ImageNet at 224×224 with 1.2M images, that's ~60 billion red-channel positions to sum and divide. The answer is `0.485` for red, `0.456` for green, `0.406` for blue.

##### Worked example — one image, red channel

```
R = [[0.2, 0.4],
     [0.6, 0.8]]

mean_R = (0.2 + 0.4 + 0.6 + 0.8) / 4 = 0.5
```

Adding a second 2×2 image to the dataset:

```
R₂ = [[0.0, 0.2],
      [0.4, 0.6]]

mean_R = (0.2 + 0.4 + 0.6 + 0.8 + 0.0 + 0.2 + 0.4 + 0.6) / 8 = 3.2 / 8 = 0.4
```

Same idea — just more data.

##### Per-channel std formula

```
std_c  =  sqrt(  (1 / N)  ×  Σ (pixel_c − mean_c)²  )
```

Sum the squared deviations from the mean, divide by N to get the variance, square-root for the std.

For the single 2×2 example with `mean_R = 0.5`:

```
deviations  = [-0.3, -0.1, +0.1, +0.3]
squared     = [0.09, 0.01, 0.01, 0.09]
sum         = 0.20
variance    = 0.20 / 4 = 0.05
std         = √0.05 ≈ 0.2236
```

##### What normalization does to a single pixel

For the top-left value (`0.2`) using `mean = 0.5`, `std = 0.224`:

```
normalized = (0.2 − 0.5) / 0.224  ≈  −1.34
```

A "slightly below average" red value (0.2 was 0.3 below the mean) becomes "1.34 standard deviations below the mean" after normalization. After applying this to the whole image, values are centered around zero and typically land in roughly `[−2, +2]` — which is the range the pretrained model's internal layers were trained to operate on.

##### Single-pass identity for variance

Naïve formula requires two passes (compute mean first, then deviations). The identity `Var(X) = E[X²] − (E[X])²` lets you compute everything in one pass.

Derivation:

```
Var(X) = E[(X − μ)²]
       = E[X² − 2μX + μ²]
       = E[X²] − 2μ·E[X] + μ²
       = E[X²] − 2μ² + μ²
       = E[X²] − μ²
```

In code:
- `mean = Σx / N`
- `variance = (Σx² / N) − (Σx / N)²`
- `std = √variance`

##### Script to compute mean/std on PlantVillage

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

basic_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),   # uint8 [0, 255] → float [0, 1]
])

dataset = datasets.ImageFolder("data/raw/PlantVillage", transform=basic_tfms)
loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)

n_pixels       = 0
channel_sum    = torch.zeros(3)
channel_sum_sq = torch.zeros(3)

for images, _ in loader:
    b, _, h, w = images.shape
    n_pixels       += b * h * w
    channel_sum    += images.sum(dim=[0, 2, 3])
    channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])

mean = channel_sum / n_pixels
var  = (channel_sum_sq / n_pixels) - mean ** 2
std  = torch.sqrt(var)

print(f"Mean: {mean.tolist()}")
print(f"Std:  {std.tolist()}")
```

##### The critical "but should we?" question

If computing PlantVillage's own mean/std is this easy, why don't we use them?

**Because we're using a pretrained backbone.** EfficientNet's weights, biases, and BatchNorm running statistics were all tuned assuming ImageNet-normalized inputs. Swapping to PlantVillage stats shifts the input distribution and breaks the contract — every layer sees inputs in a different regime than its weights were calibrated for. The cascading effect quietly costs accuracy.

The general rule:
- Using pretrained weights → use the *same normalization stats* the model was trained with (here: ImageNet).
- Training from scratch → compute and use dataset-specific stats.

For GreenVision: ImageNet stats stay locked at `[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`. Running the script above on PlantVillage is a useful sanity exercise, but the values are not what go into `Normalize(...)`.

##### Follow-up questions I asked

- How are mean and std actually computed from a dataset? *(Per-channel sum and sum-of-squares across every pixel of every image, divided by total pixel count.)*
- Why three values? *(One per RGB channel — channels have different distributions in natural photos.)*
- Why are the ImageNet std values all ~0.22? *(Roughly similar contrast distribution per channel in natural photos.)*
- Why not use PlantVillage's own stats? *(They'd break the contract with the pretrained backbone's internal calibration.)*

##### Still unclear / to validate during implementation

- What PlantVillage's actual stats turn out to be — could run the script above for curiosity, would expect something like `[0.47, 0.49, 0.40] / [0.18, 0.15, 0.20]` (greener than ImageNet, less variance because subjects are more uniform).
- Whether some weird PlantVillage images (low light, weird white-balance) sit far outside the ImageNet distribution after normalization — could be a source of accuracy degradation on those images.

#### 3.3 Train vs. validation transforms

*Quinn's note:* covered in chat overview, no deep dive needed.

**Key takeaways:**
- Train transforms include random augmentations (random crop, flip, color jitter) → the model sees different versions of each image each epoch, which improves generalization.
- Validation/inference transforms are deterministic (resize, center crop, normalize) → measurement requires consistency. Random augmentation during eval adds noise to metrics.
- **Inference must match validation *exactly*** — this is the train/inference parity rule. The FastAPI handler imports `eval_tfms` directly from `src/greenvision/data/transforms.py` — no re-implementing preprocessing inside the API.
- Bug to avoid: using `train_tfms` during validation (metrics look worse than the model is) or deterministic transforms during training (overfitting).

#### 3.4 ImageFolder class index assignment

*Quinn's note:* covered in chat overview, no deep dive needed.

**Key takeaways:**
- `ImageFolder` scans the target directory, finds subfolders, and assigns class indices **sorted alphabetically by folder name**.
- Fragile in three ways: filesystem sort order varies across OSes; adding/removing classes shifts every subsequent index; at inference you must turn an output index back into a class name.
- Fix: capture `train_dataset.classes` (the sorted list) as a JSON artifact at training time → load it at inference, look up by index. Never re-derive from filesystem at inference.
- This is the artifact that `agent.md` lists as "do not modify" — desyncing it silently corrupts every prediction the model makes.
- This is implemented as Decision 4 in IMPLEMENTATION_GUIDE.md.

### Sources

- Claude (Opus 4.7), conversation 2026-05-25 — initial overview of image data pipelines.
- *(More to add as deep dives happen — `torchvision.transforms` docs, `ImageFolder` source, ImageNet preprocessing notes.)*

---

## Session 4 — Deep learning training practices

**Date:** 2026-05-25
**Duration:** in progress
**Tool used:** Claude (Opus 4.7) via Cowork

### Question

What do each of these four training-practices sub-topics actually mean, and which ones do I need to go deeper on?

1. How to detect overfitting in a CNN (train vs. val accuracy gaps).
2. What dropout and weight decay do to prevent overfitting.
3. Batch size tradeoffs — memory, speed, gradient stability.
4. When to stop training (early stopping, loss plateau detection).

### Prompt / query

> Oh, that is really cool. That may be the concept that I explain in class on Tuesday. I would like to mark that as one of the topics now, I have a better understanding, but I feel that this is important. So, we may revisit that later on as well. Thank you for that explanation. Make sure that was logged in the research log, and lets move onto the next concept: Deep learning training practices
>
> * How to detect overfitting in a CNN (train vs. val accuracy gaps)
> * What dropout and weight decay do to prevent overfitting
> * Batch size tradeoffs — memory, speed, gradient stability
> * When to stop training (early stopping, loss plateau detection)

### Approach

Same pattern as Sessions 1–3: Claude provides an overview of all four sub-topics in chat, I pick whichever I want to go deeper on, deep dives get appended below. This session directly informs Decision 5 (regularization) and Decision 6 (training loop) in IMPLEMENTATION_GUIDE.md.

### Overview — what I learned (filled in as we go)

#### 4.1 How to detect overfitting in a CNN

*Quinn's note:* covered in chat overview, no deep dive needed.

**Key takeaways:**
- Overfitting = model fits training data well but generalizes poorly to held-out data.
- Diagnostic: plot train vs val accuracy/loss per epoch. Healthy = both improve together with a small gap. Overfitting = train continues climbing while val plateaus or drops.
- Numbers: <5–8% train/val gap is normal; 20% gap means you've overfit.
- Patterns: *gradual* overfitting (slow widening of the gap, common at the tail of training) vs *sudden* overfitting (often Phase 2 with too-high backbone LR — pretrained features get perturbed and val collapses).
- For GreenVision: detect via MLflow per-epoch logging (Decision 7) + early stopping (Decision 6). Plot train and val together every epoch — the visual is more useful than staring at numbers.

#### 4.2 Dropout and weight decay — deep dive on AdamW vs. Adam

**Deep dive prompted by wanting the actual math behind "decoupled weight decay" and why it matters.**

##### Follow-up prompt verbatim

> Great, make sure the last prompt was documented, and that this on is as well. I would like to dig into the weight decay. I want to know more about AdamW vs Adam, and the math behind it. tell me how these work, and the trade offs between them. This could be another topic I could discuss in class, because it seems very interesting as well. lets dig into this, and then we can move onto the last section.

##### SGD with L2 — the baseline that works

L2 regularization adds a penalty to the loss:

```
L_total = L_task + (λ/2) × Σ w²
```

Take the gradient with respect to a weight `w`:

```
∂L_total/∂w = ∂L_task/∂w + λ × w
```

SGD update:

```
w_t = w_{t−1} − lr × ∂L_task/∂w − lr × λ × w_{t−1}
```

Two operations: a normal gradient step plus a *shrinkage* term `−lr × λ × w_{t−1}`.

**Key fact:** with plain SGD, "L2 in the loss" and "weight decay in the update" are mathematically equivalent. Both produce the same `−lr × λ × w` subtraction every step. This is the equivalence that creates all the confusion later.

##### Adam's mechanics

Adam maintains running averages of gradients and squared gradients, and uses them to scale each parameter's update individually:

```
g_t = ∂L/∂w
m_t = β₁ × m_{t−1} + (1−β₁) × g_t       # first moment (running mean of gradients)
v_t = β₂ × v_{t−1} + (1−β₂) × g_t²      # second moment (running mean of squared grad)

m̂_t = m_t / (1 − β₁^t)                   # bias correction (matters for early iterations)
v̂_t = v_t / (1 − β₂^t)

w_t = w_{t−1} − lr × m̂_t / (√v̂_t + ε)
```

The denominator `√v̂_t + ε` is the source of Adam's adaptivity. Each parameter's effective learning rate becomes `lr / (√v̂_t + ε)`:
- Large historical gradients → big `v̂_t` → small effective LR.
- Small historical gradients → small `v̂_t` → large effective LR.

This is why Adam works well across many tasks. It's also the source of the weight-decay bug.

##### Adam + L2 — where it breaks

What if you add L2 to plain Adam? The L2 term gets added to the gradient:

```
g_t = ∂L_task/∂w + λ × w_{t−1}
```

But now this gradient flows through Adam's machinery. The `λw` term accumulates into `m_t` and `v_t`, then gets divided by `√v̂_t + ε` in the update. The **effective decay** becomes:

```
effective_decay = lr × (λ × w_{t−1}) / (√v̂_t + ε)
```

That denominator is the bug. Weights with large historical gradients get *less* regularization. Weights with small historical gradients get *more*. The regularization strength is now modulated by gradient history — which has no theoretical justification and is the opposite of what L2 was supposed to do (apply uniform pressure toward smaller weights).

Worse: the weights with active task gradients are often the ones *most at risk of overfitting* — and they're the ones getting the least regularization.

##### AdamW — the fix

Loshchilov & Hutter (2017/2019) proposed: never put the L2 term into the gradient at all. Apply weight decay as a *separate* operation that happens alongside the Adam update:

```
g_t = ∂L_task/∂w                         # ONLY the task gradient
m_t = β₁ × m_{t−1} + (1−β₁) × g_t
v_t = β₂ × v_{t−1} + (1−β₂) × g_t²
m̂_t = m_t / (1 − β₁^t)
v̂_t = v_t / (1 − β₂^t)

w_t = w_{t−1} − lr × (m̂_t / (√v̂_t + ε) + λ × w_{t−1})
```

The fix is one line. The Adam update `lr × m̂_t / (√v̂_t + ε)` and the weight decay `lr × λ × w_{t−1}` are now *added*, not multiplied through Adam's denominator. Every weight gets the same proportional shrinkage `lr × λ` regardless of gradient history.

##### Numerical example — the bug in action

Two weights, identical value (1.0), identical λ (0.01), lr = 0.001, but different gradient histories:

- **w₁:** large historical gradients → `√v̂_t + ε ≈ 1.0`
- **w₂:** small historical gradients → `√v̂_t + ε ≈ 0.1`

**Adam-with-L2** effective decay per step (`lr × λ × w / (√v̂_t + ε)`):

```
w₁:  0.001 × 0.01 × 1.0 / 1.0   =  0.00001    ← small decay
w₂:  0.001 × 0.01 × 1.0 / 0.1   =  0.00010    ← 10× larger decay
```

w₂ is regularized 10× harder than w₁ despite identical λ.

**AdamW** effective decay per step (`lr × λ × w`):

```
w₁:  0.001 × 0.01 × 1.0  =  0.00001
w₂:  0.001 × 0.01 × 1.0  =  0.00001
```

Identical and uniform.

##### LR coupling subtlety

In PyTorch's `AdamW`, weight decay is multiplied by `lr`. So doubling LR also doubles effective weight decay. The original AdamW paper proposed decoupling λ from `lr` entirely:

```
w_t = w_{t−1} − lr × m̂_t / (√v̂_t + ε) − λ_decoupled × w_{t−1}
```

Both formulations are called "AdamW" in the wild. PyTorch's convention is the one we use; it means weight decay scales with LR, which is usually fine but worth knowing.

##### Trade-offs (honest version)

AdamW is *strictly better* than plain Adam whenever you want regularization. Same compute, same memory. The "trade-off" framing is misleading — AdamW is a fix, not a variant.

Plain Adam is fine in one case: `λ = 0` (no weight decay). Then there's nothing to decouple and the two are mathematically identical.

##### PyTorch implementation specifics

This trips people up. Both optimizers take the same `weight_decay` parameter but implement different math:

```python
torch.optim.Adam(params,  lr=1e-3, weight_decay=1e-4)
# → applies weight decay as L2 added to the gradient (the broken path)

torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
# → applies decoupled weight decay (the correct path)
```

If you wrote `weight_decay=1e-4` thinking you were doing "AdamW-style" regularization but you used `torch.optim.Adam`, you're using the broken implementation.

##### Connection to GreenVision

Decision 5 in IMPLEMENTATION_GUIDE.md pins AdamW for exactly this reason:

```python
optimizer = torch.optim.AdamW([
    {"params": backbone.parameters(), "lr": 1e-4},
    {"params": head.parameters(),     "lr": 1e-3},
], weight_decay=1e-4)
```

Three things at once: (a) `AdamW` so weight decay is uniform across all 5.3M parameters; (b) param groups with different LRs to protect pretrained backbone (from §1.3/1.4); (c) `weight_decay=1e-4` applied via AdamW's decoupling.

Because PyTorch's AdamW multiplies `λ × lr`, the backbone's effective weight decay ends up 10× smaller than the head's (since its LR is 10× smaller). That happens to be roughly the right behavior — gentler treatment of pretrained weights — but it's a *side effect* of the implementation rather than something I chose explicitly.

##### Follow-up questions I asked

- What does Adam actually do differently from SGD? *(Adam maintains per-parameter running averages of gradients and squared gradients, scaling each update by `1 / √v̂_t + ε` — so each parameter gets its own effective learning rate.)*
- Why does adding L2 to Adam not work? *(The L2 term gets passed through Adam's denominator, so regularization strength becomes a function of gradient history — uniform regularization is no longer uniform.)*
- What's the fix? *(Apply weight decay as a separate operation outside the Adam machinery — that's AdamW.)*
- When would I ever use plain Adam? *(When weight_decay=0. Otherwise: don't.)*

##### Still unclear / to validate during implementation

- Whether `1e-4` is the right `weight_decay` for our scale — going to start there and tune.
- Whether to actually exclude bias and BatchNorm params from weight decay (a common practice — some frameworks do this automatically; PyTorch's AdamW does not).
- The actual empirical impact of switching from Adam to AdamW on PlantVillage — worth a quick comparison experiment during W9–10.

#### 4.3 Batch size tradeoffs

*Quinn's note:* covered in chat overview, no deep dive needed.

**Key takeaways:**
- Batch size affects three things in tension: **memory** (bigger = more), **speed** (bigger = better GPU utilization, fewer batches per epoch), **gradient stability** (bigger = lower variance).
- Counter-intuitively, *some* gradient noise helps generalization. Small-batch noise pushes the optimizer toward flat minima, which test better than the sharp minima large batches tend toward.
- **Linear scaling rule:** doubling batch size → roughly double LR to maintain effective step size. Less critical for fine-tuning where you're making small adjustments.
- For GreenVision: 32 on CPU/local, 64–128 on GPU with mixed precision if memory allows. Start at 32 as the safe baseline.

#### 4.4 When to stop training

*Quinn's note:* covered in chat overview, no deep dive needed.

**Key takeaways:**
- **Early stopping:** monitor val metric, stop after `patience` consecutive epochs of no improvement. Patience of 3 is aggressive; 10 is patient. For fine-tuning, 3 is typical.
- Always **checkpoint the best model** along the way, separate from when you stop — `if val_acc > best_val_acc: save_checkpoint(...)`.
- **Val accuracy vs. val loss:** accuracy matches the user-facing goal but is noisier; loss is smoother but can plateau while accuracy stays flat. Most classification pipelines monitor val accuracy.
- **Loss plateau detection** (e.g. `ReduceLROnPlateau`) is related but different — reduces LR when loss stagnates instead of (or before) stopping. Sometimes lets training resume making progress at lower LR.
- For GreenVision: Phase 1 = 3–5 epochs no early stopping (short enough to converge cleanly); Phase 2 = up to 15 epochs with early stopping, patience 3 on val accuracy. Best checkpoint to `artifacts/checkpoints/best.pt` whenever val improves.

### Sources

- Claude (Opus 4.7), conversation 2026-05-25 — initial overview of training practices.
- *(More to add as deep dives happen — PyTorch `optim` docs, AdamW paper, "Don't Decay the Learning Rate, Increase the Batch Size" / batch-size papers, early-stopping references.)*

---

## Session 5 — Experiment tracking and model serving

**Date:** 2026-05-25
**Duration:** ~30 min
**Tool used:** Claude (Opus 4.7) via Cowork

### Question

What do each of these four tracking-and-serving sub-topics actually mean? This section feels the most intuitive — covered in class, plus I've used FastAPI before — so I want a solid overview but no deep dives.

1. How to log CNN training runs to MLflow (metrics per epoch, model artifacts).
2. How to save and load PyTorch models correctly.
3. Image preprocessing at inference — why it must match validation preprocessing exactly.
4. FastAPI image upload handling (multipart vs. base64).

### Prompt / query

> Great, I think we found our candidate, this is complex, it is interesting, and I have more questions than answers after reading that haha. So lets move onto the last one, then we can dig really deep into this so I can explain it to the class. This last one seems pretty intuitive. I think I could figure this out on my own. Lets break this down, and finalize our research log. I think we have a total of 5 sessions after this one, so we are over the requirements for the assignment. But I want to keep digging. I think this will lead to a better overall project, and can get me a better understanding of this interesting subject. This is the last portion: Experiment tracking and model serving
>
> * How to log CNN training runs to MLflow (metrics per epoch, model artifacts)
> * How to save and load PyTorch models correctly
> * Image preprocessing at inference — why it must match validation preprocessing exactly
> * FastAPI image upload handling (multipart vs. base64)
>
> Make sure everything is logged, and then we can go right into our deep dive on adam vs adamw or come up with a better phrasing for this subject I am going to teach on Tuesday.

### Approach

Overview-only session — no deep dives. Notes captured below directly from the chat walkthrough.

### Overview — what I learned

#### 5.1 How to log CNN training runs to MLflow

*Quinn's note:* covered in chat overview, no deep dive needed.

**Key takeaways:**
- MLflow vocabulary: **experiment** (named group of runs), **run** (single training attempt), **params** (logged once per run — LR, batch size, optimizer, etc.), **metrics** (logged with `step=epoch`, e.g. train/val loss & accuracy), **artifacts** (files: checkpoints, plots, `class_names.json`), **tracking URI** (where the data lives; `file:./artifacts/mlruns` keeps it portable in the repo).
- Skeleton:
  ```python
  mlflow.set_tracking_uri("file:./artifacts/mlruns")
  mlflow.set_experiment("greenvision")
  with mlflow.start_run(run_name="phase2"):
      mlflow.log_params({"lr_head": 1e-3, "lr_backbone": 1e-4, "wd": 1e-4, ...})
      for epoch in range(num_epochs):
          # train + eval
          mlflow.log_metrics({"train_loss": ..., "val_acc": ...}, step=epoch)
      mlflow.log_artifact("artifacts/checkpoints/best.pt")
      mlflow.log_artifact("artifacts/checkpoints/class_names.json")
  ```
- View runs locally: `mlflow ui --backend-store-uri file:./artifacts/mlruns`.
- Open design choice: nested runs (parent → `phase1` / `phase2` children) vs. two flat runs. Implementation guide leaves this as `TBD`.
- `mlflow.pytorch.autolog()` exists, but I prefer explicit logging for control over what gets captured.

#### 5.2 How to save and load PyTorch models correctly

*Quinn's note:* covered in chat overview, no deep dive needed.

**Key takeaways:**
- **Save state_dict, not the whole model.** The whole-model save (`torch.save(model, "x.pt")`) pickles the Python class definition and breaks if you refactor. State_dict approach decouples weights from architecture and is the recommended pattern.
- Save a richer **checkpoint dict** with metadata for resumability:
  ```python
  torch.save({
      "model_state_dict":     model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "epoch":                epoch,
      "val_acc":              best_val_acc,
      "num_classes":          39,
      "arch":                 "efficientnet_b0",
  }, "artifacts/checkpoints/best.pt")
  ```
- At load time:
  ```python
  ckpt  = torch.load("artifacts/checkpoints/best.pt", map_location=device)
  model = build_model(num_classes=ckpt["num_classes"])
  model.load_state_dict(ckpt["model_state_dict"])
  model.to(device).eval()
  ```
- Critical at inference: `model.eval()` (turn off dropout, freeze BN stats), `with torch.no_grad():` (skip graph construction), and `map_location=device` (handle CPU/GPU mismatch).
- MLflow integration: `mlflow.log_artifact("best.pt")` to keep it decoupled, or `mlflow.pytorch.log_model(...)` for an MLflow-native package. Decoupled is simpler for FastAPI to load.

#### 5.3 Image preprocessing at inference — why it must match validation preprocessing exactly

*Quinn's note:* covered in chat overview, no deep dive needed. This is also Decision 8 in IMPLEMENTATION_GUIDE.md.

**Key takeaways:**
- A trained model is a function from a **specific input distribution** to predictions. Any drift in that distribution at inference time silently degrades predictions — no error, no warning.
- Most common drifts: wrong normalization stats (e.g., ImageNet at train, raw `[0,1]` at inference), different resize/crop (224 vs. 256), RGB vs. BGR mismatch (PIL ↔ OpenCV), grayscale or RGBA images sneaking through without `.convert("RGB")`.
- The architectural rule: **import the same `eval_tfms` object from `src/greenvision/data/transforms.py`** in both validation and inference. Do not re-implement preprocessing inside `api/`.
- Defensive details: always `.convert("RGB")` on PIL input to normalize alpha/grayscale; validate uploaded file format; validate dimensions before passing to the model.

#### 5.4 FastAPI image upload handling — multipart vs. base64

*Quinn's note:* covered in chat overview, no deep dive needed. This finalizes Decision 8.

**Key takeaways:**
- **Multipart/form-data:** native HTML/browser upload format, binary data sent as-is, most efficient. The default choice.
  ```python
  from fastapi import FastAPI, File, UploadFile, HTTPException
  from PIL import Image
  import io

  @app.post("/predict")
  async def predict(file: UploadFile = File(...)):
      try:
          img = Image.open(io.BytesIO(await file.read())).convert("RGB")
      except Exception:
          raise HTTPException(status_code=400, detail="Could not read image")
      x = eval_tfms(img).unsqueeze(0).to(device)
      # ... predict
  ```
- **Base64:** image encoded as ASCII inside a JSON body. ~33% larger payload than raw bytes but portable across systems that only handle JSON (mobile clients, web apps that already have data URLs, proxies that strip binary).
  ```python
  class PredictB64Request(BaseModel):
      image_b64: str

  @app.post("/predict_b64")
  async def predict_b64(req: PredictB64Request):
      img = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
      # ... predict
  ```
- **For GreenVision:** multipart is primary (`POST /predict`). Add a base64 endpoint later if browser clients need it.
- Other API concerns: `GET /health` for load-balancer probes; `GET /classes` to expose the 39 class names; HTTP 400 for bad uploads (never leak stack traces); FastAPI's auto-generated `/docs` (Swagger UI) is free.

### Sources

- Claude (Opus 4.7), conversation 2026-05-25 — overview of tracking and serving.
- *(More to add if I run into specifics during W9–10 — MLflow docs, FastAPI docs, `mlflow.pytorch.log_model` reference.)*

---

## Research summary — what's left to do during implementation

All five topic groups now have either a deep-dive write-up or a key-takeaways summary in this log. The eight design decisions in `IMPLEMENTATION_GUIDE.md` are all locked. What follows are *empirical tunings* to validate during W9–10 implementation — knobs to adjust based on real MLflow signal, not unresolved research questions.

- **Session 1 (Transfer learning mechanics):** all locked. Phase 1 = 3 epochs (extend to 5 only if val loss is still trending down). Phase 2 uses *gradual unfreezing* (Phase 2a/2b/2c). BN runs in `train()` for unfrozen blocks and `eval()` for frozen blocks. Catastrophic-forgetting contingency: delay next unfreeze stage or switch BN to eval-only if val accuracy crashes at a stage transition.
- **Session 2 (CNN architectures):** none — these are properties of the architecture I'm not changing.
- **Session 3 (Image data pipelines):** all locked. Open knobs: whether vertical flip helps (drop if per-class accuracy suffers); whether to add weighted sampling if rare-class recall underperforms after Phase 2; whether PlantVillage's own stats land far from ImageNet's (sanity exercise — don't use the values).
- **Session 4 (Training practices):** all locked. AdamW with `weight_decay=1e-4`, bias/BN excluded from decay, `ReduceLROnPlateau` (factor 0.5, patience 2), early-stopping patience 5 on val accuracy. Open knobs: empirical AdamW-vs-Adam comparison run (sanity check); weight-decay magnitude tuning if overfitting persists; patience adjustment if early stopping fires too early or never.
- **Session 5 (Tracking & serving):** all locked. **Nested** MLflow runs (parent + phase1/phase2 children). Multipart primary + base64 secondary FastAPI endpoints. Open knobs: API-side `BLUR_THRESHOLD` and `LOW_CONFIDENCE` thresholds tuned against real photos; file size limit raised if needed.

---

## Concept I'd teach (W9D2)

### Chosen topic ⭐ — AdamW vs. Adam: the weight-decay bug (from Session 4)

- **Candidate concept:** Why we use `torch.optim.AdamW` instead of `torch.optim.Adam` — the subtle but real bug in how plain Adam handles weight decay, and how AdamW's decoupled formulation fixes it.
- **Why this one is the right pick:** It's a small piece of optimization theory with a clear "before/after" story. The bug was in widespread use for years; the fix is one line of code; the empirical impact is measurable. It's the kind of detail that separates "I followed the tutorial" from "I understand what's happening under the hood." The math is short enough to fit in 12 minutes and surprising enough to land.
- **The "what confused me / what clicked" arc:**
  - I started by knowing "use AdamW, not Adam" as a rule of thumb, but couldn't articulate *why*.
  - It clicked when I derived from SGD-with-L2 first — seeing that L2 in the loss and weight decay in the update are mathematically equivalent in that setting (both produce a `−lr × λ × w` shrinkage).
  - Then watching Adam's adaptive denominator `√v̂_t + ε` break that equivalence by scaling the L2 term differently for each parameter based on its gradient history.
  - The numerical example with two weights (same value, same λ, different gradient histories) where Adam-with-L2 regularizes them 10× differently while AdamW regularizes them uniformly.
- **Where it shows up in GreenVision:**
  - `torch.optim.AdamW(..., weight_decay=1e-4)` in Decision 5 of IMPLEMENTATION_GUIDE.md.
  - The "W" matters. `torch.optim.Adam` with the same `weight_decay` argument uses the broken L2 path silently.
- **Planned 12-min structure (draft — needs refinement):**
  - Slide 1: SGD-with-L2 baseline (the math that works).
  - Slide 2: Adam's adaptive update rule (the `1 / (√v̂_t + ε)` per-parameter scaling).
  - Slide 3: Where L2 fails in Adam — with the two-weight numerical example showing the 10× regularization disparity.
  - Slide 4: AdamW — the decoupling fix.
  - Closer: the PyTorch one-letter difference, and why this maps directly to a real choice in our GreenVision optimizer.

### Alternate A — ImageNet normalization: where the magic numbers come from (from Session 3)

- The mean/std derivation could pivot to be the teaching topic if the AdamW math feels too dense for the audience. The worked 2×2 example and the "Var = E[X²] − (E[X])²" running-sum identity are pedagogically clean.
- Full notes still live in §3.2.

### Alternate B — Freezing & catastrophic forgetting (from Session 1)

- Third option if neither of the above lands — the freezing-vs-low-LR distinction from §1.3 + §1.4. Also a strong "what confused me, what clicked" story (PyTorch's autograd is efficient by default with `requires_grad=False`; freezing and low LR are different tools that the two-phase strategy uses in sequence; the BatchNorm `.eval()` gotcha).
