# [W8A1] GreenVision Research & Implementation Design

**Due:** Wednesday by 11:59pm
**Points:** 50
**Available until:** Jun 3 at 11:59pm

---

## Overview

You're building a plant disease classifier using **EfficientNet-B0** on **PlantVillage**. Before writing any implementation code, you need to design it — and to design it well, you need to research how CNNs, transfer learning, and image classification pipelines actually work.

This assignment has three outputs: research that builds your understanding, a concept you teach the class, and an implementation guide that becomes your build roadmap for Weeks 9–10.

### Architecture non-negotiables

- **Base model:** EfficientNet-B0 — pre-trained on ImageNet
- **Dataset:** PlantVillage — 54,306 leaf images, 38 disease and healthy classes, 14 crops
- **Experiment tracking:** MLflow
- **Serving:** FastAPI

> Reference: `GreenVision Research & Implementation Design - Hints.docx`

---

## Instructions

### Research

Explore the topics below using AI tools, documentation, and tutorials. Keep a research log documenting what you learned and what's still unclear. Aim for **3–5 focused research sessions over the 4 days**.

#### Topics to explore

**Transfer learning mechanics**
- What layers of EfficientNet learned what during ImageNet training?
- Fine-tuning vs. feature extraction vs. training from scratch — when to use each
- What "freezing" means in code and what happens during backpropagation when layers are frozen
- Why pre-trained layers need lower learning rates than new layers

**CNN architectures for image classification**
- How convolutional layers extract spatial features from images
- What pooling does and why it's used between conv layers
- Receptive field — what it means and why deeper networks "see" larger patterns
- How to calculate output dimensions after conv and pool operations

**Image data pipelines**
- Data augmentation strategies — which transforms preserve labels, which destroy them
- Why normalization matters and what breaks when normalization values are wrong
- Train vs. validation transforms — why they must be different
- How `ImageFolder` assigns class indices and why that order matters at inference

**Deep learning training practices**
- How to detect overfitting in a CNN (train vs. val accuracy gaps)
- What dropout and weight decay do to prevent overfitting
- Batch size tradeoffs — memory, speed, gradient stability
- When to stop training (early stopping, loss plateau detection)

**Experiment tracking and model serving**
- How to log CNN training runs to MLflow (metrics per epoch, model artifacts)
- How to save and load PyTorch models correctly
- Image preprocessing at inference — why it must match validation preprocessing exactly
- FastAPI image upload handling (multipart vs. base64)

> You don't need to master every topic — research until you have enough clarity to make informed design decisions and identify what you'll learn through implementation.

---

### Teach one concept (W9D2, 12 min)

Choose one concept from your research that you found particularly hard to understand at first — and that you now feel you can explain clearly.

Your 12-minute teaching segment must include:

- The concept explained in plain language (no jargon without definitions)
- One concrete example, code snippet, or diagram
- Why it matters for GreenVision specifically
- What clarified it for you during research

> This is not a polished lecture — it's showing your learning process. "Here's what confused me, here's what finally clicked, here's how I'd explain it now."

---

### Write your implementation guide (submit before W9D3)

`IMPLEMENTATION_GUIDE.md` in your `aml-greenvision` repo. Address these decisions:

#### 1. `IMPLEMENTATION_GUIDE.md` — your design document

Be honest about uncertainty. "I'm testing 3 vs. 5 epochs for Phase 1 during implementation" is better than guessing. The guide will evolve — that's expected.

#### 2. `.github/copilot-instructions.md` — AI context for your repo

This file tells Copilot (and future you) what the project is, what constraints matter, and what conventions to follow.

**Required sections:**
- **Project description:** What GreenVision does, who uses it, what it predicts
- **Dataset details:** PlantVillage structure, 38 classes, ImageFolder format, class naming convention
- **Critical constants:** ImageNet normalization values, image size (224), number of classes (38), EfficientNet feature dimension (1280)
- **Code conventions:** Import style, docstring requirements, error handling patterns
- **Architecture notes:** Two-phase fine-tuning strategy, what gets frozen when, learning rates for each phase

#### 3. `.github/agent.md` — Copilot Agent guardrails

Define what Copilot Agent can and cannot do autonomously in your repo.

**Required sections:**
- **What Agent CAN do:** Generate boilerplate (DataLoader setup, training loop structure), write docstrings, suggest augmentation transforms, scaffold FastAPI endpoints
- **What Agent MUST NOT do:** Change ImageNet normalization values, modify the class names artifact structure, alter the two-phase training sequence, remove error handling
- **Files Agent should not modify without confirmation:** `copilot-instructions.md`, model checkpoint files, MLflow artifacts

> All three documents are living. Update them as you learn more during implementation.

---

### Present your implementation guide (W9D3, 8 min + 4 min Q&A)

Walk the class through your design decisions. Structure:

- **Choices (7 min):** Pick your 2-3 strongest design & architecture decisions — explain the reasoning and where they show up in code
- **Open questions (1 min):** One thing you'll test during implementation
- **Q&A:** Other students ask questions about your choices. Defend your reasoning — not with "the tutorial said so" but with "here's the tradeoff I considered."

---

### Take audience notes (during W9D3 presentations)

While others present, capture:

- One design decision you found insightful and want to incorporate
- One question you'd ask them (write it down even if you don't get called on)
- One specific takeaway for your own guide

Submit `docs/AUDIENCE_NOTES.md` with notes from **at least 5 presentations** by end of W9D3.

---

## Deliverables

Submit all four items via your `aml-greenvision` GitHub repository:

### 1. Research log

**File:** `docs/RESEARCH_LOG.md`

**Requirements:**
- At least 3 research sessions documented
- Each session includes: question, tool used, prompt/query, what you learned, what's still unclear
- Evidence that you went deeper than one surface-level query — follow-up questions show learning in progress

### 2. Implementation guide

**File:** `IMPLEMENTATION_GUIDE.md` (project root)

**Requirements:**
- All 8 architecture and design decisions addressed with reasoning
- At least one code snippet showing how a decision becomes implementation
- Cited sources (AI tools, documentation, blog posts)
- Honest about uncertainty — "I'm not sure yet whether 3 or 5 epochs is right for Phase 1" is better than guessing

### 3. Presentation (delivered in class W9D3)

**Format:** 8 min + 4 min Q&A
**What to bring:** Your implementation guide open, ready to reference specific sections when asked

### 4. Audience observation notes

**File:** `docs/AUDIENCE_NOTES.md`

**Requirements:**
- Notes from at least 5 other students' presentations
- For each: one design decision you found insightful, one question you'd ask them, one thing you'll incorporate into your own guide

**Due:** end of W9D3 (submit after all presentations)

---

## Rubric — Project 3: Implementation

| Criterion | Exceeds | Mastery | Near | Below | No Evidence | Pts |
|---|---|---|---|---|---|---|
| **Research depth** — Log shows 3+ sessions, follow-up questions, evidence of wrestling with concepts. Not just first-response copy-paste. | 12 | 9 | 6 | 3 | 0 | 12 |
| **Implementation guide completeness** — All 8 decisions addressed with reasoning. Code snippets show translation to implementation. Honest about uncertainty. | 16 | 12 | 8 | 4 | 0 | 16 |
| **Presentation clarity** — Design decisions are explained with reasoning, not just stated. At least one decision connects to stakeholder impact. | 12 | 9 | 6 | 3 | 0 | 12 |
| **Peer engagement** — Audience notes show active listening — insights captured, questions asked, specific takeaways identified for own work. | 10 | 7.5 | 5 | 2.5 | 0 | 10 |

**Total Points:** 50
