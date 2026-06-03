"""Image transforms and normalization constants for GreenVision.

The ImageNet normalization statistics are locked to match EfficientNet-B0's
pretraining. They are used identically in ``train_tfms`` and ``eval_tfms`` and
must not be changed — any drift silently corrupts every prediction.

Three pipelines:
  - ``train_tfms``: original v3 training augmentation (light)
  - ``train_tfms_robust``: aggressive augmentation for the W10P1 v4 fine-tune,
    designed to force texture-based learning and reduce background bias
    (Decision 15)
  - ``eval_tfms``: deterministic — used at validation, test, AND serving
"""

from torchvision import transforms

IMG_SIZE: int = 224
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

# Decision 15 — robust augmentation pipeline used by the v4 fine-tune.
# Goal: force the model to attend to leaf texture and shape rather than
# the implicit "leaf on neutral background" cue v3 learned. Three knobs
# carry the load:
#   - RandomGrayscale(0.10) — 1-in-10 batches see grayscale, so color
#     alone can't drive the prediction.
#   - GaussianBlur (p=0.25) — simulates phone-photo defocus.
#   - RandomErasing(p=0.4, scale up to 25%) — destroys patches including
#     background regions, teaching the model that surrounding context
#     isn't required.
train_tfms_robust = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15
        ),
        transforms.RandomGrayscale(p=0.10),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
            p=0.25,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(
            p=0.40,
            scale=(0.02, 0.25),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ]
)

eval_tfms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
