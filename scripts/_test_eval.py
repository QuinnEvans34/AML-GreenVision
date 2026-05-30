"""One-off test-set evaluation against the registered model."""
import torch
import mlflow
import mlflow.pytorch

from greenvision.data.datasets import build_dataloaders


def main() -> None:
    mlflow.set_tracking_uri("./mlruns")

    print("Loading GreenVision v3 from registry...")
    model = mlflow.pytorch.load_model("models:/GreenVision/3")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    print("Rebuilding test loader (seed=42, same split)...")
    _, _, test_loader, _ = build_dataloaders(
        "data/raw/PlantVillage", batch_size=64, seed=42
    )

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    test_acc = correct / total

    val_acc = 0.9973
    print()
    print(f"  Best val_acc (training):  {val_acc:.4f} (epoch 15)")
    print(f"  Test acc  (held-out):     {test_acc:.4f} ({correct}/{total})")
    print(f"  Gap (val − test):         {val_acc - test_acc:+.4f}")
    print()
    gap = abs(val_acc - test_acc)
    if gap < 0.01:
        print("  → No overfitting. Val and test agree within 1%.")
    elif test_acc > 0.95:
        print("  → Mild overfit but still excellent (test > 95%).")
    elif test_acc > 0.85:
        print("  → Some overfitting. Acceptable but worth noting in TRAINING_REPORT.md.")
    else:
        print("  → Real overfit problem. Investigate before W10A1.")


if __name__ == "__main__":
    main()
