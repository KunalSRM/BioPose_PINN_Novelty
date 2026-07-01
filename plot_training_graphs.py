import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# CSV file from your final training run
csv_path = Path("earlystop_10k_final.csv")

# Output folder for graphs
out_dir = Path("training_graphs_final")
out_dir.mkdir(exist_ok=True)

df = pd.read_csv(csv_path)

# Best epoch based on validation PCK
best_idx = df["val_pck"].idxmax()
best_epoch = int(df.loc[best_idx, "epoch"])
best_pck = df.loc[best_idx, "val_pck"]

# 1. Validation PCK curve
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["val_pck"], marker="o", label="Validation PCK")
plt.axvline(best_epoch, linestyle="--", label=f"Best Epoch {best_epoch}")
plt.title("Validation PCK vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation PCK (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "validation_pck_curve.png", dpi=300)
plt.close()

# 2. Training and Validation Loss curve
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train_loss"], marker="o", label="Training Loss")
plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation Loss")
plt.axvline(best_epoch, linestyle="--", label=f"Best Epoch {best_epoch}")
plt.title("Training and Validation Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "loss_curve.png", dpi=300)
plt.close()

# 3. Validation Jitter curve
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["val_jitter"], marker="o", label="Validation Jitter")
plt.axvline(best_epoch, linestyle="--", label=f"Best Epoch {best_epoch}")
plt.title("Validation Jitter vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Jitter")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "validation_jitter_curve.png", dpi=300)
plt.close()

# 4. Component losses
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["val_data"], marker="o", label="Validation Data Loss")
plt.plot(df["epoch"], df["val_bone"], marker="o", label="Validation Bone Loss")
plt.plot(df["epoch"], df["val_smooth"], marker="o", label="Validation Smooth Loss")
plt.plot(df["epoch"], df["val_angle"], marker="o", label="Validation Angle Loss")
plt.axvline(best_epoch, linestyle="--", label=f"Best Epoch {best_epoch}")
plt.title("Validation Component Losses vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "validation_component_losses.png", dpi=300)
plt.close()

print("Graphs saved successfully in:", out_dir)
print(f"Best Validation PCK: {best_pck:.2f}% at epoch {best_epoch}")