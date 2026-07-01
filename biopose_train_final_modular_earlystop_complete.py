import os
import re
import csv
import time
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

from model_resnet50 import PoseCNN, soft_argmax, BONES, ANGLE_TRIPLETS
from losses import (
    loss_data,
    loss_bone,
    loss_smooth,
    loss_angle,
    pck_metric,
    jitter_metric,
    estimate_bone_lengths,
)


# ============================================================
# DATASET PATHS
# ============================================================

IMAGE_ROOTS = [
    Path(r"C:\Users\sriva\OneDrive\Desktop\AI_ML_PROJECTS\IIT_BHU_WORK\BioPosePINN_Novelty\ak-images\dataset"),
]

ANNOTATION_ROOT = Path(
    r"C:\Users\sriva\OneDrive\Desktop\AI_ML_PROJECTS\IIT_BHU_WORK\BioPosePINN_Novelty\annotation"
)

SUBFOLDERS = [
    "ak_P1",
    "ak_P2",
    "ak_P3_amphibian",
    "ak_P3_bird",
    "ak_P3_fish",
    "ak_P3_mammal",
    "ak_P3_reptile",
]

NUM_KEYPOINTS = 23
IMG_W, IMG_H = 640, 360


# ============================================================
# ROBUSTNESS MODULE 1: LIGHT NOISE AUGMENTATION
# Added to improve robustness against camera/video noise.
# Kept lightweight so training does not become slow.
# ============================================================

class LightGaussianNoise:
    def __init__(self, std=0.015, p=0.20):
        self.std = std
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            x = x + torch.randn_like(x) * self.std
            x = torch.clamp(x, 0.0, 1.0)
        return x


# ============================================================
# ROBUSTNESS MODULE 2: TRAINING TRANSFORMS
# ColorJitter improves robustness to lighting/background changes.
# Light noise improves robustness to low-quality video frames.
# Heavy blur/erasing removed because it slowed your laptop.
# ============================================================

def build_train_transform(img_size):
    return T.Compose([
        T.Resize(img_size),

        # ROBUSTNESS ADDED HERE
        T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),

        T.ToTensor(),

        # ROBUSTNESS ADDED HERE
        LightGaussianNoise(std=0.015, p=0.20),

        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_val_transform(img_size):
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def find_image(image_field):
    for root in IMAGE_ROOTS:
        path = root / image_field
        if path.exists():
            return path
    raise FileNotFoundError(f"Image not found: {image_field}")


def parse_clip_and_frame(image_field):
    stem = Path(image_field).stem
    match = re.search(r"_f(\d+)$", stem)

    if match:
        return stem[:match.start()], int(match.group(1))

    return stem, 0


def load_all_entries(split="train"):
    all_entries = []

    for folder in SUBFOLDERS:
        json_path = ANNOTATION_ROOT / folder / f"{split}.json"

        if not json_path.exists():
            print(f"[WARN] Missing: {json_path}")
            continue

        with open(json_path, "r") as f:
            entries = json.load(f)

        print(f"{folder}/{split}.json -> {len(entries)} entries")
        all_entries.extend(entries)

    print(f"Total entries: {len(all_entries)}")

    if len(all_entries) == 0:
        raise RuntimeError("No annotations loaded. Check annotation path.")

    return all_entries


def build_triplets(entries, min_clip_len=3, max_triplets=None):
    clip_map = defaultdict(list)

    for idx, entry in enumerate(entries):
        clip_id, frame_num = parse_clip_and_frame(entry["image"])
        clip_map[clip_id].append((frame_num, idx))

    triplets = []
    skipped = 0

    for _, frame_list in clip_map.items():
        frame_list.sort(key=lambda x: x[0])
        indices = [idx for _, idx in frame_list]

        if len(indices) < min_clip_len:
            skipped += 1
            continue

        for i in range(len(indices) - 2):
            triplets.append((indices[i], indices[i + 1], indices[i + 2]))

    if max_triplets is not None:
        triplets = triplets[:max_triplets]

    print(f"Clips total: {len(clip_map)}")
    print(f"Clips skipped: {skipped}")
    print(f"Triplets built: {len(triplets)}")

    return triplets


# ============================================================
# DATASET CLASS
# ============================================================

class AnimalKingdomTripletDataset(Dataset):
    def __init__(
        self,
        split="train",
        img_size=(256, 256),
        min_clip_len=3,
        max_triplets=None,
    ):
        print(f"[FinalFastRobustDataset] split={split}")

        self.split = split

        if split == "train":
            self.transform = build_train_transform(img_size)
        else:
            self.transform = build_val_transform(img_size)

        self.entries = load_all_entries(split)

        self.triplets = build_triplets(
            self.entries,
            min_clip_len=min_clip_len,
            max_triplets=max_triplets,
        )

    def __len__(self):
        return len(self.triplets)

    def entry_to_tensors(self, entry):
        img_path = find_image(entry["image"])
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        joints = np.array(entry.get("joints", []), dtype=np.float32)
        joints_vis = np.array(entry.get("joints_vis", []), dtype=np.float32)

        padded_joints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        padded_vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)

        n = min(len(joints), NUM_KEYPOINTS)

        padded_joints[:n] = joints[:n]
        padded_vis[:n] = joints_vis[:n]

        invisible = (padded_joints[:, 0] < 0) | (padded_joints[:, 1] < 0)
        padded_vis[invisible] = 0.0
        padded_joints[invisible] = 0.0

        padded_joints[:, 0] /= IMG_W
        padded_joints[:, 1] /= IMG_H

        keypoints = torch.from_numpy(padded_joints).float()
        visibility = torch.from_numpy(padded_vis).float()

        return img_tensor, keypoints, visibility

    def __getitem__(self, idx):
        i0, i1, i2 = self.triplets[idx]

        img0, kp0, vis0 = self.entry_to_tensors(self.entries[i0])
        img1, kp1, vis1 = self.entry_to_tensors(self.entries[i1])
        img2, kp2, vis2 = self.entry_to_tensors(self.entries[i2])

        clip_id, _ = parse_clip_and_frame(self.entries[i1]["image"])

        return {
            "img_t0": img0,
            "img_t1": img1,
            "img_t2": img2,

            "kp_t0": kp0,
            "kp_t1": kp1,
            "kp_t2": kp2,

            "vis_t0": vis0,
            "vis_t1": vis1,
            "vis_t2": vis2,

            "animal": self.entries[i1].get("animal", "Unknown"),
            "animal_parent_class": self.entries[i1].get("animal_parent_class", "default"),
            "animal_class": self.entries[i1].get("animal_class", "default"),
            "animal_subclass": self.entries[i1].get("animal_subclass", "default"),
            "clip_id": clip_id,
        }


# ============================================================
# ARGUMENTS
# ============================================================

def get_args():
    parser = argparse.ArgumentParser("Final Modular BioPosePINN Training")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_triplets", type=int, default=15000)
    parser.add_argument("--bone_estimate_n", type=int, default=3000)

    parser.add_argument("--adaptive_weights", type=str, default="adaptive_weights.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_final_modular")
    parser.add_argument("--log_file", type=str, default="final_modular_log.csv")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_angle", action="store_true")

    # EARLY STOPPING SETTINGS
    # Monitor validation PCK and stop when it stops improving meaningfully.
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.05)

    return parser.parse_args()


# ============================================================
# NOVELTY MODULE 1: SPECIES-ADAPTIVE WEIGHTING
# Different animal classes get different biological constraint weights.
# This is one of your main BioPosePINN novelty points.
# ============================================================

def load_adaptive_weights(path):
    try:
        with open(path, "r") as f:
            weights = json.load(f)

        print(f"[INFO] Loaded adaptive weights from {path}")
        return weights

    except Exception as e:
        print(f"[WARN] Could not load adaptive weights: {e}")
        print("[INFO] Using default adaptive weights.")

        return {
            "default": {
                "bone": 0.5,
                "smooth": 0.1,
                "angle": 0.05,
            }
        }


def get_species_adaptive_lambdas(animal_classes, adaptive_weights, device):
    lb, ls, la = [], [], []

    for cls in animal_classes:
        cls = str(cls)

        weights = adaptive_weights.get(
            cls,
            adaptive_weights.get(
                "default",
                {"bone": 0.5, "smooth": 0.1, "angle": 0.05},
            ),
        )

        lb.append(weights["bone"])
        ls.append(weights["smooth"])
        la.append(weights["angle"])

    return (
        torch.tensor(lb, device=device).mean(),
        torch.tensor(ls, device=device).mean(),
        torch.tensor(la, device=device).mean(),
    )


# ============================================================
# NOVELTY MODULE 2: DYNAMIC CONSTRAINT SCHEDULING
# Biological losses are not fully forced from epoch 1.
# They increase gradually for more stable optimization.
# ============================================================

def constraint_schedule(epoch, max_epochs):
    progress = epoch / max(max_epochs, 1)
    return min(1.0, 0.3 + progress)


# ============================================================
# NOVELTY MODULE 3: BIOPOSEPINN LOSS
# Combines:
# - Data loss
# - Bone length loss
# - Temporal smoothness loss
# - Joint angle loss
# - Species-adaptive weighting
# - Dynamic scheduling
# ============================================================

def biopose_pinn_loss(
    c0,
    c1,
    c2,
    gt_coords,
    visibility,
    bone_lengths,
    joint_triples,
    animal_classes,
    adaptive_weights,
    epoch,
    max_epochs,
):
    l_data = loss_data(c1, gt_coords, visibility)
    l_bone = loss_bone(c1, bone_lengths)
    l_smooth = loss_smooth(c0, c1, c2)

    lam_bone, lam_smooth, lam_angle = get_species_adaptive_lambdas(
        animal_classes,
        adaptive_weights,
        c1.device,
    )

    schedule = constraint_schedule(epoch, max_epochs)

    lam_bone = lam_bone * schedule
    lam_smooth = lam_smooth * schedule
    lam_angle = lam_angle * schedule

    total = l_data + lam_bone * l_bone + lam_smooth * l_smooth

    log = {
        "loss_total": total.item(),
        "loss_data": l_data.item(),
        "loss_bone": l_bone.item(),
        "loss_smooth": l_smooth.item(),
        "loss_angle": 0.0,
        "lambda_bone": float(lam_bone),
        "lambda_smooth": float(lam_smooth),
        "lambda_angle": float(lam_angle),
    }

    if joint_triples:
        l_angle = loss_angle(c1, joint_triples)
        total = total + lam_angle * l_angle
        log["loss_angle"] = l_angle.item()
        log["loss_total"] = total.item()

    return total, log


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def format_metrics(metrics):
    return (
        f"loss={metrics['loss_total']:.4f} "
        f"data={metrics['loss_data']:.4f} "
        f"bone={metrics['loss_bone']:.4f} "
        f"smooth={metrics['loss_smooth']:.5f} "
        f"angle={metrics.get('loss_angle', 0):.4f} "
        f"PCK={metrics.get('pck', 0):.2f}% "
        f"jitter={metrics.get('jitter', 0):.5f}"
    )


def forward_triplet(model, t0, t1, t2):
    frames = torch.cat([t0, t1, t2], dim=0)
    coords = soft_argmax(model(frames))
    c0, c1, c2 = coords.chunk(3, dim=0)
    return c0, c1, c2


def train_one_epoch(
    model,
    loader,
    optimiser,
    scaler,
    bone_lengths,
    joint_triples,
    adaptive_weights,
    device,
    epoch,
    max_epochs,
):
    model.train()

    totals = {
        "loss_total": 0.0,
        "loss_data": 0.0,
        "loss_bone": 0.0,
        "loss_smooth": 0.0,
        "loss_angle": 0.0,
    }

    n = 0

    for batch in loader:
        t0 = batch["img_t0"].to(device)
        t1 = batch["img_t1"].to(device)
        t2 = batch["img_t2"].to(device)
        gt = batch["kp_t1"].to(device)
        vis = batch["vis_t1"].to(device)

        animal_classes = batch.get("animal_class", ["default"] * t0.size(0))

        optimiser.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            c0, c1, c2 = forward_triplet(model, t0, t1, t2)

            loss, log = biopose_pinn_loss(
                c0,
                c1,
                c2,
                gt,
                vis,
                bone_lengths,
                joint_triples,
                animal_classes,
                adaptive_weights,
                epoch,
                max_epochs,
            )

        if torch.isnan(loss):
            print("[WARN] NaN loss skipped")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimiser)
        scaler.update()

        for key in totals:
            totals[key] += log.get(key, 0.0)

        n += 1

    return {key: value / max(n, 1) for key, value in totals.items()}


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    bone_lengths,
    joint_triples,
    adaptive_weights,
    device,
    epoch,
    max_epochs,
):
    model.eval()

    totals = {
        "loss_total": 0.0,
        "loss_data": 0.0,
        "loss_bone": 0.0,
        "loss_smooth": 0.0,
        "loss_angle": 0.0,
    }

    pck_sum = 0.0
    jitter_sum = 0.0
    n = 0

    for batch in loader:
        t0 = batch["img_t0"].to(device)
        t1 = batch["img_t1"].to(device)
        t2 = batch["img_t2"].to(device)
        gt = batch["kp_t1"].to(device)
        vis = batch["vis_t1"].to(device)

        animal_classes = batch.get("animal_class", ["default"] * t0.size(0))

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            c0, c1, c2 = forward_triplet(model, t0, t1, t2)

            _, log = biopose_pinn_loss(
                c0,
                c1,
                c2,
                gt,
                vis,
                bone_lengths,
                joint_triples,
                animal_classes,
                adaptive_weights,
                epoch,
                max_epochs,
            )

        for key in totals:
            totals[key] += log.get(key, 0.0)

        pck_sum += pck_metric(c1, gt, vis, threshold=0.05)
        jitter_sum += jitter_metric(c0, c1)

        n += 1

    metrics = {key: value / max(n, 1) for key, value in totals.items()}
    metrics["pck"] = pck_sum / max(n, 1)
    metrics["jitter"] = jitter_sum / max(n, 1)

    return metrics


# ============================================================
# CHECKPOINT AND LOGGING FUNCTIONS
# ============================================================

def save_checkpoint(path, model, optimiser, epoch, val_metrics, adaptive_weights, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimiser.state_dict(),
            "val_pck": val_metrics["pck"],
            "val_loss": val_metrics["loss_total"],
            "val_jitter": val_metrics["jitter"],
            "adaptive_weights": adaptive_weights,
            "args": vars(args),
        },
        path,
    )


def create_log_file(log_file):
    fields = [
        "epoch",
        "phase",
        "train_loss",
        "train_data",
        "train_bone",
        "train_smooth",
        "train_angle",
        "val_loss",
        "val_data",
        "val_bone",
        "val_smooth",
        "val_angle",
        "val_pck",
        "val_jitter",
        "lr",
        "epoch_time_s",
    ]

    with open(log_file, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    return fields


def append_log(log_file, fields, epoch, phase, train_metrics, val_metrics, lr, elapsed):
    with open(log_file, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(
            {
                "epoch": epoch,
                "phase": phase,
                "train_loss": train_metrics["loss_total"],
                "train_data": train_metrics["loss_data"],
                "train_bone": train_metrics["loss_bone"],
                "train_smooth": train_metrics["loss_smooth"],
                "train_angle": train_metrics.get("loss_angle", 0),
                "val_loss": val_metrics["loss_total"],
                "val_data": val_metrics["loss_data"],
                "val_bone": val_metrics["loss_bone"],
                "val_smooth": val_metrics["loss_smooth"],
                "val_angle": val_metrics.get("loss_angle", 0),
                "val_pck": val_metrics["pck"],
                "val_jitter": val_metrics["jitter"],
                "lr": lr,
                "epoch_time_s": elapsed,
            }
        )


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """
    Early stopping based on validation PCK.

    Training stops if validation PCK does not improve by at least
    min_delta for patience consecutive epochs. The best checkpoint is
    still saved whenever raw validation PCK improves.
    """

    def __init__(self, patience=5, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            self.counter = 0
            return True

        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return True

        self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return False


# ============================================================
# MAIN
# ============================================================

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    adaptive_weights = load_adaptive_weights(args.adaptive_weights)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    full_ds = AnimalKingdomTripletDataset(
        split="train",
        img_size=(args.img_size, args.img_size),
        max_triplets=args.max_triplets,
    )

    n_total = len(full_ds)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"[INFO] Train triplets: {n_train}")
    print(f"[INFO] Val triplets: {n_val}")

    bone_lengths = estimate_bone_lengths(
        full_ds.entries[: args.bone_estimate_n],
        BONES,
    )

    joint_triples = None if args.no_angle else ANGLE_TRIPLETS

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = PoseCNN(
        num_keypoints=NUM_KEYPOINTS,
        pretrained=True,
        freeze_encoder=args.warmup_epochs > 0,
    ).to(device)

    optimiser = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=1e-5,
    )

    log_fields = create_log_file(args.log_file)

    best_pck = 0.0
    best_epoch = 0

    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
    )

    print(
        f"[INFO] Early stopping enabled: "
        f"patience={args.early_stop_patience}, "
        f"min_delta={args.early_stop_min_delta}"
    )

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        if epoch == args.warmup_epochs + 1:
            model.unfreeze_encoder()

            optimiser = optim.AdamW(
                [
                    {"params": model.encoder.parameters(), "lr": args.lr * 0.01},
                    {
                        "params": list(model.up1.parameters())
                        + list(model.up2.parameters())
                        + list(model.up3.parameters())
                        + list(model.heatmap_conv.parameters()),
                        "lr": args.lr,
                    },
                ],
                weight_decay=1e-4,
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimiser,
                T_max=max(1, args.epochs - args.warmup_epochs),
                eta_min=1e-5,
            )

        phase = "warmup" if epoch <= args.warmup_epochs else "train"

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimiser,
            scaler,
            bone_lengths,
            joint_triples,
            adaptive_weights,
            device,
            epoch,
            args.epochs,
        )

        val_metrics = validate_one_epoch(
            model,
            val_loader,
            bone_lengths,
            joint_triples,
            adaptive_weights,
            device,
            epoch,
            args.epochs,
        )

        if epoch > args.warmup_epochs:
            scheduler.step()

        lr = optimiser.param_groups[0]["lr"]
        elapsed = time.time() - start

        print(f"\nEpoch {epoch}/{args.epochs} [{phase}] lr={lr:.2e}")
        print("Train:", format_metrics(train_metrics))
        print("Val:  ", format_metrics(val_metrics))

        # Save best checkpoint based on raw validation PCK.
        # This preserves the actual best model even if early stopping uses min_delta.
        if val_metrics["pck"] > best_pck:
            best_pck = val_metrics["pck"]
            best_epoch = epoch

            save_path = os.path.join(args.checkpoint_dir, "best_biopose_model.pt")

            save_checkpoint(
                save_path,
                model,
                optimiser,
                epoch,
                val_metrics,
                adaptive_weights,
                args,
            )

            print(f"[BEST] PCK={best_pck:.2f}% saved")

        # Early stopping checks only meaningful improvement.
        # This helps avoid unnecessary later epochs when the curve becomes noisy.
        significant_improvement = early_stopper.step(val_metrics["pck"])

        if not significant_improvement:
            print(
                f"[EARLY STOP CHECK] No significant validation PCK improvement. "
                f"Counter: {early_stopper.counter}/{early_stopper.patience}"
            )

        append_log(
            args.log_file,
            log_fields,
            epoch,
            phase,
            train_metrics,
            val_metrics,
            lr,
            elapsed,
        )

        if early_stopper.early_stop:
            print("\n" + "=" * 60)
            print("[EARLY STOPPING TRIGGERED]")
            print(
                f"No significant validation PCK improvement for "
                f"{early_stopper.patience} consecutive epochs."
            )
            print(f"Best Validation PCK: {best_pck:.2f}% at epoch {best_epoch}")
            print("=" * 60)
            break

    print("\nTraining complete.")
    print(f"Best PCK: {best_pck:.2f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()