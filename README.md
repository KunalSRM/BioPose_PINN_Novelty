# BioPosePINN: Animal Pose Estimation with Early Stopping

This repository contains the updated BioPosePINN training pipeline for animal pose estimation using a ResNet50-based encoder and physics-inspired regularization losses.

## Project Objective

The objective of this work is to train a robust animal pose estimation model and improve the stability of the learning curve using early stopping and best-checkpoint saving.

## Final Training Result

| Metric | Value |
|---|---:|
| Best Validation PCK | 89.98% |
| Best Epoch | 44 |
| Early Stopping Epoch | 49 |
| Max Epochs Configured | 80 |
| Batch Size | 4 |
| Training Triplets | 8000 |
| Validation Triplets | 2000 |
| Learning Rate | 0.0003 |
| Early Stopping Patience | 5 |
| Early Stopping Min Delta | 0.05 |

## Training Summary

The validation PCK improved smoothly from 21.64% to 89.98%. Validation loss also decreased consistently during training. After epoch 44, the model performance plateaued around 89.5–90%, and early stopping was triggered at epoch 49 after 5 consecutive epochs without significant validation PCK improvement.

## Key Features

- ResNet50-based pose estimation model
- Triplet-based temporal input
- Data loss, bone loss, smoothness loss, and angle loss
- Early stopping based on validation PCK
- Best checkpoint saving
- CSV logging for all training metrics
- Graph generation script for result visualization

## Files

| File/Folder | Description |
|---|---|
| `biopose_train_final_modular_earlystop_complete.py` | Main training script |
| `plot_training_graphs.py` | Script to generate training graphs |
| `adaptive_weights.json` | Loss weight configuration |
| `results/earlystop_10k_final.csv` | Final training log |
| `results/*.png` | Training and validation graphs |
| `checkpoints/best_biopose_model.pt` | Best saved model checkpoint |

## How to Run Training

```bash
python biopose_train_final_modular_earlystop_complete.py --epochs 80 --batch_size 4 --num_workers 0 --max_triplets 10000 --lr 0.0003 --early_stop_patience 5 --early_stop_min_delta 0.05 --checkpoint_dir checkpoints_earlystop_10k_final --log_file earlystop_10k_final.csv
