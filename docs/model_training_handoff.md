# Wheat Model Training Handoff

All experiments must use `results/data/clean_split_manifest.csv`. Do not use
the original wheat train, valid, or test folders directly because they contain
duplicate images and label conflicts.

## Completed Experiments

| Model | Training protocol | Test accuracy | Test macro F1 |
| --- | --- | ---: | ---: |
| SmallCNN baseline | 8 epochs, trained from scratch | 39.84% | 34.50% |
| ResNet 18 | ImageNet pretrained backbone frozen, classifier head trained for 1 epoch | 64.69% | 60.98% |
| MobileNetV3 Small | ImageNet pretrained backbone frozen, classifier head trained for 1 epoch | 66.52% | 63.00% |

The test figures are initial results, not final deployment claims. The first
transfer-learning epoch is already substantially better than the from-scratch
baseline. MobileNetV3 Small is the current leader and is a strong app candidate
because it is also lightweight.

## Work for the Remaining Two Models

1. **Baseline CNN**: keep the existing baseline as the from-scratch reference.
   If another run is needed, use the same manifest, seed 42, and report macro F1
   rather than only accuracy.
2. **EfficientNet B0**: add it to `train_wheat_transfer.py` using ImageNet
   weights, freeze its feature extractor, replace the final classifier with a
   15-class linear layer, and train for the same one-epoch initial protocol.

For every experiment, retain these conditions:

- Seed: `42`
- Train/validation/test manifest: `clean_split_manifest.csv`
- Input: ImageNet normalization with the same crop/resize transform
- Model selection: validation macro F1
- Final metric: one evaluation on the held-out test split
- Required outputs: `best_model.pt`, `metrics.json`, and `confusion_matrix.png`

## Next Team Decision

After EfficientNet is available, make a table from all metrics files. Prefer
the model with the best test macro F1 only if its per-class recall is acceptable
and inference time suits the app. Do not test repeatedly while tuning; use the
validation split for model choices and reserve the test set for the final
comparison.
