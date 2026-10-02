# Krishi Saarthi

Krishi Saarthi is a plant-health image classification project for wheat,
chickpea, and soybean. The current milestone is reliable data preparation
before model training.

## Build the audited dataset manifest

Run this from the repository root:

```bash
python3 src/data/build_dataset_manifest.py
```

This command does not alter source images. It validates readable images, finds
byte-identical duplicates, canonicalizes known label-name differences, and
creates deterministic 70/15/15 train/validation/test assignments from unique
image hashes.

Outputs are written to `results/data/`:

- `quality_report.json` — scope, duplicate counts, integrity issues, and class/split counts.
- `source_manifest.csv` — every included source image, including duplicate copies.
- `clean_split_manifest.csv` — one canonical image per exact-duplicate group for training.
- `invalid_images.csv` — unreadable source images, if any.
- `label_conflicts.csv` — identical images assigned incompatible labels; these are excluded.

## Dataset scope

- Wheat: all supplied image folders are re-split because the supplied splits contain duplicates.
- Chickpea: only `FUSARIUM-22/dataset_raw` is eligible; the supplied augmented images are excluded.
- Soybean: only labelled raw camera images are eligible. This supports healthy-versus-diseased classification, not disease-name identification.

Do not begin model comparison until `quality_report.json` has been reviewed.

## Wheat baseline

After installing the dependencies, run:

```bash
python3 src/training/train_wheat_baseline.py --epochs 8
```

The script uses only the clean manifest. It saves `best_model.pt`, `metrics.json`,
and a confusion matrix under `results/wheat_baseline/`. Treat this small CNN as a
pipeline sanity check; compare transfer-learning models only after reviewing its
held-out test macro F1 and per-class report.

## Transfer-learning comparison

Run the two transfer-learning experiments sequentially, keeping the same manifest,
seed, epochs, and preprocessing for a fair comparison:

```bash
python3 src/training/train_wheat_transfer.py --architecture resnet18 --epochs 4
python3 src/training/train_wheat_transfer.py --architecture mobilenet_v3_small --epochs 4
```

Both scripts download ImageNet weights on their first run, freeze the pretrained
backbone, train only the classifier head, and save results under `results/`.

See `docs/model_training_handoff.md` for the completed results and the team
protocol for the remaining models.
