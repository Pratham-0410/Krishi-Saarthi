"""Build a clean, reproducible manifest from Krishi Saarthi source images.

The script never moves, deletes, or modifies source images.  It records raw
image metadata, validates readability, finds byte-identical duplicates, and
creates deterministic train/validation/test assignments from unique images.

Only labelled raw images are eligible for modelling:
* wheat: all currently supplied images, re-split independently of their
  supplied train/valid/test directories because those directories leak images;
* chickpea: FUSARIUM-22/dataset_raw (not the pre-augmented directory);
* soybean: labelled Raw_SoyNet_Data/Camera Clicks images.  The unlabelled
  Mobile Click directory is deliberately excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "datasets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class ImageRecord:
    """One candidate source image and the task label it can support."""

    path: str
    crop: str
    task: str
    label: str
    source_dataset: str
    source_split: str


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def canonical_wheat_label(name: str) -> str:
    """Map naming differences across supplied wheat folders to one label."""
    cleaned = re.sub(r"_(?:test|valid)$", "", name.strip().lower())
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned.replace("blast_test", "blast")


def wheat_records(datasets_dir: Path) -> Iterable[ImageRecord]:
    root = datasets_dir / "wheat" / "data"
    for supplied_split in ("train", "valid", "test"):
        split_dir = root / supplied_split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            label = canonical_wheat_label(class_dir.name)
            for path in sorted(class_dir.iterdir()):
                if is_image(path):
                    yield ImageRecord(
                        str(path.resolve()), "wheat", "wheat_condition", label,
                        "wheat", supplied_split,
                    )


CHICKPEA_LABELS = {
    "1(HR)": "severity_1_highly_resistant",
    "3(R)": "severity_3_resistant",
    "5(MR)": "severity_5_moderately_resistant",
    "7(S)": "severity_7_susceptible",
    "9(HS)": "severity_9_highly_susceptible",
}


def chickpea_records(datasets_dir: Path) -> Iterable[ImageRecord]:
    root = datasets_dir / "chickpea" / "FUSARIUM-22" / "dataset_raw"
    for directory_name, label in CHICKPEA_LABELS.items():
        class_dir = root / directory_name
        for path in sorted(class_dir.iterdir()) if class_dir.is_dir() else []:
            if is_image(path):
                yield ImageRecord(
                    str(path.resolve()), "chickpea", "chickpea_fusarium_severity", label,
                    "FUSARIUM-22_raw", "raw",
                )


def soybean_records(datasets_dir: Path) -> Iterable[ImageRecord]:
    root = (
        datasets_dir / "soyabean"
        / "SoyNet Indian Soybean Image dataset with quality images captured from the agriculture field ( healthy and disease Images)"
        / "SoyNet" / "Raw_SoyNet_Data" / "Camera Clicks"
    )
    labels = {"Disease_Pic": "diseased", "Healthy Pic": "healthy"}
    for directory_name, label in labels.items():
        class_dir = root / directory_name
        for path in sorted(class_dir.iterdir()) if class_dir.is_dir() else []:
            if is_image(path):
                yield ImageRecord(
                    str(path.resolve()), "soybean", "soybean_health", label,
                    "SoyNet_raw_camera", "raw",
                )


def sha256_and_verify(record: ImageRecord) -> tuple[ImageRecord, str | None, str | None]:
    """Return a SHA-256 digest and a readable-image error, if any."""
    path = Path(record.path)
    try:
        with Image.open(path) as image:
            image.verify()
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        return record, digest, None
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return record, None, f"{type(error).__name__}: {error}"


def split_groups(
    groups: list[tuple[str, list[ImageRecord]]], seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, str]:
    """Deterministically split unique-image groups, stratified by task/label."""
    by_label: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for digest, records in groups:
        sample = records[0]
        by_label[(sample.crop, sample.task, sample.label)].append(digest)

    assignments: dict[str, str] = {}
    for key, hashes in sorted(by_label.items()):
        rng = random.Random(f"{seed}:{'|'.join(key)}")
        rng.shuffle(hashes)
        count = len(hashes)
        validation_count = round(count * ratios[1])
        test_count = round(count * ratios[2])
        # Keep a training image for every class that has data.
        while validation_count + test_count >= count and (validation_count or test_count):
            if test_count >= validation_count and test_count:
                test_count -= 1
            else:
                validation_count -= 1
        for index, digest in enumerate(hashes):
            assignments[digest] = (
                "validation" if index < validation_count else
                "test" if index < validation_count + test_count else "train"
            )
    return assignments


def write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(datasets_dir: Path, output_dir: Path, seed: int) -> dict[str, object]:
    candidates = list(wheat_records(datasets_dir))
    candidates.extend(chickpea_records(datasets_dir))
    candidates.extend(soybean_records(datasets_dir))
    if not candidates:
        raise FileNotFoundError(f"No supported source images found under {datasets_dir}")

    with ThreadPoolExecutor() as executor:
        inspected = list(executor.map(sha256_and_verify, candidates))

    invalid = []
    hash_groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record, digest, error in inspected:
        if error:
            invalid.append({"path": record.path, "error": error})
        elif digest:
            hash_groups[digest].append(record)

    conflicting = []
    valid_groups: list[tuple[str, list[ImageRecord]]] = []
    for digest, records in hash_groups.items():
        labels = {(r.crop, r.task, r.label) for r in records}
        if len(labels) != 1:
            conflicting.append({"sha256": digest, "paths": [r.path for r in records]})
        else:
            valid_groups.append((digest, records))

    assignments = split_groups(valid_groups, seed, (0.70, 0.15, 0.15))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    clean_rows = []
    for digest, records in sorted(hash_groups.items()):
        duplicate_count = len(records)
        conflict = len({(r.crop, r.task, r.label) for r in records}) != 1
        for index, record in enumerate(sorted(records, key=lambda row: row.path)):
            row = asdict(record) | {
                "sha256": digest,
                "duplicate_group_size": str(duplicate_count),
                "is_canonical": str(index == 0 and not conflict).lower(),
                "split": assignments.get(digest, "excluded_label_conflict"),
            }
            all_rows.append(row)
            if index == 0 and not conflict:
                clean_rows.append(row)

    fields = ["path", "crop", "task", "label", "source_dataset", "source_split", "sha256", "duplicate_group_size", "is_canonical", "split"]
    conflict_rows = []
    for conflict in conflicting:
        for path in conflict["paths"]:
            matching_record = next(record for record in hash_groups[conflict["sha256"]] if record.path == path)
            conflict_rows.append({
                "sha256": conflict["sha256"], "path": path, "crop": matching_record.crop,
                "task": matching_record.task, "label": matching_record.label,
            })

    write_csv(output_dir / "source_manifest.csv", all_rows, fields)
    write_csv(output_dir / "clean_split_manifest.csv", clean_rows, fields)
    write_csv(output_dir / "invalid_images.csv", invalid, ["path", "error"])
    write_csv(
        output_dir / "label_conflicts.csv", conflict_rows,
        ["sha256", "path", "crop", "task", "label"],
    )

    counts = Counter()
    split_counts = Counter()
    for row in clean_rows:
        counts[f"{row['crop']}::{row['task']}::{row['label']}"] += 1
        split_counts[f"{row['crop']}::{row['task']}::{row['label']}::{row['split']}"] += 1
    duplicate_copies = sum(len(records) - 1 for records in hash_groups.values() if len(records) > 1)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "scope": {
            "included": ["wheat supplied images (re-split)", "chickpea raw", "soybean raw camera clicks"],
            "excluded": ["chickpea pre-augmented", "soybean preprocessed", "soybean unlabelled mobile clicks"],
        },
        "candidate_images": len(candidates),
        "valid_images": sum(len(records) for records in hash_groups.values()),
        "unique_exact_images": len(hash_groups),
        "duplicate_file_copies": duplicate_copies,
        "invalid_images": invalid,
        "label_conflict_group_count": len(conflicting),
        "label_conflict_image_count": len(conflict_rows),
        "canonical_images_for_training": len(clean_rows),
        "class_counts": dict(sorted(counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
    }
    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_manifest(args.datasets_dir.resolve(), args.output_dir.resolve(), args.seed)
    print("Dataset manifest created successfully")
    print(f"Canonical training candidates: {report['canonical_images_for_training']}")
    print(f"Exact duplicate copies excluded: {report['duplicate_file_copies']}")
    print(f"Report: {(args.output_dir / 'quality_report.json').resolve()}")


if __name__ == "__main__":
    main()
