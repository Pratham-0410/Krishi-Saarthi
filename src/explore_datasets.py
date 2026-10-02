"""
===============================================================================
Krishi Saarthi - Multi-Crop Disease Detection System
Module: Dataset Explorer (Production-Grade Utility)
===============================================================================

This module provides a robust, reusable, framework-agnostic dataset exploration
tool designed for multi-crop plant disease classification projects.

Architectural Design & Decisions:
---------------------------------
1. Standard Library Only (`pathlib`, `typing`, `dataclasses`):
   - We avoid heavy third-party dependencies like Pandas, OpenCV, or PyTorch
     for basic dataset inspection. This keeps the utility lightweight, fast,
     and instantly executable in any Python environment (CI/CD pipelines,
     lightweight Docker containers, or local developer setups).

2. Pathlib over `os`:
   - `pathlib.Path` provides object-oriented filesystem paths with cross-platform
     compatibility (handling Windows '\\' and POSIX '/' seamlessly). It also
     avoids string manipulation bugs common with `os.path.join`.

3. Dynamic Hierarchy Discovery:
   - Real-world agricultural datasets rarely follow a uniform layout. Some crops
     use `crop/dataset_variant/split/class_name/`, while others use
     `crop/class_name/`. Instead of hardcoding directory depths, this script
     dynamically discovers class folders by identifying directories that contain
     supported image files (or leaf directories), grouping them by dataset variant.

4. Resilient & Defensively Programmed:
   - Ignores hidden metadata files (`.DS_Store`, `.git`, `__pycache__`).
   - Gracefully handles empty folders and flags warnings for missing data.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple


# =============================================================================
# GLOBAL CONFIGURATION & CONSTANTS
# =============================================================================

# Supported image file extensions (stored as a set for O(1) membership lookup).
# Extensions are stored in lowercase with leading dots.
SUPPORTED_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ClassInfo:
    """
    Metadata representation of a single dataset class folder.
    
    Attributes:
        name (str): Name of the class (e.g., 'Healthy', 'Yellow Rust').
        path (Path): Absolute or relative filesystem Path to the class folder.
        image_count (int): Number of valid image files directly inside the folder.
    """
    name: str
    path: Path
    image_count: int


@dataclass
class DatasetInfo:
    """
    Representation of a sub-dataset or data split (e.g., 'data/train', 'FUSARIUM-22/dataset_raw').
    
    Attributes:
        relative_path (str): Relative sub-path within the crop folder.
        classes: Dictionary mapping class name to ClassInfo.
    """
    relative_path: str
    classes: Dict[str, ClassInfo] = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        """Calculate total images across all classes in this dataset variant."""
        return sum(cls.image_count for cls in self.classes.values())


@dataclass
class CropInfo:
    """
    Representation of a top-level Crop entity (e.g., 'wheat', 'chickpea', 'soybean').
    
    Attributes:
        name (str): Name of the crop.
        path (Path): Path to the crop directory.
        datasets: Dictionary mapping dataset sub-path to DatasetInfo.
    """
    name: str
    path: Path
    datasets: Dict[str, DatasetInfo] = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        """Calculate total images across all dataset variants of this crop."""
        return sum(ds.total_images for ds in self.datasets.values())

    @property
    def total_classes(self) -> int:
        """Calculate total classes across all dataset variants of this crop."""
        return sum(len(ds.classes) for ds in self.datasets.values())


# =============================================================================
# HELPER & DISCOVERY FUNCTIONS
# =============================================================================

def is_hidden(path: Path) -> bool:
    """
    Check if a file or directory path is hidden or OS-generated metadata.

    Why this function is necessary:
    -------------------------------
    Operating systems like macOS frequently insert hidden metadata files
    (`.DS_Store`, `._file`) and version control systems create `.git` folders.
    If not explicitly ignored, these files corrupt image counting statistics
    and clutter dataset exploration outputs.

    Args:
        path (Path): The filesystem path to evaluate.

    Returns:
        bool: True if any part of the path starts with '.', False otherwise.
    """
    # Check if the filename or any parent folder segment starts with a dot
    return any(part.startswith(".") for part in path.parts)


def is_image_file(file_path: Path, valid_extensions: Set[str] = SUPPORTED_EXTENSIONS) -> bool:
    """
    Verify whether a given file path is a supported image based on its extension.

    Why case-insensitivity matters:
    --------------------------------
    Camera hardware and web datasets export extensions unpredictably
    ('.JPG', '.JPEG', '.png'). Normalizing to lowercase prevents missing images.

    Args:
        file_path (Path): Path to the file.
        valid_extensions (Set[str]): Set of allowed file extensions.

    Returns:
        bool: True if the file exists, is a regular file, and has a supported extension.
    """
    return file_path.is_file() and file_path.suffix.lower() in valid_extensions


def count_images(directory: Path, valid_extensions: Set[str] = SUPPORTED_EXTENSIONS) -> int:
    """
    Count the number of direct image files contained inside a directory.

    Why non-recursive counting here:
    --------------------------------
    Counting is performed only for direct children of a class folder to avoid
    double-counting images if a dataset has nested sub-structures.

    Args:
        directory (Path): Directory path to inspect.
        valid_extensions (Set[str]): Set of supported image extensions.

    Returns:
        int: Total number of valid image files directly inside the directory.
    """
    if not directory.is_dir():
        return 0

    count = 0
    for item in directory.iterdir():
        if not is_hidden(item) and is_image_file(item, valid_extensions):
            count += 1
    return count


def discover_crops(datasets_dir: Path) -> List[Path]:
    """
    Discover top-level crop directories inside the root datasets path.

    Why validate directory existence:
    ----------------------------------
    Gracefully prevents runtime crashes if the user points the script to a
    non-existent or misconfigured datasets folder.

    Args:
        datasets_dir (Path): Root directory containing crop folders.

    Returns:
        List[Path]: Sorted list of Path objects representing valid crop directories.
    """
    if not datasets_dir.exists():
        print(f"❌ ERROR: Datasets root directory '{datasets_dir}' does not exist.")
        return []

    if not datasets_dir.is_dir():
        print(f"❌ ERROR: Specified path '{datasets_dir}' is not a directory.")
        return []

    # Retrieve immediate subdirectories, ignoring hidden folders
    crops = [
        item for item in datasets_dir.iterdir()
        if item.is_dir() and not is_hidden(item)
    ]
    
    # Sort paths alphabetically for consistent, deterministic reporting
    return sorted(crops, key=lambda p: p.name.lower())


def discover_classes_and_datasets(crop_dir: Path) -> CropInfo:
    """
    Recursively inspect a crop directory to discover datasets, classes, and image counts.

    Hierarchy Discovery Algorithm:
    ------------------------------
    1. Traverses all subdirectories of `crop_dir` recursively.
    2. Identifies a directory as a "Class Folder" if:
       a) It directly contains one or more supported image files.
       b) OR it is a leaf directory (no further subdirectories) under the crop tree.
    3. Infers the "Dataset Variant" relative path by taking the relative path
       from `crop_dir` to the parent of the class folder.
    4. Handles empty folders gracefully and flags warnings for 0-image classes.

    Args:
        crop_dir (Path): Path to the crop folder (e.g. 'datasets/wheat').

    Returns:
        CropInfo: Structured metadata container holding all datasets and classes for the crop.
    """
    crop_info = CropInfo(name=crop_dir.name, path=crop_dir)

    # Gather all subdirectories inside crop_dir (including crop_dir itself if needed)
    all_subdirs = [p for p in crop_dir.rglob("*") if p.is_dir() and not is_hidden(p)]
    
    # Also include crop_dir in case crop_dir itself directly contains images or classes
    all_subdirs.append(crop_dir)

    # Filter directories that act as class folders:
    # Either they contain direct images OR they have no child subdirectories
    for subdir in all_subdirs:
        img_count = count_images(subdir)
        
        # Check if subdir has child directories
        has_child_dirs = any(child.is_dir() for child in subdir.iterdir() if not is_hidden(child))

        # A folder is considered a class folder if it contains images OR is a leaf directory
        if img_count > 0 or not has_child_dirs:
            # Skip the crop root itself if it only contains child directories
            if subdir == crop_dir and has_child_dirs:
                continue

            # Determine dataset relative path
            # Example:
            # If crop_dir = datasets/wheat
            # And class folder = datasets/wheat/data/train/Aphid
            # Parent folder = datasets/wheat/data/train
            # Relative dataset sub-path = "data/train"
            parent_dir = subdir.parent
            if parent_dir == crop_dir or parent_dir == crop_dir.parent:
                dataset_rel_path = "root"
            else:
                dataset_rel_path = str(parent_dir.relative_to(crop_dir))

            # Ensure DatasetInfo container exists for this sub-path
            if dataset_rel_path not in crop_info.datasets:
                crop_info.datasets[dataset_rel_path] = DatasetInfo(relative_path=dataset_rel_path)

            class_info = ClassInfo(
                name=subdir.name,
                path=subdir,
                image_count=img_count
            )
            crop_info.datasets[dataset_rel_path].classes[subdir.name] = class_info

    return crop_info


# =============================================================================
# PRESENTATION & REPORTING FUNCTIONS
# =============================================================================

def print_tree(crops_data: List[CropInfo]) -> Tuple[int, int, int, int]:
    """
    Print the discovered dataset hierarchy in a clean, visual ASCII tree format.

    Why visual tree formatting:
    ---------------------------
    Complex dataset hierarchies are hard to digest in raw text or logs. A visual
    tree representation gives engineers an immediate, intuitive view of class
    distribution and potential path anomalies.

    Args:
        crops_data (List[CropInfo]): List of explored CropInfo objects.

    Returns:
        Tuple[int, int, int, int]: Totals (total_crops, total_datasets, total_classes, total_images)
    """
    total_crops = len(crops_data)
    total_datasets = 0
    total_classes = 0
    total_images = 0

    print("\n" + "=" * 65)
    print("      KRISHI SAARTHI - MULTI-CROP DATASET HIERARCHY TREE      ")
    print("=" * 65 + "\n")

    if not crops_data:
        print("⚠️ No valid crop directories found inside datasets/.")
        return 0, 0, 0, 0

    for crop in crops_data:
        print(f"🌾 Crop: {crop.name}")

        if not crop.datasets:
            print("   └── ⚠️ [WARNING] No dataset sub-folders or images found for this crop.")
            continue

        dataset_keys = sorted(crop.datasets.keys())
        for ds_idx, ds_key in enumerate(dataset_keys):
            dataset = crop.datasets[ds_key]
            total_datasets += 1

            is_last_ds = (ds_idx == len(dataset_keys) - 1)
            ds_prefix = "   └── " if is_last_ds else "   ├── "
            indent_prefix = "       " if is_last_ds else "   │   "

            print(f"{ds_prefix}📦 Dataset Variant / Sub-path: [{dataset.relative_path}]")

            if not dataset.classes:
                print(f"{indent_prefix}└── ⚠️ [WARNING] Folder is empty or contains no recognized class subfolders.")
                continue

            class_keys = sorted(dataset.classes.keys())
            for cls_idx, cls_key in enumerate(class_keys):
                cls_info = dataset.classes[cls_key]
                total_classes += 1
                total_images += cls_info.image_count

                is_last_cls = (cls_idx == len(class_keys) - 1)
                cls_prefix = f"{indent_prefix}└── " if is_last_cls else f"{indent_prefix}├── "

                # Highlight warnings if a class folder contains 0 images
                if cls_info.image_count == 0:
                    status_str = "⚠️ [WARNING: 0 Images Found]"
                else:
                    status_str = f"{cls_info.image_count:,} images"

                print(f"{cls_prefix}📁 Class: {cls_info.name:<25} -> {status_str}")

        print()  # Blank line separator between crops

    return total_crops, total_datasets, total_classes, total_images


def print_summary(total_crops: int, total_datasets: int, total_classes: int, total_images: int) -> None:
    """
    Print a clean, professional summary box of key dataset metrics.

    Args:
        total_crops (int): Total number of crops discovered.
        total_datasets (int): Total dataset variants across all crops.
        total_classes (int): Total class folders discovered.
        total_images (int): Total image files counted.
    """
    print("-" * 40)
    print("           DATASET SUMMARY            ")
    print("-" * 40)
    print(f"  Total Crops    : {total_crops:>8,}")
    print(f"  Total Datasets : {total_datasets:>8,}")
    print(f"  Total Classes  : {total_classes:>8,}")
    print(f"  Total Images   : {total_images:>8,}")
    print("-" * 40 + "\n")


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

def explore_datasets(datasets_path_str: str = "datasets") -> None:
    """
    Main orchestration function for exploring dataset hierarchies.

    Args:
        datasets_path_str (str): Path string to the root datasets directory.
    """
    # 1. Resolve path relative to current working directory or absolute
    datasets_dir = Path(datasets_path_str).resolve()
    
    print(f"🔍 Inspecting Datasets Root: {datasets_dir}")
    print(f"ℹ️  Supported extensions : {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    # 2. Discover crops
    crop_paths = discover_crops(datasets_dir)
    
    # 3. Explore hierarchy per crop
    crops_info: List[CropInfo] = []
    for crop_path in crop_paths:
        crop_info = discover_classes_and_datasets(crop_path)
        crops_info.append(crop_info)

    # 4. Render visual tree & collect totals
    t_crops, t_datasets, t_classes, t_images = print_tree(crops_info)

    # 5. Print executive summary report
    print_summary(t_crops, t_datasets, t_classes, t_images)


if __name__ == "__main__":
    # Allows running directly via terminal: python3 src/explore_datasets.py
    # Also allows passing a custom datasets path as a command-line argument if desired.
    target_path = sys.argv[1] if len(sys.argv) > 1 else "datasets"
    explore_datasets(target_path)
