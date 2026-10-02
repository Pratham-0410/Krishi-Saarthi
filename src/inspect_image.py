"""
===============================================================================
Krishi Saarthi - Multi-Crop Disease Detection System
Entry Point: Image Inspector CLI
===============================================================================

This script serves as the command-line entry point for inspecting individual image
files within the dataset. All core logic resides in `image_inspector.py`.
"""

import sys
from pathlib import Path

# Ensure src directory is in sys.path for clean relative imports
sys.path.insert(0, str(Path(__file__).parent))

from image_inspector import inspect_image, print_report


def find_sample_image(datasets_dir: Path) -> Path | None:
    """
    Find a sample image inside the datasets directory if no path is specified by user.

    Args:
        datasets_dir (Path): Root path to datasets folder.

    Returns:
        Path | None: Path to first image found, or None.
    """
    if not datasets_dir.exists():
        return None

    # Search for common image formats recursively
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for img_path in datasets_dir.rglob(ext):
            if img_path.is_file() and not any(part.startswith(".") for part in img_path.parts):
                return img_path
    return None


def main() -> None:
    """Entry point for CLI execution."""
    if len(sys.argv) > 1:
        target_image = Path(sys.argv[1])
    else:
        # Fallback to finding an example image in datasets directory
        datasets_root = Path("datasets").resolve()
        sample = find_sample_image(datasets_root)
        if sample:
            print(f"ℹ️  No path provided. Defaulting to sample image: {sample.relative_to(datasets_root.parent)}")
            target_image = sample
        else:
            print("Usage: python3 src/inspect_image.py <path_to_image>")
            sys.exit(1)

    metadata = inspect_image(target_image)
    print_report(metadata)


if __name__ == "__main__":
    main()

