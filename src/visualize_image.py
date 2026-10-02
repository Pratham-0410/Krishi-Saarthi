"""
===============================================================================
Krishi Saarthi - Multi-Crop Disease Detection System
Entry Point: Image Visualizer CLI
===============================================================================

This script serves as the command-line entry point for visualizing dataset images
and printing structured visual metadata reports.
All core logic resides in `image_visualizer.py`.

Usage:
    python src/visualize_image.py <path_to_image>

Example:
    python src/visualize_image.py datasets/wheat/data/train/Healthy/healthy_0.png
"""

import sys
from pathlib import Path

# Ensure src directory is in sys.path for clean relative imports
sys.path.insert(0, str(Path(__file__).parent))

from inspect_image import find_sample_image
from image_visualizer import (
    analyze_image_visualization,
    display_image,
    print_visualization_report,
)


def main() -> None:
    """Entry point for CLI execution."""
    if len(sys.argv) > 1:
        target_image = Path(sys.argv[1])
    else:
        # Fallback to finding an example image in datasets directory
        datasets_root = Path("datasets").resolve()
        sample = find_sample_image(datasets_root)
        if sample:
            print(f"ℹ️  No path provided. Defaulting to sample image: {sample.relative_to(datasets_root.parent)}\n")
            target_image = sample
        else:
            print("Usage: python src/visualize_image.py <path_to_image>")
            sys.exit(1)

    metadata = analyze_image_visualization(target_image)
    if metadata:
        print_visualization_report(metadata)
        display_image(metadata)


if __name__ == "__main__":
    main()
