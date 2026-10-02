"""
===============================================================================
Krishi Saarthi - Multi-Crop Disease Detection System
Entry Point: Pixel Inspector CLI
===============================================================================

This script serves as the command-line entry point for inspecting image pixel arrays.
All core logic resides in `pixel_inspector.py`.
"""

import sys
from pathlib import Path

# Ensure src directory is in sys.path for clean relative imports
sys.path.insert(0, str(Path(__file__).parent))

from inspect_image import find_sample_image
from pixel_inspector import (
    PixelMetadata,
    generate_pixel_report,
    get_image_shape,
    get_pixel_statistics,
    inspect_pixels,
    load_pixel_array,
    print_pixel_report,
    sample_pixels,
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
            print(f"ℹ️  No path provided. Defaulting to sample image: {sample.relative_to(datasets_root.parent)}")
            target_image = sample
        else:
            print("Usage: python3 src/image_pixels.py <path_to_image>")
            sys.exit(1)

    metadata = inspect_pixels(target_image)
    print_pixel_report(metadata)


if __name__ == "__main__":
    main()
