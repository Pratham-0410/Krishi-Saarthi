"""
===============================================================================
Krishi Saarthi - Multi-Crop Disease Detection System
Module: Pixel Inspector (NumPy-based Pixel-Level Analysis)
===============================================================================

This module provides a modular, robust pixel inspection engine designed to analyze
the raw numeric pixel arrays of images using NumPy before model preprocessing or training.

Architectural Design & Decisions:
---------------------------------
1. Re-use of Core Loader:
   - Reuses `load_image()` from `image_inspector.py` to maintain consistent, defensive
     file verification, error handling, and cross-platform path handling (`pathlib.Path`).

2. NumPy Array Representation:
   - Converts PIL Image objects into standard NumPy ndarrays (`np.asarray`) for fast,
     vectorized numeric inspection.

3. Dataclass Metadata Representation:
   - `PixelMetadata` stores raw shape, datatype, statistical distribution (min/max/mean/std),
     and spatial pixel samples (top-left, center, bottom-right).

4. Modular Helper Functions:
   - Single-responsibility functions (`load_pixel_array`, `get_image_shape`,
     `get_pixel_statistics`, `sample_pixels`, `generate_pixel_report`) for maximum reusability.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from image_inspector import load_image


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PixelMetadata:
    """
    Data container holding pixel-level metadata and statistical properties of an image.

    Attributes:
        file_name (str): Name of the inspected image file.
        absolute_path (Path): Resolved absolute path to the file.
        shape (Tuple[int, ...]): Dimensions of the NumPy array (Height, Width, [Channels]).
        dtype (str): Data type of pixel array (e.g., 'uint8', 'float32').
        min_val (Union[int, float]): Minimum pixel intensity value across all channels.
        max_val (Union[int, float]): Maximum pixel intensity value across all channels.
        mean_val (float): Mean pixel intensity value across all pixels and channels.
        std_val (float): Standard deviation of pixel intensities.
        sample_top_left (Any): Pixel value at top-left corner (0, 0).
        sample_center (Any): Pixel value at center (Height // 2, Width // 2).
        sample_bottom_right (Any): Pixel value at bottom-right corner (Height - 1, Width - 1).
    """

    file_name: str
    absolute_path: Path
    shape: Tuple[int, ...]
    dtype: str
    min_val: Union[int, float]
    max_val: Union[int, float]
    mean_val: float
    std_val: float
    sample_top_left: Any
    sample_center: Any
    sample_bottom_right: Any


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_pixel_array(image_path: Path | str) -> np.ndarray:
    """
    Load an image from disk and convert it into a NumPy pixel array.
    Reuses `load_image` from `image_inspector` for defensive error handling.

    Args:
        image_path (Path | str): Path to the image file.

    Returns:
        np.ndarray: NumPy array representation of the image pixels.
    """
    path = Path(image_path).resolve()
    with load_image(path) as img:
        return np.asarray(img)


def get_image_shape(pixel_array: np.ndarray) -> Tuple[int, ...]:
    """
    Get the dimensions/shape of the pixel array.

    Args:
        pixel_array (np.ndarray): Image pixel array.

    Returns:
        Tuple[int, ...]: Shape tuple (Height, Width, [Channels]).
    """
    return pixel_array.shape


def get_pixel_statistics(pixel_array: np.ndarray) -> Dict[str, Any]:
    """
    Calculate summary statistics for the pixel array using NumPy.

    Args:
        pixel_array (np.ndarray): Image pixel array.

    Returns:
        Dict[str, Any]: Dictionary containing min_val, max_val, mean_val, std_val, and dtype.
    """
    min_val = pixel_array.min()
    max_val = pixel_array.max()
    mean_val = round(float(pixel_array.mean()), 4)
    std_val = round(float(pixel_array.std()), 4)

    if isinstance(min_val, np.generic):
        min_val = min_val.item()
    if isinstance(max_val, np.generic):
        max_val = max_val.item()

    return {
        "dtype": str(pixel_array.dtype),
        "min_val": min_val,
        "max_val": max_val,
        "mean_val": mean_val,
        "std_val": std_val,
    }


def sample_pixels(pixel_array: np.ndarray) -> Dict[str, Any]:
    """
    Extract sample pixel values from top-left, center, and bottom-right locations.

    Args:
        pixel_array (np.ndarray): Image pixel array.

    Returns:
        Dict[str, Any]: Dictionary containing top_left, center, and bottom_right pixel samples.
    """
    height, width = pixel_array.shape[:2]

    top_left = pixel_array[0, 0]
    center = pixel_array[height // 2, width // 2]
    bottom_right = pixel_array[height - 1, width - 1]

    def _clean_sample(val: Any) -> Any:
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, np.generic):
            return val.item()
        return val

    return {
        "top_left": _clean_sample(top_left),
        "center": _clean_sample(center),
        "bottom_right": _clean_sample(bottom_right),
    }


def inspect_pixels(image_path: Path | str) -> Optional[PixelMetadata]:
    """
    Main business entrypoint to perform pixel-level inspection of an image.

    Args:
        image_path (Path | str): Path to the image file.

    Returns:
        Optional[PixelMetadata]: PixelMetadata dataclass instance if successful, None if an error occurs.
    """
    path = Path(image_path).resolve()
    try:
        pixel_array = load_pixel_array(path)
        shape = get_image_shape(pixel_array)
        stats = get_pixel_statistics(pixel_array)
        samples = sample_pixels(pixel_array)

        return PixelMetadata(
            file_name=path.name,
            absolute_path=path,
            shape=shape,
            dtype=stats["dtype"],
            min_val=stats["min_val"],
            max_val=stats["max_val"],
            mean_val=stats["mean_val"],
            std_val=stats["std_val"],
            sample_top_left=samples["top_left"],
            sample_center=samples["center"],
            sample_bottom_right=samples["bottom_right"],
        )
    except Exception as err:
        print(f"❌ Pixel Inspection Failed: {err}")
        return None


def generate_pixel_report(metadata: PixelMetadata) -> str:
    """
    Format pixel metadata into a clean, human-readable inspection report.

    Args:
        metadata (PixelMetadata): Inspected pixel metadata.

    Returns:
        str: Formatted report block string.
    """
    center_y = metadata.shape[0] // 2
    center_x = metadata.shape[1] // 2
    br_y = metadata.shape[0] - 1
    br_x = metadata.shape[1] - 1

    lines = [
        "=" * 44,
        "          PIXEL INSPECTION REPORT           ",
        "=" * 44,
        f"File Name        : {metadata.file_name}",
        f"Shape            : {metadata.shape}",
        f"Data Type        : {metadata.dtype}",
        f"Min / Max Values : {metadata.min_val} / {metadata.max_val}",
        f"Mean Pixel Value : {metadata.mean_val}",
        f"Std Pixel Value  : {metadata.std_val}",
        "-" * 44,
        "Sample Pixel Values:",
        f"  - Top-Left     (0, 0)       : {metadata.sample_top_left}",
        f"  - Center       ({center_y}, {center_x})   : {metadata.sample_center}",
        f"  - Bottom-Right ({br_y}, {br_x}) : {metadata.sample_bottom_right}",
        f"Absolute Path    : {metadata.absolute_path}",
        "=" * 44,
    ]
    return "\n".join(lines)


def print_pixel_report(metadata: Optional[PixelMetadata]) -> None:
    """
    Print formatted pixel inspection report to standard output.

    Args:
        metadata (Optional[PixelMetadata]): Metadata to display.
    """
    if metadata:
        print(generate_pixel_report(metadata))
