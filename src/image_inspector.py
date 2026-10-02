"""
===============================================================================
Krishi Saarthi - Multi-Crop Disease Detection System
Module: Image Inspector (Production-Grade Business Logic)
===============================================================================

This module provides a modular, robust image inspection engine designed to analyze
individual image files before dataset preprocessing or model training.

Architectural Design & Decisions:
---------------------------------
1. Standard Library & Pillow:
   - Built on `pathlib.Path`, `dataclasses`, and `PIL.Image`.
   - Avoids `os` module entirely for clean cross-platform path handling.

2. Dataclass Metadata Representation:
   - `ImageMetadata` stores calculated properties cleanly, enabling downstream
     modules (e.g. dataset validators, augmenters, preprocessing pipelines)
     to consume metadata without re-reading files.

3. Modular Helper Functions:
   - Small, single-responsibility functions (aspect ratio, orientation, size, channels)
     prevent logic duplication and simplify testing.

4. Defensive Error Handling:
   - Handles missing files, unreadable formats, corrupt headers, and permission errors
     gracefully with descriptive custom exceptions and friendly output messages.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, UnidentifiedImageError


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ImageMetadata:
    """
    Data container holding all metadata for an inspected image.

    Attributes:
        file_name (str): Name of the file with extension.
        extension (str): Lowercase file extension (e.g., '.jpg').
        parent_directory (str): Name of the immediate containing folder.
        absolute_path (Path): Resolved absolute path to the file.
        width (int): Pixel width of the image.
        height (int): Pixel height of the image.
        resolution (str): Human-readable resolution string (e.g., '4032 × 3024').
        aspect_ratio_str (str): Simplified integer aspect ratio string (e.g., '4:3').
        aspect_ratio_value (float): Floating-point aspect ratio (width / height).
        channels (int): Number of color/alpha channels (e.g., 3 for RGB).
        mode (str): PIL image mode (e.g., 'RGB', 'RGBA', 'L').
        format (str): Image format as reported by PIL (e.g., 'JPEG', 'PNG').
        size_kb (float): File size in Kilobytes (KB).
        size_mb (float): File size in Megabytes (MB).
        is_square (bool): True if width equals height.
        orientation (str): Image orientation ('Square', 'Landscape', 'Portrait').
    """

    file_name: str
    extension: str
    parent_directory: str
    absolute_path: Path
    width: int
    height: int
    resolution: str
    aspect_ratio_str: str
    aspect_ratio_value: float
    channels: int
    mode: str
    format: str
    size_kb: float
    size_mb: float
    is_square: bool
    orientation: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_image(image_path: Path) -> Image.Image:
    """
    Safely open and verify an image file using Pillow.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        Image.Image: Opened PIL Image object.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        PermissionError: If the file cannot be accessed due to permissions.
        ValueError: If the file path is a directory instead of a file.
        UnidentifiedImageError: If PIL cannot identify or open the image format.
        OSError: If the image file is corrupted or unreadable.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found at: '{image_path}'")

    if not image_path.is_file():
        raise ValueError(f"Specified path is a directory, not a file: '{image_path}'")

    try:
        img = Image.open(image_path)
        img.verify()  # Verify header and structural integrity
        # Re-open after verify() because PIL requires re-opening to access image data
        img = Image.open(image_path)
        return img
    except PermissionError:
        raise PermissionError(f"Permission denied accessing file: '{image_path}'")
    except UnidentifiedImageError:
        raise UnidentifiedImageError(f"Unsupported or invalid image format: '{image_path.name}'")
    except OSError as err:
        raise OSError(f"Corrupted image file '{image_path.name}': {err}")


def calculate_aspect_ratio(width: int, height: int) -> Tuple[str, float]:
    """
    Calculate the simplified aspect ratio string (e.g. '4:3') and float value.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        Tuple[str, float]: (aspect_ratio_str, aspect_ratio_value)
    """
    if width <= 0 or height <= 0:
        return "0:0", 0.0

    gcd = math.gcd(width, height)
    simplified_w = width // gcd
    simplified_h = height // gcd
    ratio_str = f"{simplified_w}:{simplified_h}"
    ratio_val = round(width / height, 4)
    return ratio_str, ratio_val


def calculate_file_size(file_path: Path) -> Tuple[float, float]:
    """
    Calculate file size in Kilobytes (KB) and Megabytes (MB).

    Args:
        file_path (Path): Path to the file.

    Returns:
        Tuple[float, float]: (size_in_kb, size_in_mb)
    """
    num_bytes = file_path.stat().st_size
    size_kb = round(num_bytes / 1024, 2)
    size_mb = round(num_bytes / (1024 * 1024), 2)
    return size_kb, size_mb


def detect_orientation(width: int, height: int) -> str:
    """
    Determine image orientation based on dimensions.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        str: 'Square', 'Landscape', or 'Portrait'.
    """
    if width == height:
        return "Square"
    elif width > height:
        return "Landscape"
    else:
        return "Portrait"


def count_channels(mode: str) -> int:
    """
    Map Pillow image mode string to channel count.

    Args:
        mode (str): PIL image mode (e.g., 'RGB', 'RGBA', 'L', 'CMYK').

    Returns:
        int: Number of color/alpha channels.
    """
    mode_channel_map = {
        "1": 1,         # 1-bit bilevel
        "L": 1,         # 8-bit grayscale
        "P": 1,         # 8-bit palette
        "RGB": 3,       # 3x8-bit color
        "RGBA": 4,      # 4x8-bit color with alpha
        "CMYK": 4,      # 4x8-bit color separation
        "YCbCr": 3,     # 3x8-bit video format
        "LAB": 3,       # L*a*b color space
        "HSV": 3,       # Hue, Saturation, Value
        "I": 1,         # 32-bit signed integer pixels
        "F": 1,         # 32-bit floating point pixels
    }
    return mode_channel_map.get(mode, len(mode))


def inspect_image(image_path: Path | str) -> Optional[ImageMetadata]:
    """
    Main business entrypoint to inspect an image and return structured metadata.

    Args:
        image_path (Path | str): Path to the image file.

    Returns:
        Optional[ImageMetadata]: ImageMetadata dataclass instance if successful, None if an error occurs.
    """
    path = Path(image_path).resolve()

    try:
        with load_image(path) as img:
            width, height = img.size
            aspect_ratio_str, aspect_ratio_val = calculate_aspect_ratio(width, height)
            size_kb, size_mb = calculate_file_size(path)
            orientation = detect_orientation(width, height)
            channels = count_channels(img.mode)
            img_format = img.format if img.format else path.suffix.strip(".").upper()

            return ImageMetadata(
                file_name=path.name,
                extension=path.suffix.lower(),
                parent_directory=path.parent.name,
                absolute_path=path,
                width=width,
                height=height,
                resolution=f"{width} × {height}",
                aspect_ratio_str=aspect_ratio_str,
                aspect_ratio_value=aspect_ratio_val,
                channels=channels,
                mode=img.mode,
                format=img_format,
                size_kb=size_kb,
                size_mb=size_mb,
                is_square=(width == height),
                orientation=orientation,
            )
    except (FileNotFoundError, ValueError, PermissionError, UnidentifiedImageError, OSError) as err:
        print(f"❌ Inspection Failed: {err}")
        return None


def generate_image_report(metadata: ImageMetadata) -> str:
    """
    Format image metadata into a clean, human-readable inspection report.

    Args:
        metadata (ImageMetadata): Inspected image metadata.

    Returns:
        str: Formatted report block string.
    """
    lines = [
        "=" * 44,
        "          IMAGE INSPECTION REPORT           ",
        "=" * 44,
        f"File Name        : {metadata.file_name}",
        f"Extension        : {metadata.extension}",
        f"Parent Directory : {metadata.parent_directory}",
        f"Format           : {metadata.format}",
        f"Mode             : {metadata.mode}",
        f"Channels         : {metadata.channels}",
        f"Resolution       : {metadata.resolution}",
        f"Aspect Ratio     : {metadata.aspect_ratio_str} ({metadata.aspect_ratio_value})",
        f"Orientation      : {metadata.orientation}",
        f"Is Square        : {'Yes' if metadata.is_square else 'No'}",
        f"File Size        : {metadata.size_mb} MB ({metadata.size_kb:,} KB)",
        f"Absolute Path    : {metadata.absolute_path}",
        "=" * 44,
    ]
    return "\n".join(lines)


def print_report(metadata: Optional[ImageMetadata]) -> None:
    """
    Print formatted inspection report to standard output.

    Args:
        metadata (Optional[ImageMetadata]): Metadata to display.
    """
    if metadata:
        print(generate_image_report(metadata))
