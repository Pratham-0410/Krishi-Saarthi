"""
===============================================================================
Krishi Saarthi - Multi-Crop Disease Detection System
Module: Image Visualizer (Production-Grade Business Logic)
===============================================================================

This module provides the core image visualization and analysis engine designed
to calculate visual properties (Resolution, Mode, Channels, Brightness, Contrast,
Average RGB, Dominant Color) and display images with Matplotlib / PIL fallbacks.

Architectural Design & Decisions:
---------------------------------
1. Re-use of Core Loader:
   - Reuses `load_image()` from `image_inspector.py` for consistent defensive file
     verification and path handling.

2. Dataclass Metadata Representation:
   - `VisualizationMetadata` stores extracted visual metrics and the PIL image reference.

3. Single-Responsibility Helper Functions:
   - Separate modular functions for brightness level, contrast level, color detection,
     report generation, and plot rendering.
"""

import colorsys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from image_inspector import load_image


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VisualizationMetadata:
    """
    Data container holding calculated visual metrics and PIL image object.

    Attributes:
        file_name (str): Name of the image file.
        absolute_path (Path): Resolved absolute path to the file.
        width (int): Pixel width.
        height (int): Pixel height.
        mode (str): PIL image mode (e.g. 'RGB').
        channels (int): Number of color channels.
        brightness (str): Categorized brightness ('Low', 'Medium', 'High').
        contrast (str): Categorized contrast ('Low', 'Medium', 'High').
        avg_r (int): Rounded average Red intensity (0-255).
        avg_g (int): Rounded average Green intensity (0-255).
        avg_b (int): Rounded average Blue intensity (0-255).
        dominant_color (str): Categorized dominant color name.
        image_obj (Image.Image): Prepared PIL RGB Image object.
    """

    file_name: str
    absolute_path: Path
    width: int
    height: int
    mode: str
    channels: int
    brightness: str
    contrast: str
    avg_r: int
    avg_g: int
    avg_b: int
    dominant_color: str
    image_obj: Image.Image


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_brightness_level(mean_intensity: float) -> str:
    """
    Categorize brightness based on average pixel intensity.

    Args:
        mean_intensity (float): Mean pixel intensity (0 to 255).

    Returns:
        str: 'Low', 'Medium', or 'High'.
    """
    if mean_intensity < 85:
        return "Low"
    elif mean_intensity <= 170:
        return "Medium"
    else:
        return "High"


def get_contrast_level(std_intensity: float) -> str:
    """
    Categorize contrast based on standard deviation of pixel intensities.

    Args:
        std_intensity (float): Standard deviation of pixel intensities.

    Returns:
        str: 'Low', 'Medium', or 'High'.
    """
    if std_intensity < 40:
        return "Low"
    elif std_intensity <= 75:
        return "Medium"
    else:
        return "High"


def detect_dominant_color(avg_r: int, avg_g: int, avg_b: int) -> str:
    """
    Determine the primary dominant color from average RGB values using HSV conversion.

    Args:
        avg_r (int): Average Red intensity (0-255).
        avg_g (int): Average Green intensity (0-255).
        avg_b (int): Average Blue intensity (0-255).

    Returns:
        str: Descriptive dominant color name.
    """
    norm_r = avg_r / 255.0
    norm_g = avg_g / 255.0
    norm_b = avg_b / 255.0

    hue, sat, val = colorsys.rgb_to_hsv(norm_r, norm_g, norm_b)
    hue_deg = hue * 360.0

    # Low saturation or extreme values map to grayscale
    if sat < 0.15:
        if val < 0.2:
            return "Black"
        elif val > 0.85:
            return "White"
        else:
            return "Gray"

    # Map Hue angle to color categories
    if hue_deg < 15 or hue_deg >= 345:
        return "Red"
    elif 15 <= hue_deg < 45:
        return "Brown" if val < 0.6 else "Orange"
    elif 45 <= hue_deg < 70:
        return "Yellow"
    elif 70 <= hue_deg < 165:
        return "Green"
    elif 165 <= hue_deg < 255:
        return "Blue"
    elif 255 <= hue_deg < 290:
        return "Purple"
    else:
        return "Pink"


def analyze_image_visualization(image_path: Path | str) -> Optional[VisualizationMetadata]:
    """
    Main business entrypoint to analyze an image for visualization.

    Args:
        image_path (Path | str): Path to the image file.

    Returns:
        Optional[VisualizationMetadata]: Metadata dataclass instance if successful, None if failed.
    """
    path = Path(image_path).resolve()

    try:
        with load_image(path) as img:
            width, height = img.size
            mode = img.mode

            mode_channels = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4}
            channels = mode_channels.get(mode, len(mode))

            rgb_img = img.convert("RGB")
            arr = np.array(rgb_img)

            avg_r = int(round(float(arr[:, :, 0].mean())))
            avg_g = int(round(float(arr[:, :, 1].mean())))
            avg_b = int(round(float(arr[:, :, 2].mean())))

            gray_arr = arr.mean(axis=2)
            mean_intensity = float(gray_arr.mean())
            std_intensity = float(gray_arr.std())

            brightness = get_brightness_level(mean_intensity)
            contrast = get_contrast_level(std_intensity)
            dominant_color = detect_dominant_color(avg_r, avg_g, avg_b)

            return VisualizationMetadata(
                file_name=path.name,
                absolute_path=path,
                width=width,
                height=height,
                mode=mode,
                channels=channels,
                brightness=brightness,
                contrast=contrast,
                avg_r=avg_r,
                avg_g=avg_g,
                avg_b=avg_b,
                dominant_color=dominant_color,
                image_obj=rgb_img,
            )
    except Exception as err:
        print(f"❌ Image Visualization Analysis Failed: {err}")
        return None


def generate_visualization_report(metadata: VisualizationMetadata) -> str:
    """
    Format visualization metadata into exact specified output structure.

    Args:
        metadata (VisualizationMetadata): Inspected visualization metadata.

    Returns:
        str: Formatted report block string.
    """
    return (
        "===================================\n\n"
        "IMAGE ANALYSIS\n\n"
        "===================================\n\n"
        f"Resolution : {metadata.width} × {metadata.height}\n\n"
        f"Mode       : {metadata.mode}\n\n"
        f"Channels   : {metadata.channels}\n\n"
        f"Brightness : {metadata.brightness}\n\n"
        f"Contrast   : {metadata.contrast}\n\n"
        "Average RGB:\n\n"
        f"R : {metadata.avg_r}\n\n"
        f"G : {metadata.avg_g}\n\n"
        f"B : {metadata.avg_b}\n\n"
        "Dominant Color:\n\n"
        f"{metadata.dominant_color}\n\n"
        "Displaying Image.."
    )


def print_visualization_report(metadata: Optional[VisualizationMetadata]) -> None:
    """
    Print formatted visualization report to standard output.

    Args:
        metadata (Optional[VisualizationMetadata]): Metadata to display.
    """
    if metadata:
        print(generate_visualization_report(metadata))


def display_image(metadata: Optional[VisualizationMetadata]) -> None:
    """
    Display the image visually using Matplotlib with PIL fallback.

    Args:
        metadata (Optional[VisualizationMetadata]): Metadata containing PIL image.
    """
    if not metadata:
        return

    plt.figure(figsize=(8, 6))
    plt.imshow(metadata.image_obj)
    plt.title(
        f"{metadata.file_name}\nResolution: {metadata.width}x{metadata.height} | "
        f"Dominant Color: {metadata.dominant_color}",
        fontsize=12,
        pad=10,
    )
    plt.axis("off")
    plt.tight_layout()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            plt.show()
        except Exception:
            metadata.image_obj.show()
