"""
===============================================================================
Krishi Saarthi - Multi-Crop Disease Detection System
Module: Image Preprocessor (CNN-Ready Preprocessing Pipeline)
===============================================================================

This module provides a modular, robust image preprocessing pipeline designed to
standardize input images into a consistent format suitable for CNN model training
and evaluation.

Architectural Design & Decisions:
---------------------------------
1. Standard Library & Core Reusability:
   - Built on `pathlib.Path`, `dataclasses`, `PIL.Image`, and `numpy`.
   - Reuses defensive loader `load_image()` from `image_inspector.py` to maintain
     consistent file verification and error handling.

2. Aspect-Ratio Preserving Resize & Safe Padding:
   - Avoids image distortion/stretching by proportionally scaling the image to fit
     within the target bounding box (default: 224 × 224) and applying uniform
     padding (letterboxing/pillarboxing) around the edges.
   - Ensures no leaf or lesion features are cropped.

3. Configurable Normalization & Tensor Readiness:
   - Converts standard 8-bit [0, 255] RGB pixel values into [0.0, 1.0] `float32`
     NumPy arrays, ready for downstream CNN model input.

4. Dataclass Metadata Representation:
   - `PreprocessingMetadata` stores comprehensive transformation metrics alongside
     both the processed PIL Image and normalized NumPy array.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image, UnidentifiedImageError

from image_inspector import load_image


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PreprocessingMetadata:
    """
    Data container holding preprocessed image attributes, transformation details,
    and resulting image/array representations.

    Attributes:
        file_name (str): Name of the source image file.
        absolute_path (Path): Resolved absolute path to the source image.
        original_width (int): Width of the input image before processing.
        original_height (int): Height of the input image before processing.
        original_mode (str): Color mode of the input image (e.g., 'RGB', 'RGBA', 'L').
        target_size (Tuple[int, int]): Configured target dimensions (width, height).
        resized_width (int): Width after aspect-ratio-preserving proportional resize.
        resized_height (int): Height after aspect-ratio-preserving proportional resize.
        processed_width (int): Final pixel width after padding (equals target width).
        processed_height (int): Final pixel height after padding (equals target height).
        padding_applied (Tuple[int, int, int, int]): Pixels added as (left, top, right, bottom).
        is_normalized (bool): True if pixel values were normalized to [0.0, 1.0].
        array_shape (Tuple[int, ...]): Final NumPy array shape (Height, Width, Channels).
        dtype (str): Data type of the final NumPy array (e.g., 'float32').
        min_val (float): Minimum pixel value in the processed array.
        max_val (float): Maximum pixel value in the processed array.
        mean_val (float): Mean pixel value in the processed array.
        std_val (float): Standard deviation of pixel values in the processed array.
        processed_image: Image.Image: Final processed PIL Image object.
        processed_array: np.ndarray: Final processed NumPy array ready for model input.
    """

    file_name: str
    absolute_path: Path
    original_width: int
    original_height: int
    original_mode: str
    target_size: Tuple[int, int]
    resized_width: int
    resized_height: int
    processed_width: int
    processed_height: int
    padding_applied: Tuple[int, int, int, int]
    is_normalized: bool
    array_shape: Tuple[int, ...]
    dtype: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    processed_image: Image.Image
    processed_array: np.ndarray


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_and_load_image(image_path: Path | str) -> Image.Image:
    """
    Validate and safely load an image file from disk.
    Reuses `load_image` from `image_inspector` for defensive verification.

    Args:
        image_path (Path | str): Path to the image file.

    Returns:
        Image.Image: Opened PIL Image object.
    """
    path = Path(image_path).resolve()
    return load_image(path)


def convert_to_rgb(img: Image.Image) -> Image.Image:
    """
    Convert a PIL Image to RGB mode if it is not already in RGB format.

    Args:
        img (Image.Image): Input PIL Image object.

    Returns:
        Image.Image: PIL Image converted to RGB mode.
    """
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def calculate_resized_dimensions(
    original_width: int,
    original_height: int,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[int, int]:
    """
    Calculate new dimensions that fit within target_size while strictly
    preserving the original aspect ratio.

    Args:
        original_width (int): Original pixel width.
        original_height (int): Original pixel height.
        target_size (Tuple[int, int]): Target bounding box as (target_width, target_height).

    Returns:
        Tuple[int, int]: Proportionally resized dimensions as (new_width, new_height).
    """
    target_width, target_height = target_size

    if original_width <= 0 or original_height <= 0:
        return target_width, target_height

    scale = min(target_width / original_width, target_height / original_height)
    new_width = max(1, int(round(original_width * scale)))
    new_height = max(1, int(round(original_height * scale)))

    return new_width, new_height


def resize_image(
    img: Image.Image,
    new_size: Tuple[int, int],
) -> Image.Image:
    """
    Resize a PIL Image to specified dimensions using high-quality bilinear interpolation.

    Args:
        img (Image.Image): PIL Image to resize.
        new_size (Tuple[int, int]): New dimensions as (width, height).

    Returns:
        Image.Image: Resized PIL Image.
    """
    resample_filter = getattr(Image, "Resampling", Image).BILINEAR
    return img.resize(new_size, resample=resample_filter)


def add_padding(
    img: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    fill_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Pad an image centered within a target canvas without cropping any region.

    Args:
        img (Image.Image): Proportionally resized PIL Image.
        target_size (Tuple[int, int]): Desired canvas size as (target_width, target_height).
        fill_color (Tuple[int, int, int]): RGB background color for padding (default black).

    Returns:
        Tuple[Image.Image, Tuple[int, int, int, int]]:
            (padded_image, (pad_left, pad_top, pad_right, pad_bottom))
    """
    target_width, target_height = target_size
    current_width, current_height = img.size

    pad_w = max(0, target_width - current_width)
    pad_h = max(0, target_height - current_height)

    pad_left = pad_w // 2
    pad_top = pad_h // 2
    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    padded_img = Image.new("RGB", (target_width, target_height), fill_color)
    padded_img.paste(img, (pad_left, pad_top))

    padding_applied = (pad_left, pad_top, pad_right, pad_bottom)
    return padded_img, padding_applied


def image_to_numpy_array(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a float32 NumPy array.

    Args:
        img (Image.Image): Input PIL Image.

    Returns:
        np.ndarray: NumPy array with shape (Height, Width, Channels) and dtype float32.
    """
    return np.asarray(img, dtype=np.float32)


def normalize_pixels(pixel_array: np.ndarray) -> np.ndarray:
    """
    Normalize raw pixel values from [0, 255] to [0.0, 1.0].

    Args:
        pixel_array (np.ndarray): Input NumPy array with values in [0, 255].

    Returns:
        np.ndarray: Normalized float32 array with values in [0.0, 1.0].
    """
    return (pixel_array / 255.0).astype(np.float32)


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

def preprocess_image(
    image_path: Path | str,
    target_size: Tuple[int, int] = (224, 224),
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    normalize: bool = True,
) -> Optional[PreprocessingMetadata]:
    """
    Complete image preprocessing pipeline:
    1. Validate & load image safely
    2. Extract original metadata (dimensions, mode)
    3. Convert to RGB
    4. Proportional aspect-ratio-preserving resize
    5. Centered padding to exact target size
    6. Convert to NumPy array
    7. Normalize pixel values to [0.0, 1.0]
    8. Package and return PreprocessingMetadata

    Args:
        image_path (Path | str): Path to the image file.
        target_size (Tuple[int, int]): Target dimensions as (width, height). Defaults to (224, 224).
        pad_color (Tuple[int, int, int]): Background RGB fill color for padding. Defaults to (0, 0, 0).
        normalize (bool): Whether to scale pixel values to [0.0, 1.0]. Defaults to True.

    Returns:
        Optional[PreprocessingMetadata]: Dataclass instance with processed outputs, or None if error.
    """
    path = Path(image_path).resolve()

    try:
        # Step 1: Validate and load image
        with validate_and_load_image(path) as raw_img:
            orig_w, orig_h = raw_img.size
            orig_mode = raw_img.mode

            # Step 2: Convert to RGB
            rgb_img = convert_to_rgb(raw_img)

            # Step 3: Calculate aspect-ratio-preserving resized dimensions
            new_w, new_h = calculate_resized_dimensions(orig_w, orig_h, target_size)

            # Step 4: Proportional resize
            resized_img = resize_image(rgb_img, (new_w, new_h))

            # Step 5: Add padding to reach target_size exactly
            padded_img, padding_applied = add_padding(resized_img, target_size, fill_color=pad_color)

            # Step 6: Convert to NumPy array
            array = image_to_numpy_array(padded_img)

            # Step 7: Normalize pixel intensities
            if normalize:
                processed_array = normalize_pixels(array)
            else:
                processed_array = array

            # Calculate statistical properties
            min_val = round(float(processed_array.min()), 4)
            max_val = round(float(processed_array.max()), 4)
            mean_val = round(float(processed_array.mean()), 4)
            std_val = round(float(processed_array.std()), 4)

            return PreprocessingMetadata(
                file_name=path.name,
                absolute_path=path,
                original_width=orig_w,
                original_height=orig_h,
                original_mode=orig_mode,
                target_size=target_size,
                resized_width=new_w,
                resized_height=new_h,
                processed_width=padded_img.width,
                processed_height=padded_img.height,
                padding_applied=padding_applied,
                is_normalized=normalize,
                array_shape=processed_array.shape,
                dtype=str(processed_array.dtype),
                min_val=min_val,
                max_val=max_val,
                mean_val=mean_val,
                std_val=std_val,
                processed_image=padded_img,
                processed_array=processed_array,
            )

    except (FileNotFoundError, ValueError, PermissionError, UnidentifiedImageError, OSError) as err:
        print(f"❌ Preprocessing Failed: {err}")
        return None
    except Exception as err:
        print(f"❌ Unexpected Error Preprocessing Image: {err}")
        return None


# =============================================================================
# REPORTING FUNCTIONS
# =============================================================================

def generate_preprocessing_report(metadata: PreprocessingMetadata) -> str:
    """
    Format preprocessing metadata into a clean, human-readable report.

    Args:
        metadata (PreprocessingMetadata): Metadata of preprocessed image.

    Returns:
        str: Formatted report block string.
    """
    pad_left, pad_top, pad_right, pad_bottom = metadata.padding_applied
    pad_str = f"L: {pad_left}px, T: {pad_top}px, R: {pad_right}px, B: {pad_bottom}px"

    lines = [
        "=" * 44,
        "        IMAGE PREPROCESSING REPORT          ",
        "=" * 44,
        f"File Name            : {metadata.file_name}",
        f"Original Dimensions  : {metadata.original_width} × {metadata.original_height}",
        f"Original Mode        : {metadata.original_mode}",
        f"Target Dimensions    : {metadata.target_size[0]} × {metadata.target_size[1]}",
        f"Resized Dimensions   : {metadata.resized_width} × {metadata.resized_height}",
        f"Processed Dimensions : {metadata.processed_width} × {metadata.processed_height}",
        f"Aspect Ratio         : Preserved (No Distortion)",
        f"Padding Applied      : {pad_str}",
        f"Normalization (0-1)  : {'Enabled (pixel / 255.0)' if metadata.is_normalized else 'Disabled'}",
        f"Final Array Shape    : {metadata.array_shape}",
        f"Data Type            : {metadata.dtype}",
        f"Pixel Value Range    : [{metadata.min_val}, {metadata.max_val}]",
        f"Mean / Std Value     : {metadata.mean_val} / {metadata.std_val}",
        f"Absolute Path        : {metadata.absolute_path}",
        "=" * 44,
    ]
    return "\n".join(lines)


def print_preprocessing_report(metadata: Optional[PreprocessingMetadata]) -> None:
    """
    Print formatted preprocessing report to standard output.

    Args:
        metadata (Optional[PreprocessingMetadata]): Metadata to display.
    """
    if metadata:
        print(generate_preprocessing_report(metadata))
