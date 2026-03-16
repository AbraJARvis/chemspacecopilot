#!/usr/bin/env python
# coding: utf-8
"""
Shared utility functions for the cs_copilot tools package.
"""

import base64
import logging
from pathlib import Path
from typing import Any, Callable

from cs_copilot.storage import S3

from ..constants import IMAGE_MIME_TYPES

logger = logging.getLogger(__name__)


def get_file_extension(file_path: str) -> str:
    """Get file extension from path."""
    return Path(file_path).suffix.lower()


def get_mime_type(file_path: str) -> str:
    """Get MIME type for image file based on extension."""
    ext = get_file_extension(file_path)
    return IMAGE_MIME_TYPES.get(ext, "image/png")


def validate_positive_int(value: int, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def safe_file_operation(
    operation: Callable[..., Any], file_path: str, *args: Any, **kwargs: Any
) -> Any:
    """Safely perform file operations with proper error handling."""
    try:
        return operation(file_path, *args, **kwargs)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied accessing file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error performing file operation on {file_path}: {e}")
        raise


def convert_s3_image_to_base64(s3_path: str) -> str:
    """Convert S3 image to base64."""
    logger.debug(f"Reading image from S3: {s3_path}")

    with S3.open(s3_path, "rb") as img_file:
        img_data = img_file.read()

    if not img_data:
        raise ValueError(f"S3 image file is empty: {s3_path}")

    b64 = base64.b64encode(img_data).decode()
    mime_type = get_mime_type(s3_path)

    return f"data:{mime_type};base64,{b64}"


def convert_local_image_to_base64(image_path: str) -> str:
    """Convert local image to base64."""
    # Resolve the path
    img_path = Path(image_path).resolve()

    logger.debug(f"Reading local image: {img_path}")

    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # Read and encode the image
    with open(img_path, "rb") as img_file:
        img_data = img_file.read()

    if not img_data:
        raise ValueError(f"Local image file is empty: {img_path}")

    b64 = base64.b64encode(img_data).decode()
    mime_type = get_mime_type(str(img_path))

    return f"data:{mime_type};base64,{b64}"


def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 encoded string for display.

    This function handles both local files and S3 URLs automatically.

    Args:
        image_path: Path to the image file (can be relative, absolute, or S3 URL)

    Returns:
        Base64 encoded image data with data URL format (data:image/png;base64,...)

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the file path is empty or conversion fails
    """
    if not image_path:
        raise ValueError("image_path cannot be empty")

    logger.info(f"Converting image to base64: {image_path}")

    try:
        # Handle S3 URLs
        if str(image_path).startswith("s3://"):
            return convert_s3_image_to_base64(image_path)

        # Handle local files
        return convert_local_image_to_base64(image_path)

    except Exception as e:
        logger.error(f"Failed to convert image to base64: {e}")
        raise ValueError(f"Failed to convert image to base64: {e}") from e
