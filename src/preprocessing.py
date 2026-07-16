"""
Preprocessing pipeline for DigitVision.
Handles normalization, reshaping, and preparing user-drawn canvas images.
"""

import cv2
import numpy as np


def preprocess_canvas_image(canvas_image: np.ndarray) -> np.ndarray:
    """
    Processes a raw RGBA image from the Streamlit canvas into a (1, 28, 28, 1) tensor.
    Centers the drawn digit using bounding boxes to match MNIST dataset standards.

    Args:
        canvas_image (np.ndarray): RGBA image array (H, W, 4) from the UI canvas.

    Returns:
        np.ndarray: Normalized, reshaped array ready for model inference.
    """
    # 1. Convert RGBA to Grayscale
    # Streamlit drawable canvas returns a 4-channel RGBA array.
    gray = cv2.cvtColor(canvas_image, cv2.COLOR_RGBA2GRAY)
    
    # 2. Invert colors
    # We assume the UI uses a black brush on a white canvas.
    # MNIST requires white digits (pixel values > 0) on a black background (pixel value 0).
    gray = cv2.bitwise_not(gray)
    
    # 3. Find the bounding box of the drawn digit
    # This prevents the digit from being squashed or scaled improperly.
    coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = gray[y:y+h, x:x+w]
        
        # 4. Make it a square by padding the shorter dimension
        length = max(w, h)
        pad_x = (length - w) // 2
        pad_y = (length - h) // 2
        padded_digit = cv2.copyMakeBorder(
            digit, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0
        )
        
        # 5. Resize to 20x20 pixels (MNIST standard for the digit itself)
        resized = cv2.resize(padded_digit, (20, 20), interpolation=cv2.INTER_AREA)
        
        # 6. Pad to 28x28 (MNIST standard image size) with a 4-pixel border
        final_image = cv2.copyMakeBorder(
            resized, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0
        )
    else:
        # Fallback if the canvas is empty
        final_image = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
    # 7. Normalize pixel values to [0.0, 1.0]
    final_image = final_image.astype("float32") / 255.0
    
    # 8. Reshape to include batch and channel dimensions: (1, 28, 28, 1)
    tensor = final_image.reshape(1, 28, 28, 1)
    
    return tensor
