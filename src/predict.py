"""
Inference module for DigitVision.
Handles model prediction and extracting confidence scores.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict


def predict_digit(
    model: tf.keras.Model, 
    preprocessed_image: np.ndarray
) -> Tuple[int, float, List[Dict[str, float]], np.ndarray]:
    """
    Predicts the digit from a preprocessed image tensor.

    Args:
        model (tf.keras.Model): The loaded, trained Keras model.
        preprocessed_image (np.ndarray): The (1, 28, 28, 1) image tensor.

    Returns:
        Tuple containing:
            - int: The predicted digit (0-9).
            - float: The confidence score of the top prediction.
            - List[Dict]: The top 3 predictions with their respective confidences.
            - np.ndarray: The full probability array for all 10 classes.
    """
    # 1. Run inference
    predictions = model.predict(preprocessed_image, verbose=0)
    probabilities = predictions[0]  # Extract the first (and only) batch result
    
    # 2. Extract top prediction
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    
    # 3. Extract Top 3 predictions for the UI
    # argsort returns indices in ascending order, so we slice the last 3 and reverse
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3 = [
        {"digit": int(i), "confidence": float(probabilities[i])} 
        for i in top_3_indices
    ]
    
    return predicted_class, confidence, top_3, probabilities