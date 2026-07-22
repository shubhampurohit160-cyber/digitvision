import os
import time
import numpy as np
from src.preprocessing import preprocess_canvas_image

def save_feedback(canvas_image: np.ndarray, correct_label: int) -> None:
    """Saves misclassified user drawings to a feedback directory for batch retraining."""
    # This automatically builds the path: data/feedback/0, data/feedback/1, etc.
    feedback_dir = os.path.join("data", "feedback", str(correct_label))
    
    # exist_ok=True means it will automatically create the folders if they don't exist!
    os.makedirs(feedback_dir, exist_ok=True)
    
    # Preprocess the canvas image exactly how the model expects it
    tensor = preprocess_canvas_image(canvas_image)
    
    # Remove the batch dimension to save it as a simple 28x28 numpy array
    image_array = tensor.squeeze() 
    
    # Save using a unique timestamp to prevent overwriting
    filename = f"user_{int(time.time())}.npy"
    filepath = os.path.join(feedback_dir, filename)
    
    np.save(filepath, image_array)