"""
Defines the Convolutional Neural Network (CNN) architecture for DigitVision.
"""

import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore


def build_model(input_shape: tuple = (28, 28, 1), num_classes: int = 10) -> tf.keras.Model:
    """
    Builds and compiles a CNN for handwritten digit classification.

    Args:
        input_shape (tuple): The shape of the input images. Defaults to (28, 28, 1).
        num_classes (int): The number of output classes (digits 0-9). Defaults to 10.

    Returns:
        tf.keras.Model: A compiled TensorFlow Keras model ready for training.
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First Convolutional Block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Fully Connected Block
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Prevents overfitting
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model