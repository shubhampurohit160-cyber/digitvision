"""
Training and retraining pipeline for the DigitVision CNN model.
Handles loading MNIST, incorporating user feedback data, training, evaluation, 
and artifact saving.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from src.model import build_model


def load_feedback_data(feedback_dir: str = os.path.join("data", "feedback")) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Loads user-corrected edge cases saved from the Streamlit UI feedback loop.

    Args:
        feedback_dir (str): Base directory where feedback .npy files are saved.

    Returns:
        tuple: (x_feedback, y_feedback) numpy arrays, or (None, None) if no feedback exists.
    """
    x_feedback = []
    y_feedback = []

    if not os.path.exists(feedback_dir):
        return None, None

    for label in range(10):
        folder = os.path.join(feedback_dir, str(label))
        if os.path.exists(folder):
            for filepath in glob.glob(os.path.join(folder, "*.npy")):
                img = np.load(filepath)
                x_feedback.append(img)
                y_feedback.append(label)

    if not x_feedback:
        return None, None

    # Reshape to match model input: (N, 28, 28, 1)
    x_array = np.array(x_feedback, dtype="float32")[..., np.newaxis]
    y_array = np.array(y_feedback, dtype="int32")

    return x_array, y_array


def save_training_history(history: dict, save_dir: str = "reports/") -> None:
    """Plots and saves training history curves to the reports directory."""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_path = os.path.join(save_dir, "training_history.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training history plot saved to {plot_path}")


def main():
    """Main execution block for training or fine-tuning the model."""
    print("Loading base MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess MNIST
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Check for User Feedback Data
    x_feed, y_feed = load_feedback_data()
    if x_feed is not None:
        print(f"Found {len(x_feed)} custom user feedback samples!")
        # Concatenate feedback data with standard MNIST training data
        x_train = np.concatenate((x_train, x_feed), axis=0)
        y_train = np.concatenate((y_train, y_feed), axis=0)
    else:
        print("No user feedback data found. Training on standard MNIST dataset.")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "digitvision_model.keras")

    # If an existing model exists, load it for fine-tuning; otherwise build a new one
    if os.path.exists(model_path):
        print(f"Existing model found at {model_path}. Loading for fine-tuning...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Building new CNN model architecture...")
        model = build_model(input_shape=(28, 28, 1), num_classes=10)

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    print("Starting training...")
    history = model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    print("Evaluating model performance on test dataset...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    print(f"Saving updated model to {model_path}...")
    model.save(model_path)

    print("Saving training graphs...")
    save_training_history(history.history)
    print("Training pipeline execution completed successfully.")


if __name__ == "__main__":
    main()