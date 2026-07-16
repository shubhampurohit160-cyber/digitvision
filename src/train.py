"""
Training pipeline for the DigitVision CNN model.
Handles data loading, training, evaluation, and artifact saving.
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from src.model import build_model
# Note: preprocessing will be implemented in Phase 5
# from src.preprocessing import preprocess_for_training


def save_training_history(history: dict, save_dir: str = "reports/") -> None:
    """
    Plots and saves training history (accuracy and loss curves) to the reports directory.
    
    Args:
        history (dict): The history dictionary from the trained Keras model.
        save_dir (str): Directory to save the generated plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot Accuracy
    plt.figure(figsize=(10, 4))
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

    # Save artifact
    plot_path = os.path.join(save_dir, "training_history.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training history plot saved to {plot_path}")


def main():
    """Main execution block for training the model."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocessing (Inline for now, will map to src.preprocessing in Phase 5 to ensure inference match)
    print("Preprocessing data...")
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    print("Building model...")
    model = build_model(input_shape=(28, 28, 1), num_classes=10)
    model.summary()

    # Setup callbacks
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "digitvision_model.keras")
    
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

    print("Evaluating on test data...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    print(f"Saving model to {model_path}...")
    model.save(model_path)
    
    print("Generating training reports...")
    save_training_history(history.history)
    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()