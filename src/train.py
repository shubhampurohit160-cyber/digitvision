import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from PIL import Image

def train_model():
    print("🧠 Initializing DigitVision Retraining Pipeline...")

    # 1. Load the Base MNIST Dataset
    (X_train_base, y_train_base), (X_test, y_test) = mnist.load_data()
    
    # Preprocess base data (normalize to 0-1)
    X_train_base = X_train_base.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    X_train = X_train_base
    y_train = y_train_base
    
    # 2. Inject Active Learning Data (If it exists)
    dataset_path = "data/retrain_dataset.npz"
    if os.path.exists(dataset_path):
        print("🔄 Found cloud feedback dataset! Processing and merging with base MNIST data...")
        custom_data = np.load(dataset_path, allow_pickle=True)
        raw_images = custom_data['images']
        y_custom = custom_data['labels']
        
        processed_images = []
        for img in raw_images:
            # Check if the image is already 28x28
            if img.shape == (28, 28) or img.shape == (28, 28, 1):
                processed_images.append(np.reshape(img, (28, 28)))
            else:
                # It's a raw high-res canvas (e.g., 600x800x4 RGBA)
                # Convert to a Pillow Image, make it Grayscale ('L'), and resize to 28x28
                pil_img = Image.fromarray(img.astype(np.uint8))
                pil_img = pil_img.convert('L').resize((28, 28))
                processed_images.append(np.array(pil_img))
                
        # Normalize custom data identically to base data
        X_custom = np.array(processed_images).astype("float32") / 255.0
        
        # Combine the datasets
        X_train = np.concatenate((X_train_base, X_custom))
        y_train = np.concatenate((y_train_base, y_custom))
        
        # Shuffle the combined dataset
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        print(f"📈 Total Training Samples: {len(X_train)} (Base: {len(X_train_base)} | Cloud: {len(X_custom)})")
    else:
        print("⚠️ No custom dataset found. Training on base MNIST only.")

    # Expand dimensions to include the single grayscale channel (28, 28, 1)
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    # 3. Build the CNN Architecture
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5), # Dropout prevents overfitting on our custom data
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # 4. Train the Model
    print("🚀 Training starting...")
    model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

    # 5. Evaluate on a strict, untouched test set
    print("📊 Evaluating model on untouched test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Final Test Accuracy: {test_acc * 100:.2f}%")

    # 6. Save the new production model (overwriting the old one)
    os.makedirs("models", exist_ok=True)
    model.save("models/model.keras")
    print("💾 New model saved to models/model.keras! Ready for deployment.")

if __name__ == "__main__":
    train_model()