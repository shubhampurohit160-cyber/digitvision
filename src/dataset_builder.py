import os
import base64
import io
import numpy as np
import tomllib
from supabase import create_client, Client

def get_supabase_client() -> Client:
    """Reads local secrets and connects to Supabase."""
    with open(".streamlit/secrets.toml", "rb") as f:
        secrets = tomllib.load(f)
    return create_client(secrets["SUPABASE_URL"], secrets["SUPABASE_KEY"])

def build_dataset():
    print("☁️ Connecting to Supabase...")
    supabase = get_supabase_client()
    
    # 1. Fetch all unprocessed rows
    response = supabase.table("model_feedback").select("*").eq("processed", False).execute()
    data = response.data
    
    if not data:
        print("✅ No new feedback found. The dataset is up to date!")
        return
        
    print(f"📥 Downloading {len(data)} new feedback samples...")
    
    images = []
    labels = []
    processed_ids = []
    
    # 2. Decode and collect the data
    for row in data:
        buffer = io.BytesIO(base64.b64decode(row["image_data"]))
        img_array = np.load(buffer, allow_pickle=True)
        
        images.append(img_array)
        labels.append(row["actual_label"])
        processed_ids.append(row["id"])
        
    # Convert lists to NumPy arrays for TensorFlow
    X_new = np.array(images)
    y_new = np.array(labels)
    
    # 3. Save to a compressed .npz archive in your data folder
    os.makedirs("data", exist_ok=True)
    dataset_path = "data/retrain_dataset.npz"
    
    # If a previous retraining dataset exists, load it and append to it
    if os.path.exists(dataset_path):
        existing_data = np.load(dataset_path)
        X_new = np.concatenate((existing_data['images'], X_new))
        y_new = np.concatenate((existing_data['labels'], y_new))
        
    np.savez(dataset_path, images=X_new, labels=y_new)
    print(f"📦 Packaged {len(processed_ids)} new samples! Total pending dataset size: {len(X_new)} images.")
    
    # 4. Mark the rows as processed in the cloud database
    print("🔄 Marking records as processed in Supabase...")
    for row_id in processed_ids:
        supabase.table("model_feedback").update({"processed": True}).eq("id", row_id).execute()
        
    print("🚀 Dataset build complete! Ready for retraining.")

if __name__ == "__main__":
    build_dataset()