import base64
import io
import numpy as np
import tomllib
import matplotlib.pyplot as plt
from supabase import create_client, Client

def get_supabase_client() -> Client:
    """Reads local secrets and connects to Supabase."""
    with open(".streamlit/secrets.toml", "rb") as f:
        secrets = tomllib.load(f)
    return create_client(secrets["SUPABASE_URL"], secrets["SUPABASE_KEY"])

def fetch_and_decode_feedback():
    """Fetches unprocessed feedback from the cloud and decodes it."""
    print("☁️ Connecting to Supabase...")
    supabase = get_supabase_client()
    
    # Query all rows where processed is False
    response = supabase.table("model_feedback").select("*").eq("processed", False).execute()
    data = response.data
    
    if not data:
        print("✅ No new feedback found. Model is up to date!")
        return
        
    print(f"📥 Found {len(data)} new feedback samples! Decoding...")
    
    for row in data:
        actual_label = row["actual_label"]
        
        # Decode Base64 -> Byte Stream -> Original NumPy Array
        buffer = io.BytesIO(base64.b64decode(row["image_data"]))
        img_array = np.load(buffer, allow_pickle=True)
        
        print(f"🎯 Successfully decoded sample for digit '{actual_label}' | Array shape: {img_array.shape}")
        
        # Visualize the first sample
        plt.imshow(img_array, cmap="gray")
        plt.title(f"Cloud Retrieval: Actual Digit '{actual_label}'")
        plt.axis("off")
        plt.show()
        break

if __name__ == "__main__":
    fetch_and_decode_feedback()