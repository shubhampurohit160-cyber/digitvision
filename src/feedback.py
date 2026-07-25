import base64
import numpy as np
import streamlit as st
from supabase import create_client, Client

# 1. Initialize the Supabase connection using Streamlit's secure secrets
@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

def save_feedback(image_data: np.ndarray, actual_label: int, predicted_label: int, confidence: float) -> None:
    """Encodes misclassified user drawings and uploads them to the Supabase feedback queue."""
    
    # 2. Serialize the NumPy array to bytes, then to a Base64 string
    img_bytes = image_data.tobytes()
    encoded_image = base64.b64encode(img_bytes).decode('utf-8')
    
    # 3. Build the row exactly as we defined it in your SQL schema
    db_row = {
        "actual_label": actual_label,
        "predicted_label": predicted_label,
        "confidence": float(confidence), # Ensure it's a standard float, not a numpy float
        "image_data": encoded_image
    }
    
    # 4. Insert the row into the cloud database
    supabase.table("model_feedback").insert(db_row).execute()