import base64
import io
import numpy as np
import streamlit as st
from supabase import create_client, Client

@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

def save_feedback(image_data: np.ndarray, actual_label: int, predicted_label: int, confidence: float) -> None:
    """Encodes misclassified user drawings cleanly with NumPy headers and uploads them to Supabase."""
    
    # 1. Package NumPy array into a byte stream (preserves exact shape & dtype)
    buffer = io.BytesIO()
    np.save(buffer, image_data)
    
    # 2. Encode to Base64 string
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 3. Build telemetry payload
    db_row = {
        "actual_label": actual_label,
        "predicted_label": predicted_label,
        "confidence": float(confidence),
        "image_data": encoded_image
    }
    
    # 4. Push to cloud
    supabase.table("model_feedback").insert(db_row).execute()