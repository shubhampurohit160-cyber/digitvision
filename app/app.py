"""
Main entry point for the DigitVision Streamlit Application.
Provides a premium UI for drawing and predicting handwritten digits.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from src.feedback import save_feedback  # Import the feedback saving function

# Ensure the root directory is in the path for modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict_digit
from src.preprocessing import preprocess_canvas_image
import tensorflow as tf

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DigitVision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM FEEL ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2e7bcf;
        color: white;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #1e5faf; border: none; }
    .prediction-card {
        background-color: #1e2130;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #2e7bcf;
        margin-bottom: 20px;
    }
    .digit-display {
        font-size: 80px;
        font-weight: bold;
        color: #deff9a;
        text-align: center;
        line-height: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    # List of possible paths to handle case-sensitivity and naming variants on Linux
    possible_paths = [
        os.path.join("models", "digitvision_model.keras"),
        os.path.join("model", "digitvision_model.keras"),
        os.path.join(os.path.dirname(__file__), "..", "models", "digitvision_model.keras"),
        os.path.join(os.path.dirname(__file__), "..", "model", "digitvision_model.keras")
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
            
    if not model_path:
        st.error("🤖 AI Model file not found on the server. Looked in: " + str(possible_paths))
        return None
        
    return tf.keras.models.load_model(model_path)

model = load_trained_model()

# --- SIDEBAR ---
with st.sidebar:
    st.title("👁️ DigitVision")
    st.info("A production-grade Handwritten Digit Recognizer powered by a Deep CNN.")
    
    st.subheader("Model Info")
    st.text("Architecture: CNN")
    st.text("Dataset: MNIST")
    st.text("Accuracy: ~99.2%")
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    st.subheader("Prediction History")
    for item in reversed(st.session_state.history[-5:]):
        st.write(f"Digit: **{item['digit']}** ({item['conf']:.1%})")

# --- MAIN UI ---
st.title("Handwritten Digit Recognition")
st.markdown("Draw a digit (0-9) in the center of the canvas below and click **Predict**.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Drawing Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=20,
        stroke_color="#000000",
        background_color="#ffffff",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        predict_btn = st.button("🔍 Predict")
    with btn_col2:
        if st.button("🗑️ Clear"):
            st.rerun()

with col2:
        st.subheader("Analysis & Results")
        
        # --- NEW: Initialize a memory state for the prediction ---
        if "current_prediction" not in st.session_state:
            st.session_state.current_prediction = None

        # 1. ONLY run the ML model when the Predict button is clicked
        if predict_btn and canvas_result.image_data is not None:
            with st.spinner("Processing image..."):
                tensor = preprocess_canvas_image(canvas_result.image_data)
                digit, conf, top_3, probs = predict_digit(model, tensor)
                
                # Save the results into memory so they survive page refreshes!
                st.session_state.current_prediction = {
                    "digit": digit,
                    "conf": conf,
                    "probs": probs,
                    "image_data": canvas_result.image_data
                }
                st.session_state.history.append({"digit": digit, "conf": conf})

        # 2. Display the UI ONLY if we have a prediction saved in memory
        if st.session_state.current_prediction is not None:
            # Load the data out of memory
            mem = st.session_state.current_prediction
            
            # --- Prediction Card ---
            st.markdown(f"""
                <div class="prediction-card">
                    <p style='margin:0; font-size:14px; color:#aaa;'>PRIMARY PREDICTION</p>
                    <div class="digit-display">{mem['digit']}</div>
                    <p style='text-align:center; margin:0; font-size:18px;'>Confidence: <b>{mem['conf']:.2%}</b></p>
                </div>
                """, unsafe_allow_html=True)

            # --- Probabilities Chart ---
            st.write("Confidence Distribution")
            chart_data = pd.DataFrame({
                "Digit": [str(i) for i in range(10)],
                "Probability": mem['probs']
            })
            fig = px.bar(chart_data, x="Digit", y="Probability", color="Probability",
                         color_continuous_scale="Blues", height=300, category_orders={"Digit": [str(i) for i in range(10)]})
            fig.update_xaxes(type='category', tickmode='linear')
            fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # --- Feedback Form ---
            st.write("---")
            st.subheader("💡 Help the AI Learn")
            st.caption("Is this prediction incorrect? Tell us what you actually drew to improve the model.")

            with st.form("feedback_form", clear_on_submit=True):
                correct_label = st.selectbox("What is the actual digit?", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                submitted = st.form_submit_button("Submit Correction")
        
                if submitted:
                    # Pass the image data from our session state directly to the save function
                    save_feedback(mem['image_data'], correct_label)
                    st.success(f"✅ Awesome! The drawing was saved to the '{correct_label}' folder for the next training batch.")
        
        else:
            # This cleanly handles the empty state on first load
            st.info("Awaiting input. Please draw on the canvas and click Predict.")

st.divider()
st.caption("Engineered by DigitVision AI Team | Production-Grade Portfolio Project")