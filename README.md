# DigitVision 👁️

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

DigitVision is a production-quality AI application designed to recognize handwritten digits in real-time. Built with a deep Convolutional Neural Network (CNN) and a polished Streamlit interface, it bridges the gap between academic MNIST models and real-world user interaction.

## 🚀 Key Features
- **Real-Time Inference:** Draw on a canvas and get immediate predictions.
- **Robust Preprocessing:** OpenCV-powered pipeline that centers, pads, and normalizes user drawings to match MNIST standards.
- **Deep Analytics:** Displays top-3 predictions and probability distributions via Plotly.
- **Premium UI:** Dark-themed, responsive interface with custom CSS.

## 🏗️ Architecture
The project follows a modular, scalable architecture with a strict separation of concerns:
- `src/`: Core logic (Preprocessing, Model Architecture, Inference).
- `app/`: Streamlit UI components and state management.
- `models/`: Persistent storage for trained artifacts.
- `tests/`: Automated unit tests for data integrity.

## 🛠️ Tech Stack
- **Engine:** Python, TensorFlow, Keras, NumPy.
- **Vision:** OpenCV, Pillow.
- **UI:** Streamlit, Streamlit-Drawable-Canvas.
- **Analysis:** Plotly, Pandas.

## 📦 Installation & Usage
1. **Clone the repo:** `git clone https://github.com/username/digitvision.git`
2. **Install deps:** `pip install -r requirements.txt`
3. **Train the model:** `python src/train.py`
4. **Launch the App:** `streamlit run app/app.py`

## 🧪 Testing
Run the test suite to verify the preprocessing and prediction pipelines:
```bash
pytest tests/

