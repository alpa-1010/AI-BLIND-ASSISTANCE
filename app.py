import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import collections
import io

# --- 1. PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="AI Blind Assistance", page_icon="👁️", layout="centered")

st.markdown("""
    <style>
    /* Main background and typography */
    .stApp {
        background-color: #f4f6f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    h1 {
        color: #1E3A8A;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Cards for visual outputs */
    .output-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-top: 20px;
    }
    
    /* Status text styling */
    .status-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0F172A;
        padding: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD YOLO MODEL ---
# @st.cache_resource ensures the model only loads once, keeping the app fast
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# --- 3. UI LAYOUT ---
st.title("👁️ AI Environment Scanner")
st.markdown("<div class='subtitle'>Point your camera and capture to hear what is around you.</div>", unsafe_allow_html=True)

# Streamlit's native camera input (replaces the 30 lines of JS from Colab)
camera_image = st.camera_input("Capture Environment", label_visibility="hidden")

# --- 4. PROCESSING THE IMAGE ---
if camera_image is not None:
    with st.spinner("Analyzing environment..."):
        # Convert the uploaded file to a PIL Image, then to an OpenCV array
        img = Image.open(camera_image)
        frame = np.array(img)
        
        # YOLO expects BGR format (OpenCV standard) for accurate color representation, 
        # though for basic object detection, RGB often works fine. We'll convert to BGR.
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run YOLOv8 Inference
        results = model(frame_bgr, verbose=False)

        # Extract detected objects
        detected_names = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_names.append(class_name)

        # --- 5. AUDIO & VISUAL OUTPUT ---
        st.markdown("<div class='output-card'>", unsafe_allow_html=True)
        
        if detected_names:
            # Count the items
            item_counts = dict(collections.Counter(detected_names))

            # Format the spoken sentence
            speech_parts = [f"{count} {item}{'s' if count > 1 else ''}" for item, count in item_counts.items()]
            spoken_sentence = "I detect " + ", ".join(speech_parts)
            
            # Display text
            st.markdown(f"<div class='status-text'>🔊 {spoken_sentence}</div>", unsafe_allow_html=True)

            # Generate audio in memory (avoids saving local mp3 files)
            tts = gTTS(text=spoken_sentence, lang='en', slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # Play audio automatically
            st.audio(fp, format="audio/mp3", autoplay=True)
            
        else:
            spoken_sentence = "No objects detected in the current view."
            st.markdown(f"<div class='status-text'>🔊 {spoken_sentence}</div>", unsafe_allow_html=True)
            
            tts = gTTS(text=spoken_sentence, lang='en', slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            st.audio(fp, format="audio/mp3", autoplay=True)

        # Draw bounding boxes and display the image
        annotated_frame_bgr = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.image(annotated_frame_rgb, caption="Analyzed Image", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
