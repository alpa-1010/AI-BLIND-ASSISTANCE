# 👁️ AI Blind Assistance

> **Real-time object detection with voice feedback — built to empower the visually impaired.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Samsung Innovation Campus](https://img.shields.io/badge/Samsung-Innovation%20Campus-1428A0)](https://www.samsung.com/global/sustainability/social/education/)

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Demo](#-demo)
4. [Architecture](#-architecture)
5. [Tech Stack](#-tech-stack)
6. [Getting Started](#-getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running the App](#running-the-app)
7. [Usage Guide](#-usage-guide)
8. [Project Structure](#-project-structure)
9. [Configuration](#-configuration)
10. [Future Goals](#-future-goals)
11. [Team](#-team)
12. [Acknowledgements](#-acknowledgements)

---

## 🌟 Project Overview

**AI Blind Assistance** is an accessibility-first web application that uses a live camera feed to detect objects in a scene and immediately reads them aloud to the user. It was built as the **Final Capstone Project** for the *Samsung Innovation Campus — AI Training* programme.

The tool bridges the gap between computer-vision research and day-to-day accessibility needs. With a single tap on a mobile browser or desktop webcam, a visually impaired person receives an instant, clear audio description of their surroundings — no specialised hardware required.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📷 **Live Camera Capture** | One-click snapshot via any device camera using Streamlit's built-in camera widget |
| 🤖 **YOLOv8 Object Detection** | Fast, accurate real-time detection across 80 common object categories |
| 🔊 **Voice Feedback (TTS)** | Detected objects are automatically converted to natural speech via Google TTS and played back in the browser |
| 🖼️ **Annotated Image View** | Bounding-box overlay rendered on the captured frame for sighted caregivers or developers |
| ⚙️ **Adjustable Confidence** | Slider to tune the detection confidence threshold (0.1 – 1.0) |
| 🐢 **Slow Speech Toggle** | Optional slower TTS output for users who need extra time to process audio |
| 🎨 **Modern UI** | Glassmorphism design with gradient backgrounds, chip tags, and a responsive layout |

---

## 🎥 Demo

> Launch the app, point your camera at any scene, and press **Capture**.  
> The app will display a sentence such as:  
> **"I detect 1 person, 2 chairs, 1 laptop"**  
> and play it back immediately as audio.

---

## 🏗️ Architecture

```
User Camera
     │
     ▼
Streamlit Camera Widget
     │  (JPEG bytes)
     ▼
PIL → NumPy → OpenCV (RGB→BGR)
     │
     ▼
YOLOv8n Inference  ──→  Annotated Frame (BGR→RGB) → st.image()
     │
     ▼
Object Name List  ──→  Counter  ──→  Natural Language Sentence
     │
     ▼
gTTS (Google Text-to-Speech)  ──→  In-memory MP3  ──→  st.audio (autoplay)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | [Streamlit](https://streamlit.io/) |
| **Object Detection** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8n.pt`) |
| **Text-to-Speech** | [gTTS (Google Text-to-Speech)](https://gtts.readthedocs.io/) |
| **Image Processing** | [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [Pillow](https://python-pillow.org/) |
| **Language** | Python 3.9+ |
| **System Dependency** | `libglib2.0-0`, `libgl1` (required by OpenCV on Linux; declared in `packages.txt` for Streamlit Community Cloud) |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9** or higher
- A working **webcam** (built-in or USB)
- Internet connection (for the first run to download `yolov8n.pt` weights and for TTS API calls)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/alpa-1010/AI-BLIND-ASSISTANCE.git
cd AI-BLIND-ASSISTANCE

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install system-level dependencies (Linux only)
sudo apt-get update && sudo apt-get install -y libglib2.0-0 libgl1

# 4. Install Python dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

Streamlit will print a local URL (default: `http://localhost:8501`). Open it in your browser, grant camera permission when prompted, and you're ready to go.

> **Note:** On the first run, Ultralytics will automatically download the `yolov8n.pt` model weights (~6 MB).

> **Streamlit Community Cloud deployment:** System-level OS packages are declared in `packages.txt` at the repo root. This file instructs Streamlit Cloud to install `libglib2.0-0` (GLib, which provides `libgthread-2.0.so.0`) and `libgl1` before starting the app. Without these libraries, importing `cv2` fails with an `ImportError` on the hosted Linux environment.

---

## 📋 Usage Guide

1. **Open** the app in your browser.
2. **Adjust Settings** in the left sidebar:
   - *Detection Confidence* — raise it to filter out uncertain detections; lower it to catch more objects.
   - *Slow speech* — toggle on if you prefer a slower audio pace.
   - *Show annotated image* — toggle off to skip the bounding-box overlay.
3. **Click "Take a photo"** (the camera widget in the main area).
4. Wait briefly while the AI analyses the scene.
5. **Listen** to the spoken description and read the detection chips.
6. Review the **annotated image** (if enabled) to see exactly which objects were found.

---

## 📁 Project Structure

```
AI-BLIND-ASSISTANCE/
├── app.py              # Main Streamlit application (UI, model inference, TTS)
├── requirements.txt    # Python package dependencies
├── packages.txt        # System-level apt packages (libglib2.0-0, libgl1)
└── README.md           # Project documentation (this file)
```

> The `yolov8n.pt` model weights file is downloaded automatically by Ultralytics on the first run and cached locally.

---

## ⚙️ Configuration

All tunable constants live at the top of `app.py`:

| Constant | Default | Description |
|---|---|---|
| `APP_VERSION` | `"1.0"` | Displayed version string |
| `DEFAULT_CONFIDENCE` | `0.4` | Initial confidence threshold |
| `CONFIDENCE_MIN` | `0.1` | Minimum slider value |
| `CONFIDENCE_MAX` | `1.0` | Maximum slider value |
| `CONFIDENCE_STEP` | `0.05` | Slider step increment |

To switch to a larger/more accurate YOLO model, change the model path in `load_model()`:

```python
def load_model():
    return YOLO('yolov8s.pt')   # swap 'n' → 's', 'm', 'l', or 'x'
```

---

## 🔭 Future Goals

The team has identified the following enhancements for future iterations:

| # | Goal | Description |
|---|---|---|
| 1 | 🌍 **Multi-language TTS** | Let users choose their preferred spoken language (Urdu, Arabic, French, etc.) |
| 2 | 📍 **Spatial Audio & Distance Estimation** | Indicate direction and proximity of objects via depth cues or stereo audio panning |
| 3 | 🎙️ **Voice Commands** | Hands-free Q&A loop — "What's in front of me?" — via speech-to-text |
| 4 | 📱 **Mobile-first PWA** | Progressive Web App that installs on phones and runs offline |
| 5 | ⚡ **Edge Inference** | Quantised YOLO on-device (CoreML / TFLite) for zero-latency, offline detection |
| 6 | 🗺️ **Obstacle Navigation** | GPS + indoor mapping layer to guide users and warn about hazards |
| 7 | 📖 **Scene Captioning** | Vision-language model (BLIP-2, LLaVA) for rich natural-language scene descriptions |
| 8 | 🔔 **Hazard Alerts** | Priority urgent tone for safety-critical objects (cars, fire, steps) |
| 9 | 🧏 **Sign-language Output** | Sign-language animations for deaf-blind users |
| 10 | 📊 **Usage Dashboard** | Detection history and analytics for researchers and caregivers |

---

## 👥 Team

This project was developed by a four-person team as part of the **Samsung Innovation Campus — AI Training Final Capstone (2026)**:

| Name | Role |
|---|---|
| **Muhammad Ahmad Shoaib** | Group Leader & ML Development |
| **Osama Saeed** | UI/UX Design & Documentation |
| **Ahmed Shafique Lone** | ML Development |
| **Afzaal Ur Rahman** | ML Development |

---

## 🙏 Acknowledgements

- [Ultralytics](https://ultralytics.com/) for the YOLOv8 model and easy-to-use Python API
- [Streamlit](https://streamlit.io/) for making it trivial to turn a Python script into a shareable web app
- [Google Text-to-Speech (gTTS)](https://gtts.readthedocs.io/) for the voice feedback engine
- **Samsung Innovation Campus** for sponsoring and supporting this project

---

<div align="center">
  Built with ❤️ by the AI Blind Assistance Team &nbsp;·&nbsp; Powered by YOLOv8 &amp; Streamlit &nbsp;·&nbsp; 2026
</div>
