import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import collections
import io

# --- 0. CONSTANTS ---
APP_VERSION = "1.0"
DEFAULT_CONFIDENCE = 0.4
CONFIDENCE_MIN = 0.1
CONFIDENCE_MAX = 1.0
CONFIDENCE_STEP = 0.05

# --- 1. PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(
    page_title="AI Blind Assistance",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, rgba(99,102,241,0.8), rgba(168,85,247,0.8));
        border-radius: 20px;
        padding: 48px 36px 40px;
        text-align: center;
        margin-bottom: 32px;
        box-shadow: 0 20px 60px rgba(99,102,241,0.35);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute; inset: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 12px;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: rgba(255,255,255,0.85);
        font-weight: 400;
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 999px;
        padding: 4px 16px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #fff;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 18px;
    }

    /* ── Section headings ── */
    .section-heading {
        font-size: 1rem;
        font-weight: 700;
        color: #a5b4fc;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 24px 0 12px;
    }

    /* ── Glass card ── */
    .glass-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(16px);
        border-radius: 18px;
        padding: 28px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }

    /* ── Detection result pill ── */
    .result-sentence {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f1f5f9;
        background: rgba(99,102,241,0.25);
        border: 1px solid rgba(99,102,241,0.5);
        border-radius: 12px;
        padding: 16px 20px;
        line-height: 1.5;
    }

    /* ── Object tag chips ── */
    .chip-container { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
    .chip {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #fff;
        border-radius: 999px;
        padding: 6px 18px;
        font-size: 0.88rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(99,102,241,0.35);
    }

    /* ── Team card grid ── */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-top: 12px;
    }
    .team-member {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 20px 14px;
        text-align: center;
        transition: transform 0.2s;
    }
    .team-member:hover { transform: translateY(-4px); }
    .team-avatar {
        width: 52px; height: 52px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem;
        margin: 0 auto 10px;
        box-shadow: 0 4px 14px rgba(99,102,241,0.4);
    }
    .team-name {
        font-size: 0.9rem;
        font-weight: 600;
        color: #e2e8f0;
        line-height: 1.3;
    }
    .team-role {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* ── Enhancement list ── */
    .enhance-item {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 12px 0;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .enhance-item:last-child { border-bottom: none; }
    .enhance-icon {
        font-size: 1.3rem;
        flex-shrink: 0;
        margin-top: 1px;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.8rem;
        margin-top: 48px;
        padding-bottom: 24px;
    }

    /* ── Streamlit overrides ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #fff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 24px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    div[data-testid="stCameraInput"] label { color: #a5b4fc !important; font-weight: 600; }

    /* Override default Streamlit spinner / info text colours */
    .stSpinner > div { color: #a5b4fc !important; }
    </style>
""", unsafe_allow_html=True)


# --- 2. LOAD YOLO MODEL ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    confidence_threshold = st.slider(
        "Detection Confidence",
        min_value=CONFIDENCE_MIN,
        max_value=CONFIDENCE_MAX,
        value=DEFAULT_CONFIDENCE,
        step=CONFIDENCE_STEP,
        help="Minimum confidence score for a detection to be reported."
    )
    audio_speed = st.toggle("Slow speech", value=False)
    show_annotated = st.toggle("Show annotated image", value=True)

    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown(
        "1. Point your camera at the scene.\n"
        "2. Press **Capture** to snap a photo.\n"
        "3. The AI detects objects and reads them aloud.\n"
        "4. Review the annotated image below."
    )

    st.markdown("---")
    st.markdown("### 👥 Team")
    team = [
        ("Muhammad Ahmad Shoaib", "🧑‍💻"),
        ("Osama Saeed", "🧑‍💻"),
        ("Ahmed Shafique Lone", "🧑‍💻"),
        ("Afzaal Ur Rahman", "🧑‍💻"),
    ]
    for name, icon in team:
        st.markdown(f"{icon} **{name}**")

    st.markdown("---")
    st.caption(f"AI Blind Assistance · v{426.001}")


# ══════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>AI · Accessibility · Vision</div>
    <div class='hero-title'>👁️ OBJECT DETECTION WITH VOICE FEEDBACK</div>
    <div class='hero-subtitle'>
        Empowering the visually impaired with real-time object detection and
        natural voice feedback — powered by YOLOv12.
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CAMERA INPUT
# ══════════════════════════════════════════════════════════
st.markdown("<div class='section-heading'>📷 Capture Scene</div>", unsafe_allow_html=True)
camera_image = st.camera_input("Take a photo", label_visibility="collapsed")


# ══════════════════════════════════════════════════════════
# PROCESSING
# ══════════════════════════════════════════════════════════
if camera_image is not None:
    with st.spinner("🔍 Analyzing environment with AI…"):
        img = Image.open(camera_image)
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame_bgr, verbose=False, conf=confidence_threshold)

        detected_names = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_names.append(class_name)

    # ── Results ──
    col_result, col_img = st.columns([1, 1], gap="large")

    with col_result:
        st.markdown("<div class='section-heading'>🔊 Detection Results</div>", unsafe_allow_html=True)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        if detected_names:
            item_counts = dict(collections.Counter(detected_names))
            speech_parts = [
                f"{count} {item}{'s' if count > 1 else ''}"
                for item, count in item_counts.items()
            ]
            spoken_sentence = "I detect " + ", ".join(speech_parts)

            st.markdown(
                f"<div class='result-sentence'>🔊 {spoken_sentence}</div>",
                unsafe_allow_html=True
            )

            # Chip tags for each unique object
            chips_html = "<div class='chip-container'>" + "".join(
                f"<span class='chip'>{item} × {count}</span>"
                for item, count in item_counts.items()
            ) + "</div>"
            st.markdown(chips_html, unsafe_allow_html=True)

        else:
            spoken_sentence = "No objects detected in the current view."
            st.markdown(
                f"<div class='result-sentence'>🔊 {spoken_sentence}</div>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Audio
        tts = gTTS(text=spoken_sentence, lang='en', slow=audio_speed)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        st.audio(fp, format="audio/mp3", autoplay=True)

    with col_img:
        if show_annotated:
            st.markdown("<div class='section-heading'>🖼️ Annotated View</div>", unsafe_allow_html=True)
            annotated_frame_bgr = results[0].plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_frame_rgb, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TEAM SECTION
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<div class='section-heading'>👥 Meet our Team</div>", unsafe_allow_html=True)

team_members = [
    {"name": "Muhammad Ahmad Shoaib", "role": "Group Leader & ML Development", "emoji": "🧑‍💻"},
    {"name": "Osama Saeed",           "role": "UI/UX Design & Documentation",    "emoji": "🎨"},
    {"name": "Ahmed Shafique Lone",   "role": "ML Development",  "emoji": "🧑‍💻"},
    {"name": "Afzaal Ur Rahman",      "role": "ML Development",   "emoji": "🧑‍💻"},
]

cols = st.columns(len(team_members), gap="medium")
for col, member in zip(cols, team_members):
    with col:
        st.markdown(f"""
        <div class='team-member'>
            <div class='team-avatar'>{member['emoji']}</div>
            <div class='team-name'>{member['name']}</div>
            <div class='team-role'>{member['role']}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# FUTURE GOALS
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<div class='section-heading'>FUTURE GOALS</div>", unsafe_allow_html=True)

enhancements = [
    ("🌍", "Multi-language TTS — Let users choose their preferred spoken language (Urdu, Arabic, French, etc.) so the tool is accessible worldwide."),
    ("📍", "Spatial audio & distance estimation — Use depth cues or stereo audio panning to indicate how far away and in which direction objects are."),
    ("🎙️", "Voice commands — Allow users to ask questions like \"What's in front of me?\" using a speech-to-text loop for a hands-free experience."),
    ("📱", "Mobile-first PWA — Package the app as a Progressive Web App so it installs on phones and runs offline with a native-app feel."),
    ("⚡", "Edge inference — Deploy a quantised YOLO model directly on-device (iOS/Android via CoreML / TFLite) for zero-latency, no-internet detection."),
    ("🗺️", "Obstacle navigation — Integrate GPS and an indoor mapping layer to guide users through routes and warn about upcoming hazards."),
    ("📖", "Scene captioning — Add a vision-language model (e.g. BLIP-2, LLaVA) to generate natural-language scene descriptions beyond simple object lists."),
    ("🔔", "Hazard alerts — Prioritise and call out safety-critical objects (cars, fire, steps) with a distinct urgent tone before listing other items."),
    ("🧏", "Sign-language output — Render detected text or descriptions as sign-language animations for deaf-blind users."),
    ("📊", "Usage dashboard — Log detection history and display analytics (most common objects, session counts) to researchers and caregivers."),
]

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
for icon, text in enhancements:
    st.markdown(f"""
    <div class='enhance-item'>
        <span class='enhance-icon'>{icon}</span>
        <span>{text}</span>
    </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class='footer'>
    Built with ❤️ by the AI Blind Assistance Team &nbsp;·&nbsp;
    Powered by YOLOv12 &amp; Streamlit &nbsp;·&nbsp; 2026
    Samsung Innovation Campus (AI Training Final Capstone) &nbsp;·&nbsp;
</div>
""", unsafe_allow_html=True)
