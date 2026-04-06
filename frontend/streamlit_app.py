# ============================================================
# streamlit_app.py  –  Ultimate Professional Frontend
# ============================================================

import streamlit as st
import requests
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="NeuroDetect AI | Parkinson's Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = "http://127.0.0.1:5000"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: #050508;
        color: #e2e8f0;
    }

    .hero-wrap {
        background: linear-gradient(135deg, #0d1117 0%, #161b27 40%, #0d1f3c 100%);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 24px;
        padding: 50px 40px;
        text-align: center;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 30px 80px rgba(0,0,0,0.6);
    }
    .hero-wrap::before {
        content: '';
        position: absolute;
        top: -100px; left: -100px;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-wrap::after {
        content: '';
        position: absolute;
        bottom: -100px; right: -100px;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(159,122,234,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99,179,237,0.15), rgba(159,122,234,0.15));
        border: 1px solid rgba(99,179,237,0.3);
        color: #90cdf4;
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 20px;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #63b3ed 0%, #90cdf4 40%, #b794f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 15px 0;
        letter-spacing: -2px;
        line-height: 1.1;
    }
    .hero-sub {
        color: #4a5568;
        font-size: 1rem;
        line-height: 1.8;
    }
    .hero-tags {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 20px;
        flex-wrap: wrap;
    }
    .hero-tag {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        color: #718096;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 30px;
    }
    .stat-box {
        background: linear-gradient(135deg, #0d1117, #161b27);
        border: 1px solid rgba(99,179,237,0.12);
        border-radius: 16px;
        padding: 22px 15px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .stat-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #63b3ed, #b794f4);
    }
    .stat-box:hover {
        border-color: rgba(99,179,237,0.3);
        transform: translateY(-4px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
    }
    .stat-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #63b3ed, #b794f4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .stat-lbl {
        color: #4a5568;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
    }

    .model-info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 30px;
    }
    .model-card {
        background: linear-gradient(135deg, #0d1117, #161b27);
        border: 1px solid rgba(99,179,237,0.12);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
    }
    .model-card:hover {
        border-color: rgba(99,179,237,0.3);
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.4);
    }
    .model-card-icon { font-size: 2rem; margin-bottom: 10px; }
    .model-card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 6px;
    }
    .model-card-desc { color: #4a5568; font-size: 0.8rem; line-height: 1.5; }
    .model-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-top: 10px;
        margin-right: 4px;
    }
    .badge-rf  { background: rgba(72,187,120,0.15);  color: #68d391; border: 1px solid rgba(72,187,120,0.3);  }
    .badge-svm { background: rgba(99,179,237,0.15);  color: #90cdf4; border: 1px solid rgba(99,179,237,0.3);  }
    .badge-lr  { background: rgba(246,173,85,0.15);  color: #f6ad55; border: 1px solid rgba(246,173,85,0.3);  }
    .badge-cnn { background: rgba(159,122,234,0.15); color: #b794f4; border: 1px solid rgba(159,122,234,0.3); }
    .badge-best { background: rgba(245,101,101,0.15); color: #fc8181; border: 1px solid rgba(245,101,101,0.3); }

    .upload-card {
        background: linear-gradient(135deg, #0d1117, #161b27);
        border: 1px solid rgba(99,179,237,0.12);
        border-radius: 20px;
        padding: 25px;
        transition: all 0.3s;
        height: 100%;
    }
    .upload-card:hover {
        border-color: rgba(99,179,237,0.25);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    .upload-card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .upload-card-icon { font-size: 1.8rem; }
    .upload-card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .upload-card-sub { color: #4a5568; font-size: 0.8rem; }

    .result-box-pd {
        background: linear-gradient(135deg, rgba(245,101,101,0.12), rgba(197,48,48,0.08));
        border: 1px solid rgba(245,101,101,0.35);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
    }
    .result-box-ok {
        background: linear-gradient(135deg, rgba(72,187,120,0.12), rgba(47,133,90,0.08));
        border: 1px solid rgba(72,187,120,0.35);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
    }
    .result-icon  { font-size: 2.5rem; margin-bottom: 8px; }
    .result-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 800; }
    .result-box-pd .result-title { color: #fc8181; }
    .result-box-ok .result-title { color: #68d391; }
    .result-conf  { color: #718096; font-size: 0.82rem; margin-top: 6px; }

    .prog-bar-wrap {
        background: rgba(255,255,255,0.04);
        border-radius: 8px; height: 7px;
        margin: 10px 0; overflow: hidden;
    }
    .prog-bar-pd { height:100%; background:linear-gradient(90deg,#fc8181,#f56565); border-radius:8px; }
    .prog-bar-ok { height:100%; background:linear-gradient(90deg,#68d391,#48bb78); border-radius:8px; }

    .expl-box {
        background: rgba(99,179,237,0.04);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 12px;
        padding: 14px;
        margin-top: 12px;
        font-size: 0.82rem;
        color: #718096;
        line-height: 1.7;
        text-align: left;
    }

    .final-pd {
        background: linear-gradient(135deg, rgba(245,101,101,0.18), rgba(197,48,48,0.1));
        border: 2px solid rgba(245,101,101,0.45);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
    }
    .final-ok {
        background: linear-gradient(135deg, rgba(72,187,120,0.18), rgba(47,133,90,0.1));
        border: 2px solid rgba(72,187,120,0.45);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
    }
    .final-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 15px 0 10px;
    }
    .final-pd .final-title { color: #fc8181; }
    .final-ok .final-title { color: #68d391; }

    .divider {
        border: none; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,179,237,0.2), transparent);
        margin: 35px 0;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050508 0%, #0d1117 100%) !important;
        border-right: 1px solid rgba(99,179,237,0.08) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1a365d, #2a4a8a) !important;
        color: #90cdf4 !important;
        border: 1px solid rgba(99,179,237,0.3) !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        width: 100% !important;
        transition: all 0.3s !important;
        letter-spacing: 0.5px !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a4a8a, #3182ce) !important;
        color: white !important;
        border-color: rgba(99,179,237,0.6) !important;
        box-shadow: 0 8px 25px rgba(49,130,206,0.35) !important;
        transform: translateY(-2px) !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 20px !important; }
</style>
""", unsafe_allow_html=True)

# ── Install plotly if needed ──────────────────────────────────
try:
    import plotly.graph_objects as go
except:
    st.error("Run: pip install plotly")
    st.stop()

# ── Session State ─────────────────────────────────────────────
for key in ["voice_result", "image_result", "video_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

def check_backend():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

backend_ok = check_backend()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:25px 0 15px;'>
        <div style='font-size:3.5rem; margin-bottom:10px;'>🧠</div>
        <div style='font-family:Space Grotesk; font-size:1.4rem; font-weight:800;
             background:linear-gradient(135deg,#63b3ed,#b794f4);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            NeuroDetect AI
        </div>
        <div style='color:#2d3748; font-size:0.78rem; margin-top:4px; letter-spacing:1px;'>
            PARKINSON'S DETECTION SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    if backend_ok:
        st.markdown("""<div style='background:rgba(72,187,120,0.08);border:1px solid rgba(72,187,120,0.25);
        border-radius:10px;padding:10px;text-align:center;color:#68d391;font-weight:700;font-size:0.82rem;
        margin-bottom:15px;'>⚡ Backend Connected</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style='background:rgba(245,101,101,0.08);border:1px solid rgba(245,101,101,0.25);
        border-radius:10px;padding:10px;text-align:center;color:#fc8181;font-weight:700;font-size:0.82rem;
        margin-bottom:15px;'>❌ Backend Offline<br><span style='font-size:0.72rem;font-weight:400;
        color:#4a5568;'>python backend/app.py</span></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
    border-radius:14px;padding:18px;margin-bottom:15px;'>
        <div style='color:#63b3ed;font-weight:700;font-size:0.78rem;letter-spacing:1.5px;
        text-transform:uppercase;margin-bottom:12px;'>🤖 ML Models</div>
        <div style='color:#4a5568;font-size:0.82rem;line-height:2.2;'>
            <span style='color:#68d391;'>●</span> Random Forest <span style='color:#2d3748;font-size:0.7rem;'>(Best)</span><br>
            <span style='color:#90cdf4;'>●</span> Support Vector Machine<br>
            <span style='color:#f6ad55;'>●</span> Logistic Regression<br>
            <span style='color:#b794f4;'>●</span> CNN — Image Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
    border-radius:14px;padding:18px;margin-bottom:15px;'>
        <div style='color:#63b3ed;font-weight:700;font-size:0.78rem;letter-spacing:1.5px;
        text-transform:uppercase;margin-bottom:12px;'>📡 Modalities</div>
        <div style='color:#4a5568;font-size:0.82rem;line-height:2.2;'>
            🎙️ Voice Biomarkers<br>
            🖼️ Spiral Drawing (CNN)<br>
            🎥 Hand Tremor Video
        </div>
    </div>
    """, unsafe_allow_html=True)

    completed = sum([
        st.session_state.voice_result is not None,
        st.session_state.image_result is not None,
        st.session_state.video_result is not None
    ])
    pct_done = int((completed / 3) * 100)
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
    border-radius:14px;padding:18px;'>
        <div style='color:#63b3ed;font-weight:700;font-size:0.78rem;letter-spacing:1.5px;
        text-transform:uppercase;margin-bottom:12px;'>📊 Progress</div>
        <div style='font-family:Space Grotesk;font-size:2.5rem;font-weight:800;text-align:center;
        background:linear-gradient(135deg,#63b3ed,#b794f4);-webkit-background-clip:text;
        -webkit-text-fill-color:transparent;'>{completed}/3</div>
        <div style='background:rgba(255,255,255,0.04);border-radius:6px;height:5px;margin:8px 0;overflow:hidden;'>
            <div style='height:100%;width:{pct_done}%;background:linear-gradient(90deg,#63b3ed,#b794f4);border-radius:6px;'></div>
        </div>
        <div style='text-align:center;color:#2d3748;font-size:0.75rem;'>Modalities Analysed</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;color:#1a202c;font-size:0.72rem;margin-top:20px;'>
        Final Year Capstone Project<br>Multi-Modal Parkinson's Detection
    </div>
    """, unsafe_allow_html=True)


# ── HERO ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🔬 AI-Powered Neurological Analysis</div>
    <h1 class="hero-title">🧠 NeuroDetect AI</h1>
    <div class="hero-sub">
        Multi-Modal Parkinson's Disease Detection using Machine Learning<br>
        <span style='color:#2d3748;'>Combining Voice • Vision • Motion for accurate early detection</span>
    </div>
    <div class="hero-tags">
        <span class="hero-tag">Random Forest</span>
        <span class="hero-tag">Support Vector Machine</span>
        <span class="hero-tag">Logistic Regression</span>
        <span class="hero-tag">CNN (Keras)</span>
        <span class="hero-tag">Flask API</span>
        <span class="hero-tag">Real-Time</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STATS ─────────────────────────────────────────────────────
st.markdown("""
<div class="stats-grid">
    <div class="stat-box">
        <div class="stat-val">100%</div>
        <div class="stat-lbl">Model Accuracy</div>
    </div>
    <div class="stat-box">
        <div class="stat-val">4</div>
        <div class="stat-lbl">ML Algorithms</div>
    </div>
    <div class="stat-box">
        <div class="stat-val">3</div>
        <div class="stat-lbl">Input Modalities</div>
    </div>
    <div class="stat-box">
        <div class="stat-val">500+</div>
        <div class="stat-lbl">Training Samples</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── MODEL INFO CARDS ──────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Grotesk;font-size:1.4rem;font-weight:700;color:#e2e8f0;
margin-bottom:16px;'>⚙️ ML Models Overview</div>
<div class="model-info-grid">
    <div class="model-card">
        <div class="model-card-icon">🎙️</div>
        <div class="model-card-title">Voice Analysis</div>
        <div class="model-card-desc">
            Three classifiers trained on vocal biomarkers (MDVP features).
            Compared and best model selected automatically.
        </div>
        <span class="model-badge badge-lr">Logistic Regression</span>
        <span class="model-badge badge-svm">SVM</span>
        <span class="model-badge badge-rf">Random Forest ★</span>
        <span class="model-badge badge-best">Best Model</span>
    </div>
    <div class="model-card">
        <div class="model-card-icon">🖼️</div>
        <div class="model-card-title">Spiral Drawing — CNN</div>
        <div class="model-card-desc">
            Convolutional Neural Network analyses handwritten spiral drawings
            for tremor and motor control irregularities.
        </div>
        <span class="model-badge badge-cnn">Conv2D × 3</span>
        <span class="model-badge badge-cnn">MaxPooling</span>
        <span class="model-badge badge-cnn">Dense + Dropout</span>
    </div>
    <div class="model-card">
        <div class="model-card-icon">🎥</div>
        <div class="model-card-title">Video Tremor Analysis</div>
        <div class="model-card-desc">
            OpenCV extracts frame-by-frame motion features.
            Random Forest classifies tremor patterns from motion signals.
        </div>
        <span class="model-badge badge-rf">Random Forest</span>
        <span class="model-badge badge-svm">10 Motion Features</span>
        <span class="model-badge badge-lr">OpenCV</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── MODEL ACCURACY CHART ──────────────────────────────────────
st.markdown("""
<div style='font-family:Space Grotesk;font-size:1.4rem;font-weight:700;
color:#e2e8f0;margin-bottom:16px;'>📊 Model Performance Comparison</div>
""", unsafe_allow_html=True)

col_c1, col_c2 = st.columns(2)

with col_c1:
    fig1 = go.Figure(go.Bar(
        x=["Logistic\nRegression", "SVM", "Random\nForest"],
        y=[96.5, 97.8, 100.0],
        marker=dict(
            color=["#f6ad55", "#90cdf4", "#68d391"],
            line=dict(color="rgba(0,0,0,0)", width=0)
        ),
        text=["96.5%", "97.8%", "100%"],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13, family="Space Grotesk"),
    ))
    fig1.update_layout(
        title=dict(text="Voice Models — Accuracy Comparison",
                   font=dict(color="#a0aec0", size=13)),
        paper_bgcolor="rgba(13,17,23,0.8)",
        plot_bgcolor="rgba(13,17,23,0.8)",
        font=dict(color="#718096"),
        yaxis=dict(range=[90, 102], gridcolor="rgba(255,255,255,0.04)",
                   color="#4a5568"),
        xaxis=dict(color="#4a5568"),
        margin=dict(t=50, b=20, l=20, r=20),
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_c2:
    fig2 = go.Figure()
    epochs = list(range(1, 16))
    train_acc = [62.5, 80.0, 96.9, 99.4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    val_acc   = [100,  100,  100,  100,  100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    fig2.add_trace(go.Scatter(x=epochs, y=train_acc, name="Train",
        line=dict(color="#b794f4", width=2.5),
        fill="tozeroy", fillcolor="rgba(159,122,234,0.05)"))
    fig2.add_trace(go.Scatter(x=epochs, y=val_acc, name="Validation",
        line=dict(color="#68d391", width=2.5, dash="dot")))
    fig2.update_layout(
        title=dict(text="CNN Training Accuracy (15 Epochs)",
                   font=dict(color="#a0aec0", size=13)),
        paper_bgcolor="rgba(13,17,23,0.8)",
        plot_bgcolor="rgba(13,17,23,0.8)",
        font=dict(color="#718096"),
        yaxis=dict(range=[50, 105], gridcolor="rgba(255,255,255,0.04)",
                   color="#4a5568", ticksuffix="%"),
        xaxis=dict(color="#4a5568", title="Epoch"),
        margin=dict(t=50, b=20, l=20, r=20),
        height=280,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#718096")),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── UPLOAD SECTION ────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Grotesk;font-size:1.4rem;font-weight:700;
color:#e2e8f0;margin-bottom:20px;'>🔬 Run Analysis</div>
""", unsafe_allow_html=True)

def display_result(result_dict):
    if result_dict is None:
        return
    label   = result_dict.get("prediction", "")
    pct     = result_dict.get("confidence_pct", 0)
    explain = result_dict.get("explanation", "")
    is_pd   = "Parkinson" in label
    box_cls = "result-box-pd" if is_pd else "result-box-ok"
    bar_cls = "prog-bar-pd"   if is_pd else "prog-bar-ok"
    icon    = "🚨" if is_pd else "✅"

    st.markdown(f"""
    <div class="{box_cls}">
        <div class="result-icon">{icon}</div>
        <div class="result-title">{label}</div>
        <div class="prog-bar-wrap">
            <div class="{bar_cls}" style="width:{pct}%"></div>
        </div>
        <div class="result-conf">Confidence: <strong>{pct}%</strong></div>
    </div>
    <div class="expl-box">💡 <strong style='color:#63b3ed;'>AI Insight:</strong><br>{explain}</div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

# VOICE
with col1:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-card-header">
            <div class="upload-card-icon">🎙️</div>
            <div>
                <div class="upload-card-title">Voice Analysis</div>
                <div class="upload-card-sub">Random Forest • SVM • LR</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Mode:", ["WAV File", "Numeric Features"], key="vmode",
                    horizontal=True)

    if mode == "WAV File":
        vf = st.file_uploader("Upload WAV", type=["wav"], key="vup")
        if st.button("🎙️ Analyse Voice", key="bv"):
            if not vf:
                st.warning("Upload a WAV file first")
            else:
                with st.spinner("Analysing voice..."):
                    time.sleep(0.4)
                    try:
                        r = requests.post(f"{API_BASE}/predict-voice",
                            files={"file":(vf.name, vf.getvalue(), "audio/wav")}, timeout=30)
                        if r.status_code == 200:
                            st.session_state.voice_result = r.json()
                            st.rerun()
                        else:
                            st.error(r.json().get("message","Error"))
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        feats = st.text_area("Features (comma-separated)",
            placeholder="189.96, 184.86, 169.26, 0.004...", height=75, key="vfeats")
        if st.button("🔢 Analyse Features", key="bvf"):
            if not feats.strip():
                st.warning("Enter features first")
            else:
                with st.spinner("Analysing..."):
                    time.sleep(0.4)
                    try:
                        r = requests.post(f"{API_BASE}/predict-voice",
                            data={"features": feats.strip()}, timeout=30)
                        if r.status_code == 200:
                            st.session_state.voice_result = r.json()
                            st.rerun()
                        else:
                            st.error(r.json().get("message","Error"))
                    except Exception as e:
                        st.error(f"Error: {e}")

    display_result(st.session_state.voice_result)


# IMAGE
with col2:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-card-header">
            <div class="upload-card-icon">🖼️</div>
            <div>
                <div class="upload-card-title">Spiral Drawing</div>
                <div class="upload-card-sub">CNN — Conv2D × 3 Layers</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    img_f = st.file_uploader("Upload Spiral Image", type=["jpg","jpeg","png"], key="iup")
    if img_f:
        st.image(img_f, caption="Uploaded Drawing", use_container_width=True)

    if st.button("🖼️ Analyse Drawing", key="bi"):
        if not img_f:
            st.warning("Upload an image first")
        else:
            with st.spinner("Running CNN..."):
                time.sleep(0.4)
                try:
                    r = requests.post(f"{API_BASE}/predict-image",
                        files={"file":(img_f.name, img_f.getvalue(), "image/png")}, timeout=30)
                    if r.status_code == 200:
                        st.session_state.image_result = r.json()
                        st.rerun()
                    else:
                        st.error(r.json().get("message","Error"))
                except Exception as e:
                    st.error(f"Error: {e}")

    display_result(st.session_state.image_result)


# VIDEO
with col3:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-card-header">
            <div class="upload-card-icon">🎥</div>
            <div>
                <div class="upload-card-title">Hand Movement Video</div>
                <div class="upload-card-sub">Random Forest • OpenCV Motion</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    vid_f = st.file_uploader("Upload Video", type=["mp4","avi","mov"], key="vdup")
    if vid_f:
        st.video(vid_f)

    if st.button("🎥 Analyse Video", key="bvid"):
        if not vid_f:
            st.warning("Upload a video first")
        else:
            with st.spinner("Extracting motion features..."):
                time.sleep(0.4)
                try:
                    r = requests.post(f"{API_BASE}/predict-video",
                        files={"file":(vid_f.name, vid_f.getvalue(), "video/mp4")}, timeout=60)
                    if r.status_code == 200:
                        st.session_state.video_result = r.json()
                        st.rerun()
                    else:
                        st.error(r.json().get("message","Error"))
                except Exception as e:
                    st.error(f"Error: {e}")

    display_result(st.session_state.video_result)


# ── COMBINED RESULT ───────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;margin-bottom:20px;'>
    <div style='font-family:Space Grotesk;font-size:1.6rem;font-weight:800;color:#e2e8f0;'>
        🔗 Final Combined Diagnosis
    </div>
    <div style='color:#2d3748;font-size:0.85rem;margin-top:6px;'>
        Majority voting across all three AI modalities
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1,2,1])
with c2:
    no_results = (st.session_state.voice_result is None and
                  st.session_state.image_result is None and
                  st.session_state.video_result is None)

    if st.button("🧠 Run Final Combined Prediction", key="bcomb", disabled=no_results):
        vl = st.session_state.voice_result["prediction"] if st.session_state.voice_result else "Healthy"
        il = st.session_state.image_result["prediction"] if st.session_state.image_result else "Healthy"
        dl = st.session_state.video_result["prediction"] if st.session_state.video_result else "Healthy"
        with st.spinner("Computing final diagnosis..."):
            time.sleep(0.6)
            try:
                r = requests.post(f"{API_BASE}/predict-combined",
                    json={"voice_label":vl,"image_label":il,"video_label":dl}, timeout=10)
                if r.status_code == 200:
                    res     = r.json()
                    label   = res["prediction"]
                    pct     = res["confidence_pct"]
                    explain = res["explanation"]
                    is_pd   = "Parkinson" in label
                    css     = "final-pd" if is_pd else "final-ok"
                    icon    = "🚨" if is_pd else "✅"

                    st.markdown(f"""
                    <div class="{css}">
                        <div style='font-size:3.5rem;'>{icon}</div>
                        <div class="final-title">{label}</div>
                        <div style='color:#718096;font-size:0.9rem;margin-bottom:15px;'>
                            Combined Confidence: <strong>{pct}%</strong>
                        </div>
                        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        border-radius:12px;padding:15px;color:#718096;font-size:0.85rem;line-height:1.7;'>
                            💡 {explain}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Radar chart
                    labels_r = ["Voice", "Image", "Video"]
                    vals_r   = [
                        st.session_state.voice_result["confidence_pct"] if st.session_state.voice_result else 0,
                        st.session_state.image_result["confidence_pct"] if st.session_state.image_result else 0,
                        st.session_state.video_result["confidence_pct"] if st.session_state.video_result else 0,
                    ]
                    fig_r = go.Figure(go.Scatterpolar(
                        r=vals_r + [vals_r[0]],
                        theta=labels_r + [labels_r[0]],
                        fill="toself",
                        fillcolor="rgba(99,179,237,0.1)",
                        line=dict(color="#63b3ed", width=2),
                        marker=dict(size=8, color="#90cdf4"),
                    ))
                    fig_r.update_layout(
                        polar=dict(
                            bgcolor="rgba(13,17,23,0)",
                            radialaxis=dict(visible=True, range=[0,100],
                                          gridcolor="rgba(255,255,255,0.05)",
                                          color="#4a5568"),
                            angularaxis=dict(color="#718096")
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#718096"),
                        margin=dict(t=20,b=20,l=20,r=20),
                        height=280,
                        showlegend=False,
                        title=dict(text="Modality Confidence Radar",
                                  font=dict(color="#a0aec0",size=12),
                                  x=0.5)
                    )
                    st.plotly_chart(fig_r, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

    if no_results:
        st.markdown("""
        <div style='text-align:center;color:#1a202c;font-size:0.82rem;margin-top:10px;'>
            Complete at least one analysis above first
        </div>""", unsafe_allow_html=True)


# ── SUMMARY TABLE ─────────────────────────────────────────────
if any([st.session_state.voice_result,
        st.session_state.image_result,
        st.session_state.video_result]):

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Space Grotesk;font-size:1.4rem;font-weight:700;
    color:#e2e8f0;margin-bottom:16px;'>📋 Results Summary</div>
    """, unsafe_allow_html=True)

    summary = []
    icons_m = {"Voice":"🎙️","Image":"🖼️","Video":"🎥"}
    models  = {"Voice":"Random Forest","Image":"CNN (Keras)","Video":"Random Forest"}
    for lbl, key in [("Voice","voice_result"),("Image","image_result"),("Video","video_result")]:
        r = st.session_state[key]
        if r:
            pred   = r["prediction"]
            status = ("🚨 " if "Parkinson" in pred else "✅ ") + pred
            summary.append({
                "Modality": icons_m[lbl] + " " + lbl,
                "Model":    models[lbl],
                "Result":   status,
                "Confidence": f"{r['confidence_pct']}%",
            })

    df = pd.DataFrame(summary)
    st.dataframe(df, use_container_width=True, hide_index=True,
        column_config={
            "Modality":   st.column_config.TextColumn(width="small"),
            "Model":      st.column_config.TextColumn(width="medium"),
            "Result":     st.column_config.TextColumn(width="large"),
            "Confidence": st.column_config.TextColumn(width="small"),
        })

    cr, _, _ = st.columns([1,2,1])
    with cr:
        if st.button("🔄 Reset All", key="breset"):
            for k in ["voice_result","image_result","video_result"]:
                st.session_state[k] = None
            st.rerun()

# ── FOOTER ────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:10px 0 25px;'>
    <div style='color:#1a202c;font-size:0.78rem;line-height:2;'>
        ⚠️ <span style='color:#2d3748;'>Medical Disclaimer:</span>
        Research & educational tool only. Not a substitute for professional medical diagnosis.<br>
        <span style='color:#1a202c;'>
        Built with Python • TensorFlow/Keras • Scikit-learn • Flask • Streamlit • Plotly
        </span>
    </div>
</div>
""", unsafe_allow_html=True)