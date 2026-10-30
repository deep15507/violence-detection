"""Streamlit front-end for the Real-Life Violence Detection API.

Run:  streamlit run frontend/app.py
"""
import html
import json
import os
import time

import plotly.graph_objects as go
import requests
import streamlit as st

DEFAULT_API = os.getenv("API_URL", "http://localhost:8000")
RED, GREEN, INDIGO = "#ef4444", "#22c55e", "#6366f1"

st.set_page_config(page_title="Violence Detection AI", page_icon="🛡️", layout="wide")

# ----------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1200px;}

.hero {
  padding: 2rem 2.2rem; border-radius: 20px; margin-bottom: 1.5rem;
  background: radial-gradient(1200px 400px at 0% 0%, rgba(99,102,241,.35), transparent 60%),
              radial-gradient(900px 400px at 100% 100%, rgba(239,68,68,.20), transparent 60%),
              #131a2e;
  border: 1px solid rgba(255,255,255,.07);
}
.hero h1 {margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -.5px;}
.hero p  {margin: .5rem 0 0; color: #aab2c8; font-size: 1.02rem;}
.pill {display:inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .75rem;
       background: rgba(99,102,241,.18); color:#c7caff; margin: .9rem .4rem 0 0;
       border: 1px solid rgba(99,102,241,.35);}

.card {background:#131a2e; border:1px solid rgba(255,255,255,.07); border-radius:16px; padding:1.2rem 1.4rem;}
.step {font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; color:#8b93ad; margin-bottom:.4rem;}

.verdict {border-radius:18px; padding:1.6rem 1.8rem; border:1px solid; text-align:center;}
.verdict.bad  {background:linear-gradient(135deg,rgba(239,68,68,.22),rgba(239,68,68,.06)); border-color:rgba(239,68,68,.55);}
.verdict.good {background:linear-gradient(135deg,rgba(34,197,94,.22),rgba(34,197,94,.06)); border-color:rgba(34,197,94,.55);}
.verdict .icon {font-size:2.6rem; line-height:1;}
.verdict .label {font-size:2rem; font-weight:800; margin:.3rem 0 0;}
.verdict.bad .label {color:#fca5a5;} .verdict.good .label {color:#86efac;}
.verdict .sub {color:#c3c9dc; margin-top:.3rem; font-size:.95rem;}

.metric {background:#131a2e; border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:.9rem 1rem; text-align:center;}
.metric .v {font-size:1.35rem; font-weight:700;}
.metric .k {font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; color:#8b93ad;}

div.stButton > button[kind="primary"] {
  width:100%; height:3.1rem; font-weight:700; font-size:1.05rem; border:0; border-radius:12px;
  background:linear-gradient(90deg,#6366f1,#8b5cf6);
}
div.stButton > button[kind="primary"]:hover {filter:brightness(1.12);}
[data-testid="stFileUploader"] section {border:2px dashed rgba(99,102,241,.5); border-radius:16px; background:#10162a;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- state
st.session_state.setdefault("history", [])
st.session_state.setdefault("result", None)
st.session_state.setdefault("error", None)


# ----------------------------------------------------------------------------- helpers
@st.cache_data(ttl=5, show_spinner=False)
def check_health(api_url: str):
    try:
        r = requests.get(f"{api_url}/health", timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def call_predict(api_url: str, file, threshold: float):
    files = {"file": (file.name, file.getvalue(), file.type or "video/mp4")}
    r = requests.post(f"{api_url}/predict", files=files, params={"threshold": threshold}, timeout=600)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:  # noqa: BLE001
            detail = r.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        raise RuntimeError(f"{detail} (HTTP {r.status_code})")
    return r.json()


def fmt_time(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def gauge(p_violence: float, threshold: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_violence * 100,
            number={"suffix": "%", "font": {"size": 44}},
            title={"text": "Violence probability", "font": {"size": 15, "color": "#aab2c8"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#586082"},
                "bar": {"color": RED if p_violence >= threshold else GREEN, "thickness": 0.28},
                "bgcolor": "#1b2340",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(34,197,94,.18)"},
                    {"range": [threshold * 100, 100], "color": "rgba(239,68,68,.18)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 3}, "thickness": 0.8, "value": threshold * 100},
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6e9f2"})
    return fig


def timeline(segments: list, threshold: float):
    labels = [f"{fmt_time(s['start_seconds'])}–{fmt_time(s['end_seconds'])}" for s in segments]
    vals = [s["violence_probability"] * 100 for s in segments]
    colors = [RED if s["label"] == "Violence" else GREEN for s in segments]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors,
                           text=[f"{v:.0f}%" for v in vals], textposition="outside",
                           hovertemplate="%{x}<br>Violence: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=threshold * 100, line_dash="dash", line_color="#ffffff",
                  annotation_text=f"threshold {threshold:.0%}", annotation_position="top left")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 115], title="P(Violence) %"),
                      xaxis_title="Video time", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#e6e9f2"}, showlegend=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)")
    return fig


def metric(col, value, key):
    col.markdown(f'<div class="metric"><div class="v">{value}</div><div class="k">{key}</div></div>',
                 unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_url = st.text_input("API URL", DEFAULT_API, help="Where the FastAPI server is running").rstrip("/")
    threshold = st.slider("Violence threshold", 0.05, 0.95, 0.50, 0.05,
                          help="A video is flagged as Violence when P(Violence) ≥ this value. "
                               "Lower = more sensitive (more false alarms).")

    health = check_health(api_url)
    if "error" in health:
        st.error("🔴 API unreachable")
        st.caption(f"Start it with:\n`uvicorn backend.main:app --port 8000`")
    elif health.get("model_loaded"):
        st.success("🟢 API online · model loaded")
    else:
        st.warning("🟠 API online but model NOT loaded — check server logs")
    if st.button("↻ Re-check connection", use_container_width=True):
        check_health.clear()
        st.rerun()

    st.divider()
    st.markdown("### 🕘 Recent analyses")
    if not st.session_state.history:
        st.caption("Nothing yet — upload a video to get started.")
    for h in reversed(st.session_state.history[-6:]):
        icon = "🔴" if h["label"] == "Violence" else "🟢"
        st.markdown(f"{icon} **{html.escape(h['name'][:26])}**  \n"
                    f"<span style='color:#8b93ad;font-size:.8rem'>{h['label']} · {h['conf']:.0%}</span>",
                    unsafe_allow_html=True)

    st.divider()
    with st.expander("ℹ️ About the model"):
        st.markdown(
            "**Architecture:** MobileNetV2 (fine-tuned) → BiLSTM → dense head\n\n"
            "**Input:** 16 frames spread evenly across the video, resized to 64×64\n\n"
            "**Test accuracy:** 93% (200 held-out videos)\n\n"
            "**Dataset:** Real Life Violence Situations\n\n"
            "_Predictions are probabilistic and can be wrong — use as decision support, not as proof._"
        )

# ----------------------------------------------------------------------------- hero
st.markdown(
    """
<div class="hero">
  <h1>🛡️ Real-Life Violence Detection</h1>
  <p>Upload a video and our deep-learning model will tell you whether it contains violence.</p>
  <span class="pill">MobileNetV2</span><span class="pill">BiLSTM</span>
  <span class="pill">FastAPI</span><span class="pill">93% test accuracy</span>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- main layout
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="step">Step 1 · Upload</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop a video here", type=["mp4", "avi", "mov", "mkv", "webm"],
                                label_visibility="collapsed")
    if uploaded:
        st.video(uploaded.getvalue())
        st.caption(f"📄 {uploaded.name} · {len(uploaded.getvalue()) / 1024 / 1024:.1f} MB")

    st.markdown('<div class="step" style="margin-top:1rem">Step 2 · Analyze</div>', unsafe_allow_html=True)
    go_btn = st.button("🔍 Analyze video", type="primary", disabled=uploaded is None)

    if go_btn and uploaded:
        st.session_state.error = None
        try:
            with st.spinner("Extracting frames and running the model…"):
                t0 = time.perf_counter()
                res = call_predict(api_url, uploaded, threshold)
                res["_roundtrip_s"] = time.perf_counter() - t0
            st.session_state.result = res
            st.session_state.history.append(
                {"name": res["filename"], "label": res["label"], "conf": res["confidence"]})
            st.rerun()
        except requests.exceptions.ConnectionError:
            st.session_state.result = None
            st.session_state.error = f"Cannot reach the API at {api_url}. Is the FastAPI server running?"
        except Exception as e:  # noqa: BLE001
            st.session_state.result = None
            st.session_state.error = str(e)

    if st.session_state.error:
        st.error(st.session_state.error)

with right:
    st.markdown('<div class="step">Result</div>', unsafe_allow_html=True)
    res = st.session_state.result
    if not res:
        st.markdown(
            '<div class="card" style="text-align:center;padding:3rem 1rem;color:#8b93ad">'
            '<div style="font-size:2.5rem">🎬</div>'
            'Your analysis will appear here.<br>Upload a video and press <b>Analyze</b>.</div>',
            unsafe_allow_html=True,
        )
    else:
        violent = res["is_violent"]
        cls, icon = ("bad", "🚨") if violent else ("good", "✅")
        text = "Violence Detected" if violent else "No Violence Detected"
        st.markdown(
            f'<div class="verdict {cls}"><div class="icon">{icon}</div>'
            f'<div class="label">{text}</div>'
            f'<div class="sub">Confidence <b>{res["confidence"]:.1%}</b> · file '
            f'<i>{html.escape(res["filename"])}</i></div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(gauge(res["probabilities"]["Violence"], res["threshold"]),
                        use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------------- details (full width)
res = st.session_state.result
if res:
    v = res["video"]
    st.markdown("#### 📊 Video details")
    cols = st.columns(5)
    metric(cols[0], fmt_time(v["duration_seconds"]), "Duration")
    metric(cols[1], f'{v["width"]}×{v["height"]}', "Resolution")
    metric(cols[2], f'{v["fps"]:g}', "FPS")
    metric(cols[3], f'{v["total_frames"]:,}', "Frames")
    metric(cols[4], f'{res["processing_time_ms"] / 1000:.2f}s', "Analysis time")

    segs = res.get("segments", [])
    if segs:
        st.markdown("#### ⏱️ Timeline analysis")
        n_bad = sum(s["label"] == "Violence" for s in segs)
        st.caption(f"The video was split into {len(segs)} parts and each was classified separately — "
                   f"**{n_bad} of {len(segs)}** parts flagged as violent.")
        st.plotly_chart(timeline(segs, res["threshold"]), use_container_width=True,
                        config={"displayModeBar": False})
        flagged = [s for s in segs if s["label"] == "Violence"]
        if flagged:
            st.warning("Flagged segments: " + ", ".join(
                f"{fmt_time(s['start_seconds'])}–{fmt_time(s['end_seconds'])}" for s in flagged))
    else:
        st.info("Video is too short for a timeline breakdown — showing the overall verdict only.")

    with st.expander("🧾 Raw API response"):
        st.json({k: v for k, v in res.items() if not k.startswith("_")})
    st.download_button("⬇️ Download report (JSON)",
                       json.dumps({k: v for k, v in res.items() if not k.startswith("_")}, indent=2),
                       file_name=f"{os.path.splitext(res['filename'])[0]}_report.json",
                       mime="application/json")
