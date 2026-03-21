"""
ui/app.py — Streamlit UI for Intelligent Document Processing
Premium glassmorphism design
"""

import streamlit as st
import requests
import json
import plotly.graph_objects as go

API_URL = "http://localhost:8000"

st.set_page_config(page_title="IDP — Smart Document Extraction", page_icon="📄", layout="wide")

# ── Premium CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 40%, #1e3a5f 100%);
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; }
header[data-testid="stHeader"] { background: transparent; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0a2e 0%, #16082a 100%) !important;
    border-right: 1px solid rgba(168,85,247,0.15);
}
section[data-testid="stSidebar"] .stMarkdown { color: #d4c4e8; }

.hero {
    text-align: center; padding: 40px 20px 20px 20px;
}
.hero h1 {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #a855f7, #ec4899, #f97316);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px; letter-spacing: -1px;
}
.hero p { color: #9f8cb8; font-size: 1.15rem; font-weight: 300; }

.glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(168,85,247,0.12);
    border-radius: 16px; padding: 24px; margin: 12px 0; color: #d4c4e8;
}

.field-card {
    background: rgba(255,255,255,0.04);
    border-radius: 10px; padding: 14px 18px; margin: 6px 0;
    border-left: 4px solid rgba(168,85,247,0.4);
    color: #d4c4e8; transition: transform 0.2s ease;
}
.field-card:hover { transform: translateX(4px); }
.field-high   { border-left-color: #34d399; }
.field-medium { border-left-color: #f59e0b; }
.field-low    { border-left-color: #ef4444; }

.conf-high   { color: #34d399; font-weight: bold; }
.conf-medium { color: #f59e0b; font-weight: bold; }
.conf-low    { color: #ef4444; font-weight: bold; }

.metric-glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(168,85,247,0.12);
    border-radius: 14px; padding: 20px; text-align: center;
}
.metric-glass .value {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-glass .label { color: #9f8cb8; font-size: 0.85rem; margin-top: 4px; }

.status-review {
    background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05));
    border: 1px solid rgba(245,158,11,0.4);
    border-radius: 10px; padding: 12px 18px; color: #fbbf24;
}
.status-auto {
    background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(52,211,153,0.05));
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 10px; padding: 12px 18px; color: #34d399;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03); border-radius: 12px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #9f8cb8; font-weight: 500; }
.stTabs [aria-selected="true"] {
    background: rgba(168,85,247,0.2) !important; color: #d8b4fe !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #a855f7, #ec4899) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(168,85,247,0.4) !important;
}

.stTextArea textarea, .stTextInput input, .stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(168,85,247,0.15) !important;
    border-radius: 10px !important; color: #d4c4e8 !important;
}

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important; color: #9f8cb8 !important;
}
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* Streamlit overrides */
footer, #MainMenu, .stDeployButton, div[data-testid="stDecoration"] { display: none !important; }
[data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebarContent"] { background: transparent !important; }
[data-testid="stBottomBlockContainer"] { background: transparent !important; }
div[data-testid="stMetricValue"] > div { color: #e2e8f0 !important; }
div[data-testid="stMetricDelta"] { color: #9f8cb8 !important; }
div[data-testid="stMetricLabel"] { color: #9f8cb8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📄 Intelligent Document Processing</h1>
    <p>AI-powered extraction with confidence scoring, human-in-the-loop review, and batch processing</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 IDP Dashboard")
    st.caption("Processing metrics & status")
    st.divider()
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success("🟢 API Connected")
        stats = requests.get(f"{API_URL}/stats", timeout=5).json()
        st.metric("Processed",    stats.get("total_processed", 0))
        st.metric("Auto-cleared", stats.get("auto_processed", 0))
        st.metric("Review Queue", stats.get("review_queue_size", 0))
        avg = stats.get("avg_processing_ms", 0)
        st.metric("Avg Time", f"{avg:.0f}ms")
    except Exception:
        st.error("🔴 API Offline")
        st.stop()
    st.divider()
    st.caption("Built with OpenAI · FastAPI · Streamlit")

tab_extract, tab_queue, tab_batch = st.tabs(["📤 Extract", "🔍 Review Queue", "📦 Batch Processing"])

# ── EXTRACT TAB ───────────────────────────────────────────────────────────
with tab_extract:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 📤 Upload & Extract Document Fields")
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader("Upload document", type=["pdf", "docx", "txt", "md", "png", "jpg"])
    with col2:
        try:
            dt_resp = requests.get(f"{API_URL}/document-types", timeout=5).json()
            doc_types = ["auto-detect"] + dt_resp.get("supported_types", [])
        except Exception:
            doc_types = ["auto-detect"]
        selected_type = st.selectbox("Document type", doc_types)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded and st.button("⚡ Extract Fields", type="primary", use_container_width=True):
        with st.spinner("Extracting fields..."):
            files = [("file", (uploaded.name, uploaded.getvalue(), uploaded.type))]
            data = {} if selected_type == "auto-detect" else {"doc_type": selected_type}
            resp = requests.post(f"{API_URL}/process", files=files, data=data, timeout=120)

        if resp.ok:
            result = resp.json()
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.markdown(
                    f'<div class="metric-glass"><div class="value" style="-webkit-text-fill-color:unset;'
                    f'color:#d8b4fe;font-size:1.3rem">{result["doc_type"].replace("_", " ").title()}</div>'
                    f'<div class="label">Document Type</div></div>',
                    unsafe_allow_html=True
                )
            with col_b:
                st.markdown(
                    f'<div class="metric-glass"><div class="value">{result["classification_conf"]:.0%}</div>'
                    f'<div class="label">Classification</div></div>',
                    unsafe_allow_html=True
                )
            with col_c:
                st.markdown(
                    f'<div class="metric-glass"><div class="value">{result["page_count"]}</div>'
                    f'<div class="label">Pages</div></div>',
                    unsafe_allow_html=True
                )
            with col_d:
                st.markdown(
                    f'<div class="metric-glass"><div class="value">{result["processing_time_ms"]:.0f}ms</div>'
                    f'<div class="label">Processing Time</div></div>',
                    unsafe_allow_html=True
                )

            if result["needs_review"]:
                st.markdown('<div class="status-review">⚠️ <b>NEEDS REVIEW</b> — Low confidence on some fields</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-auto">✅ <b>AUTO-PROCESSED</b> — All fields above threshold</div>',
                            unsafe_allow_html=True)

            if result["low_confidence_fields"]:
                st.warning(f"Low confidence fields: {', '.join(result['low_confidence_fields'])}")

            st.markdown("#### Extracted Fields")
            fields = result["extracted_fields"]
            conf_vals = []
            for fname, fdata in fields.items():
                if not isinstance(fdata, dict) or "value" not in fdata:
                    continue
                conf = fdata.get("confidence", 0)
                conf_vals.append(conf)
                cls = "field-high" if conf >= 0.85 else "field-medium" if conf >= 0.5 else "field-low"
                conf_cls = "conf-high" if conf >= 0.85 else "conf-medium" if conf >= 0.5 else "conf-low"
                value = fdata.get("value", "—") or "—"
                st.markdown(
                    f'<div class="field-card {cls}">'
                    f'<b>{fname.replace("_", " ").title()}</b>: {value} '
                    f'<span class="{conf_cls}">({conf:.0%})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if conf_vals:
                fig = go.Figure(go.Histogram(
                    x=conf_vals, nbinsx=10,
                    marker_color="rgba(168,85,247,0.6)",
                    marker_line=dict(color="rgba(168,85,247,0.8)", width=1),
                ))
                fig.update_layout(
                    title="Confidence Distribution", height=220,
                    xaxis_title="Confidence", yaxis_title="Count",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9f8cb8"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Raw JSON"):
                st.json(result)
        else:
            st.error(f"Error: {resp.json().get('detail', resp.text)}")

# ── REVIEW QUEUE TAB ──────────────────────────────────────────────────────
with tab_queue:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Human Review Queue")
    st.caption("Documents requiring manual verification")
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        queue = requests.get(f"{API_URL}/queue", timeout=10).json()
    except Exception:
        queue = []

    if not queue:
        st.info("🎉 No documents pending review — all clear!")
    else:
        st.warning(f"{len(queue)} document(s) need human review")
        for item in queue:
            with st.expander(f"📄 {item['filename']} — {item['doc_type']} (ID: {item['id']})"):
                st.write(f"**Low confidence fields:** {', '.join(item['low_confidence_fields'])}")
                corrections = {}
                for fname in item["low_confidence_fields"]:
                    fdata = item["extracted_fields"].get(fname, {})
                    curr_val = fdata.get("value", "") if isinstance(fdata, dict) else ""
                    curr_conf = fdata.get("confidence", 0) if isinstance(fdata, dict) else 0
                    corrected = st.text_input(
                        f"{fname.replace('_', ' ').title()} (current: {curr_val}, conf: {curr_conf:.0%})",
                        value=str(curr_val) if curr_val else "",
                        key=f"{item['id']}_{fname}",
                    )
                    if corrected:
                        corrections[fname] = corrected

                if st.button(f"✅ Submit Corrections", key=f"submit_{item['id']}"):
                    resp = requests.post(
                        f"{API_URL}/queue/{item['id']}/resolve",
                        json=corrections, timeout=30,
                    )
                    if resp.ok:
                        st.success("Corrections submitted!")
                        st.rerun()
                    else:
                        st.error("Failed to submit corrections.")

# ── BATCH TAB ─────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 📦 Batch Processing")
    st.caption("Upload and process multiple documents at once")
    uploaded_batch = st.file_uploader(
        "Upload multiple documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_batch:
        st.write(f"**{len(uploaded_batch)} files selected**")
        if st.button("⚡ Process Batch", type="primary", use_container_width=True):
            with st.spinner(f"Processing {len(uploaded_batch)} documents..."):
                files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_batch]
                resp = requests.post(f"{API_URL}/process/batch", files=files, timeout=300)

            if resp.ok:
                data = resp.json()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f'<div class="metric-glass"><div class="value">{data["total"]}</div>'
                        f'<div class="label">Total Processed</div></div>',
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f'<div class="metric-glass"><div class="value">{data["auto"]}</div>'
                        f'<div class="label">Auto-processed</div></div>',
                        unsafe_allow_html=True
                    )
                with col3:
                    st.markdown(
                        f'<div class="metric-glass"><div class="value">{data["needs_review"]}</div>'
                        f'<div class="label">Need Review</div></div>',
                        unsafe_allow_html=True
                    )

                for r in data["results"]:
                    icon = "⚠️" if r["needs_review"] else "✅"
                    st.write(f"{icon} **{r['filename']}** → {r['doc_type']} ({r['classification_conf']:.0%}) | {r['processing_time_ms']:.0f}ms")
            else:
                st.error(resp.text)
