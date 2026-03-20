"""
ui/app.py — Streamlit UI for Intelligent Document Processing
"""

import streamlit as st
import requests
import json
import plotly.graph_objects as go

API_URL = "http://localhost:8000"

st.set_page_config(page_title="IDP — Document Extraction", page_icon="📄", layout="wide")

st.markdown("""
<style>
.high   { color: #28a745; font-weight: bold; }
.medium { color: #fd7e14; font-weight: bold; }
.low    { color: #dc3545; font-weight: bold; }
.field-card {
    background: #f8f9fa; border-radius: 6px;
    padding: 10px 14px; margin: 4px 0;
    border-left: 4px solid #dee2e6;
}
.field-high   { border-left-color: #28a745; }
.field-medium { border-left-color: #fd7e14; }
.field-low    { border-left-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📄 IDP System")
    st.caption("Intelligent Document Processing")
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
    except:
        st.error("🔴 API Offline")
        st.stop()

tab_extract, tab_queue, tab_batch = st.tabs(["📤 Extract", "🔍 Review Queue", "📦 Batch"])

# ── EXTRACT TAB ───────────────────────────────────────────────────────────
with tab_extract:
    st.header("Extract Document Fields")
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader("Upload document", type=["pdf", "docx", "txt", "md", "png", "jpg"])
    with col2:
        try:
            dt_resp = requests.get(f"{API_URL}/document-types", timeout=5).json()
            doc_types = ["auto-detect"] + dt_resp.get("supported_types", [])
        except:
            doc_types = ["auto-detect"]
        selected_type = st.selectbox("Document type", doc_types)

    if uploaded and st.button("⚡ Extract", type="primary"):
        with st.spinner("Extracting fields..."):
            files  = [("file", (uploaded.name, uploaded.getvalue(), uploaded.type))]
            data   = {} if selected_type == "auto-detect" else {"doc_type": selected_type}
            resp   = requests.post(f"{API_URL}/process", files=files, data=data, timeout=120)

        if resp.ok:
            result = resp.json()
            # Header
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Document Type", result["doc_type"].replace("_", " ").title())
            col_b.metric("Classification", f"{result['classification_conf']:.0%}")
            col_c.metric("Pages", result["page_count"])
            col_d.metric("Processing Time", f"{result['processing_time_ms']:.0f}ms")

            review_flag = "⚠️ NEEDS REVIEW" if result["needs_review"] else "✅ AUTO-PROCESSED"
            st.markdown(f"**Status:** {review_flag}")
            if result["low_confidence_fields"]:
                st.warning(f"Low confidence fields: {', '.join(result['low_confidence_fields'])}")

            # Fields
            st.subheader("Extracted Fields")
            fields = result["extracted_fields"]
            conf_vals = []
            for fname, fdata in fields.items():
                if not isinstance(fdata, dict) or "value" not in fdata:
                    continue
                conf = fdata.get("confidence", 0)
                conf_vals.append(conf)
                cls = "field-high" if conf >= 0.85 else "field-medium" if conf >= 0.5 else "field-low"
                conf_cls = "high" if conf >= 0.85 else "medium" if conf >= 0.5 else "low"
                value = fdata.get("value", "—") or "—"
                st.markdown(
                    f'<div class="field-card {cls}">'
                    f'<b>{fname.replace("_"," ").title()}</b>: {value} '
                    f'<span class="{conf_cls}">({conf:.0%})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Confidence distribution
            if conf_vals:
                fig = go.Figure(go.Histogram(
                    x=conf_vals, nbinsx=10, marker_color="#007bff",
                ))
                fig.update_layout(
                    title="Confidence Distribution", height=200,
                    xaxis_title="Confidence", yaxis_title="Count",
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Raw JSON"):
                st.json(result)
        else:
            st.error(f"Error: {resp.json().get('detail', resp.text)}")

# ── REVIEW QUEUE TAB ──────────────────────────────────────────────────────
with tab_queue:
    st.header("Human Review Queue")
    try:
        queue = requests.get(f"{API_URL}/queue", timeout=10).json()
    except:
        queue = []

    if not queue:
        st.info("🎉 No documents pending review.")
    else:
        st.warning(f"{len(queue)} document(s) need human review")
        for item in queue:
            with st.expander(f"📄 {item['filename']} — {item['doc_type']} (ID: {item['id']})"):
                st.write(f"**Low confidence fields:** {', '.join(item['low_confidence_fields'])}")
                corrections = {}
                for fname in item["low_confidence_fields"]:
                    fdata      = item["extracted_fields"].get(fname, {})
                    curr_val   = fdata.get("value", "") if isinstance(fdata, dict) else ""
                    curr_conf  = fdata.get("confidence", 0) if isinstance(fdata, dict) else 0
                    corrected  = st.text_input(
                        f"{fname.replace('_',' ').title()} (current: {curr_val}, conf: {curr_conf:.0%})",
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
    st.header("Batch Processing")
    uploaded_batch = st.file_uploader(
        "Upload multiple documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_batch:
        st.write(f"{len(uploaded_batch)} files selected")
        if st.button("⚡ Process Batch", type="primary"):
            with st.spinner(f"Processing {len(uploaded_batch)} documents..."):
                files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_batch]
                resp  = requests.post(f"{API_URL}/process/batch", files=files, timeout=300)

            if resp.ok:
                data = resp.json()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total",         data["total"])
                col2.metric("Auto-processed", data["auto"])
                col3.metric("Need Review",    data["needs_review"])

                for r in data["results"]:
                    icon = "⚠️" if r["needs_review"] else "✅"
                    st.write(f"{icon} **{r['filename']}** → {r['doc_type']} ({r['classification_conf']:.0%}) | {r['processing_time_ms']:.0f}ms")
            else:
                st.error(resp.text)
