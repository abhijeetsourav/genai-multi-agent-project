import streamlit as st
import asyncio
import json
from backend.analysis_engine import run_review_analysis

st.set_page_config(page_title="Customer Review Intelligence", layout="centered")

st.title("📊 Customer Review Intelligence")

query = st.text_input(
    "Ask a question about customer reviews",
    placeholder="Why are users unhappy with computer games?",
)

api_key = st.secrets["GROQCLOUD_API_KEY"]

if st.button("Analyze") and query:

    with st.spinner("Analyzing customer reviews..."):
        analysis = asyncio.run(run_review_analysis(query, api_key))

        if isinstance(analysis, str):
            analysis = json.loads(analysis)

    # ---- UI Rendering ----

    meta = analysis["meta"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Topic", meta["topic"])
    col2.metric("Scope", meta["analysis_scope"])
    col3.metric("Confidence", meta["confidence_level"])
    col4.metric("Evidence", meta["evidence_count"])

    st.divider()

    st.subheader("🔴 Primary Issue")
    st.error(analysis["insight"]["primary_issue"])

    st.subheader("📉 Business Impact")
    for item in analysis["impact"]:
        st.write("•", item)

    st.subheader("✅ Recommended Actions")
    for i, action in enumerate(analysis["actions"], 1):
        st.success(f"{i}. {action}")

    st.subheader("🧾 Customer Evidence")
    for ev in analysis["evidence"]:
        with st.expander("View Review"):
            st.write(ev["text"])
            st.caption(f"Rating: {ev['score']}")

    if meta["confidence_level"].lower() != "high":
        st.warning("Insight based on limited evidence. " "Validate with more samples.")
