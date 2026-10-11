"""Streamlit frontend for Educational Video Retrieval System."""

import os
import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path so imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils.api_client import BackendAPIClient
from frontend.components.video_card import render_video_card

# Page layout & styling
st.set_page_config(
    page_title="Educational Video Retrieval System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API Client
api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
api_client = BackendAPIClient(base_url=api_base_url)


def main():
    # Header
    st.title("🎓 Educational Video Retrieval System")
    st.markdown(
        "Ask a conceptual question to retrieve the most relevant educational YouTube lectures "
        "and **jump directly to the exact timestamped portion** explaining your topic."
    )

    # Sidebar: System health & example queries
    with st.sidebar:
        st.header("⚙️ System Status")
        health = api_client.check_health()
        if health.get("healthy"):
            st.success("Backend: Active & Ready")
            st.caption(f"Model: `{health.get('data', {}).get('embedding_model', 'all-MiniLM-L6-v2')}`")
        else:
            st.error("Backend: Disconnected")
            st.caption(f"Error: {health.get('error')}")
            st.info("Start backend with: `uvicorn backend.main:app --port 8000`")

        st.markdown("---")
        st.subheader("💡 Sample Questions")
        example_queries = [
            "What is backpropagation in neural networks?",
            "How does gradient descent optimize weights?",
            "Explain attention mechanism in Transformers",
            "What is convolution in deep learning?",
            "How does merge sort work and what is its time complexity?"
        ]

        for eq in example_queries:
            if st.button(f"📌 {eq}", key=f"btn_{eq}", use_container_width=True):
                st.session_state["search_box"] = eq
                st.session_state["auto_search"] = True
                st.rerun()

    # Main search input
    if "search_box" not in st.session_state:
        st.session_state["search_box"] = ""

    query = st.text_input(
        "Enter your question or lecture concept:",
        placeholder="e.g. Binary search algorithm or Photosynthesis in plants",
        key="search_box"
    )

    col_btn, col_count, _ = st.columns([1, 1, 3])
    with col_btn:
        search_clicked = st.button("🔍 Search Portions", type="primary", use_container_width=True)
    with col_count:
        top_k = st.selectbox("Top Results", options=[3, 6, 8], index=1, label_visibility="collapsed")

    # Check if auto_search was triggered by a sidebar button
    auto_trigger = st.session_state.pop("auto_search", False)
    do_search = search_clicked or auto_trigger

    # Search execution
    if do_search:
        if not query.strip():
            st.warning("Please type a question or select one of the sample questions above.")
        else:
            with st.spinner("Retrieving videos, extracting transcripts, and computing semantic matches with FAISS..."):
                response = api_client.search_videos(query=query.strip(), top_k=top_k)

                if not response.get("success"):
                    st.error(response.get("error", "An error occurred during search."))
                else:
                    results = response.get("results", [])
                    exec_time = response.get("execution_time", 0.0)
                    total_candidates = response.get("total_candidates", 0)
                    total_chunks = response.get("total_chunks", 0)

                    if not results:
                        st.info(
                            "No relevant lecture segments could be matched. "
                            "Try rephrasing your query or using broader conceptual terms."
                        )
                    else:
                        st.success(
                            f"Found **{len(results)}** relevant educational videos in **{exec_time:.2f}s** "
                            f"(processed {total_chunks} transcript portions across {total_candidates} candidates)."
                        )

                        st.subheader("Top Relevant Lecture Portions")
                        for rank, item in enumerate(results, start=1):
                            render_video_card(item, rank_index=rank)


if __name__ == "__main__":
    main()
