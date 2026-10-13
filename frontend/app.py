"""Streamlit frontend for Educational Video Retrieval System."""

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils.api_client import BackendAPIClient
from frontend.components.video_card import render_video_card


st.set_page_config(
    page_title="Learn smarter | Educational Video Search",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
api_client = BackendAPIClient(base_url=api_base_url)

TOPICS = [
    "Machine Learning",
    "Data Structures",
    "DBMS",
    "Computer Networks",
    "Python",
]


def _select_topic(topic: str) -> None:
    st.session_state["search_box"] = topic


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --canvas: #080d1b;
            --surface: #10182b;
            --surface-raised: #151f36;
            --ink: #f3f5ff;
            --muted: #a5b0c9;
            --violet: #a78bfa;
            --blue: #60a5fa;
            --line: rgba(166, 183, 220, .15);
        }
        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse at 50% -20%, rgba(91, 74, 190, .24), transparent 55%),
                var(--canvas);
            color: var(--ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMain"] > div {
            max-width: 1120px;
            padding-top: 1.5rem;
            padding-bottom: 4rem;
        }
        [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
        h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
            color: var(--ink);
        }
        .brand-row {
            display: flex; align-items: center; justify-content: space-between;
            margin: 0 auto 4.5rem; max-width: 940px;
        }
        .brand {
            display: flex; align-items: center; gap: .7rem;
            color: #e7eaff; font-size: 1rem; font-weight: 700;
            letter-spacing: .01em;
        }
        .brand-mark {
            display: grid; place-items: center; width: 2.35rem; height: 2.35rem;
            border-radius: .8rem; color: white;
            background: linear-gradient(135deg, #8b5cf6, #3b82f6);
            box-shadow: 0 8px 26px rgba(96, 87, 220, .28);
        }
        .health-pill {
            border: 1px solid var(--line); border-radius: 999px;
            padding: .45rem .75rem; color: #c3cde2; font-size: .78rem;
            background: rgba(16, 24, 43, .68);
        }
        .hero { max-width: 850px; margin: 0 auto 2.2rem; text-align: center; }
        .eyebrow {
            color: #b8a3ff; font-size: .78rem; font-weight: 700;
            letter-spacing: .16em; text-transform: uppercase;
        }
        .hero h1 {
            margin: .9rem 0 1rem; font-size: clamp(2.25rem, 5.3vw, 4.25rem);
            line-height: 1.08; letter-spacing: -.045em; font-weight: 750;
        }
        .gradient-text {
            background: linear-gradient(100deg, #c4b5fd 10%, #93c5fd 85%);
            background-clip: text; -webkit-background-clip: text;
            color: transparent;
        }
        .hero-copy {
            max-width: 640px; margin: 0 auto; color: var(--muted);
            font-size: 1.08rem; line-height: 1.7;
        }
        [data-testid="stTextInput"] label,
        [data-testid="stSelectbox"] label { color: #dbe4fa; font-weight: 600; }
        [data-testid="stTextInput"] input {
            min-height: 3.45rem; color: var(--ink);
            border: 1px solid rgba(167, 139, 250, .3);
            border-radius: 1rem; background: rgba(16, 24, 43, .9);
            font-size: 1rem; box-shadow: 0 14px 45px rgba(0, 0, 0, .15);
        }
        [data-testid="stTextInput"] input:focus {
            border-color: var(--violet);
            box-shadow: 0 0 0 2px rgba(167, 139, 250, .2);
        }
        [data-testid="stTextInput"] input::placeholder { color: #7f8ba5; }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            min-height: 3rem; border-radius: .85rem;
            border-color: var(--line); background: var(--surface);
        }
        .stButton > button, [data-testid="stFormSubmitButton"] button {
            min-height: 2.75rem; border-radius: .85rem;
            border: 1px solid var(--line); background: var(--surface);
            color: #e7eaff; font-weight: 600;
            transition: border-color .18s ease, transform .18s ease, background .18s ease;
        }
        .stButton > button:hover {
            border-color: rgba(167, 139, 250, .65);
            background: #19243c; transform: translateY(-1px);
        }
        [data-testid="stFormSubmitButton"] button {
            width: 100%; border: 0; color: white;
            background: linear-gradient(100deg, #7c3aed, #2563eb);
            box-shadow: 0 10px 30px rgba(81, 76, 210, .25);
        }
        [data-testid="stFormSubmitButton"] button:hover {
            border: 0; color: white;
            background: linear-gradient(100deg, #8b5cf6, #3b82f6);
        }
        .suggest-label {
            margin: 1.1rem 0 .55rem; color: #9ba8c2;
            font-size: .8rem; text-align: center;
        }
        .results-heading { margin: 2.4rem 0 1rem; }
        .results-heading p { margin: .35rem 0 0; color: var(--muted); }
        .empty-state {
            margin-top: 2rem; padding: 2rem; text-align: center;
            border: 1px solid var(--line); border-radius: 1.2rem;
            background: rgba(16, 24, 43, .7);
        }
        .empty-state p { color: var(--muted); }
        [data-testid="stSpinner"] { color: #c4b5fd; }
        [data-testid="stAlert"] { border-radius: 1rem; }
        @media (max-width: 700px) {
            [data-testid="stMain"] > div { padding: 1rem 1rem 3rem; }
            .brand-row { margin-bottom: 3rem; }
            .hero { margin-bottom: 1.4rem; }
            .hero-copy { font-size: .98rem; }
            [data-testid="stHorizontalBlock"] { gap: .55rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important; transition-duration: .01ms !important;
                animation-duration: .01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_health() -> dict:
    now = time.time()
    checked_at = st.session_state.get("_health_checked_at", 0)
    if now - checked_at > 10:
        st.session_state["_health_cache"] = api_client.check_health(timeout=5.0)
        st.session_state["_health_checked_at"] = now
    return st.session_state.get("_health_cache", {"healthy": None})


def main() -> None:
    _inject_styles()
    health = _get_health()
    health_text = (
        "● Backend ready"
        if health.get("healthy") is True
        else "● Checking backend"
        if health.get("healthy") is None
        else "● Backend unavailable"
    )

    st.markdown(
        f"""
        <div class="brand-row">
            <div class="brand"><span class="brand-mark">✦</span> LectureLens</div>
            <div class="health-pill">{health_text}</div>
        </div>
        <section class="hero">
            <div class="eyebrow">Your study session, upgraded</div>
            <h1>Learn smarter.<br><span class="gradient-text">Find the exact moment.</span></h1>
            <p class="hero-copy">
                Search a concept and jump straight to the part of a lecture
                that makes it click.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.form("lecture_search", clear_on_submit=False):
        query = st.text_input(
            "What do you want to understand?",
            placeholder="Try “how does gradient descent work?”",
            key="search_box",
            label_visibility="collapsed",
        )
        count_col, submit_col = st.columns([1, 2])
        with count_col:
            top_k = st.selectbox(
                "Videos to show",
                options=[3, 6, 8],
                index=1,
                format_func=lambda count: f"{count} videos",
            )
        with submit_col:
            submitted = st.form_submit_button("⌕  Find lecture moments")

    st.markdown('<div class="suggest-label">Explore a popular subject</div>', unsafe_allow_html=True)
    topic_columns = st.columns(len(TOPICS))
    for column, topic in zip(topic_columns, TOPICS):
        with column:
            st.button(
                topic,
                key=f"topic_{topic}",
                use_container_width=True,
                on_click=_select_topic,
                args=(topic,),
            )

    if submitted:
        if not query.strip():
            st.session_state["_search_error"] = "Add a topic or question to start a search."
            st.session_state.pop("_search_data", None)
        else:
            st.session_state.pop("_search_error", None)
            try:
                with st.spinner("Finding lectures and matching transcript moments…"):
                    response = api_client.search_videos(
                        query=query.strip(),
                        top_k=top_k,
                    )
                if response.get("success"):
                    st.session_state["_search_data"] = {
                        "query": query.strip(),
                        "top_k": top_k,
                        "response": response,
                    }
                    st.session_state.pop("_search_error", None)
                else:
                    st.session_state.pop("_search_data", None)
                    st.session_state["_search_error"] = response.get(
                        "error",
                        "The search could not be completed. Please try again.",
                    )
            except Exception as exc:
                st.session_state.pop("_search_data", None)
                st.session_state["_search_error"] = (
                    f"Search failed: {exc}. Check the backend connection and try again."
                )

    error = st.session_state.get("_search_error")
    if error:
        st.error(error)

    search_data = st.session_state.get("_search_data")
    if not search_data:
        return

    response = search_data["response"]
    results = response.get("results", [])
    st.markdown(
        f"""
        <div class="results-heading">
            <div class="eyebrow">Your search</div>
            <h2>Moments for “{_escape(search_data["query"])}”</h2>
            <p>{len(results)} unique lecture{"s" if len(results) != 1 else ""} found
            · requested {search_data["top_k"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not results:
        st.markdown(
            """
            <div class="empty-state">
                <h3>No lecture moments found yet</h3>
                <p>Try a broader phrase, or search another subject.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.caption(
        f"{response.get('total_candidates', 0)} videos checked · "
        f"{response.get('total_chunks', 0)} transcript segments searched · "
        f"{response.get('execution_time', 0.0):.1f}s"
    )
    for rank, item in enumerate(results, start=1):
        render_video_card(item, rank_index=rank)


def _escape(value: str) -> str:
    import html

    return html.escape(value, quote=True)


if __name__ == "__main__":
    main()
