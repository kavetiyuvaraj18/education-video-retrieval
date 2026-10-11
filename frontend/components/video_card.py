"""Streamlit UI Component for displaying ranked educational video cards."""

import streamlit as st
from typing import Dict, Any


def render_video_card(item: Dict[str, Any], rank_index: int):
    """
    Renders an educational video card with video thumbnail, rank badge,
    channel info, highlighted topic transcript, and direct timestamp link.
    """
    with st.container(border=True):
        col_thumb, col_info = st.columns([1, 2])

        with col_thumb:
            thumb_url = item.get("thumbnail_url")
            if thumb_url:
                st.image(thumb_url, use_container_width=True)

            # Direct playable link button
            timestamp_url = item.get("timestamp_url", "#")
            st.link_button(
                label=f"▶️ Jump to {item.get('start_time_formatted', '00:00')}",
                url=timestamp_url,
                type="primary",
                use_container_width=True
            )

        with col_info:
            title = item.get("title", "Lecture Video")
            st.markdown(f"### Rank #{rank_index}: [{title}]({timestamp_url})")

            # Channel and similarity score badges
            col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
            with col_b1:
                st.markdown(f"👤 **Channel:** `{item.get('channel_name', 'YouTube')}`")
            with col_b2:
                relevance = item.get("relevance_score", 0.0)
                st.markdown(f"⭐ **Score:** `{relevance:.3f}`")
            with col_b3:
                time_range = f"{item.get('start_time_formatted', '00:00')} - {item.get('end_time_formatted', '00:00')}"
                st.markdown(f"⏱️ **Range:** `{time_range}`")

            st.markdown("---")

            # Relevant transcript snippet
            topic_text = item.get("topic_text", "").strip()
            if topic_text:
                st.markdown("**🎯 Relevant Lecture Content:**")
                st.info(f'"{topic_text}"')
            else:
                st.caption("No transcript text snippet available.")
