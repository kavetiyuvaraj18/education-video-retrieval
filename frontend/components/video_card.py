"""Streamlit UI component for timestamped educational video results."""

import html
from typing import Any, Dict

import streamlit as st


def _text(value: Any, fallback: str = "") -> str:
    return html.escape(str(value if value is not None else fallback), quote=True)


def _duration_label(start: Any, end: Any) -> str:
    try:
        seconds = max(0, round(float(end) - float(start)))
    except (TypeError, ValueError):
        return ""
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}:{seconds:02d} segment"


def render_video_card(item: Dict[str, Any], rank_index: int) -> None:
    """Render a lecture result with its transcript-matched time and jump link."""
    title = _text(item.get("title"), "Lecture video")
    channel = _text(item.get("channel_name"), "YouTube")
    thumbnail_url = _text(item.get("thumbnail_url"))
    timestamp_url = _text(item.get("timestamp_url"))
    start_time = _text(item.get("start_time_formatted"), "00:00")
    end_time = _text(item.get("end_time_formatted"), start_time)
    duration = _text(
        _duration_label(
            item.get("start_time_seconds", 0),
            item.get("end_time_seconds", 0),
        )
    )
    excerpt = _text(item.get("topic_text"), "Transcript excerpt unavailable.")

    image = (
        f'<img class="lecture-thumb" src="{thumbnail_url}" alt="Thumbnail for {title}" loading="lazy">'
        if thumbnail_url
        else '<div class="lecture-thumb thumb-placeholder" aria-hidden="true">▶</div>'
    )
    jump_link = (
        f'<a class="jump-link" href="{timestamp_url}" target="_blank" '
        f'rel="noopener noreferrer" aria-label="Jump to {start_time} in {title}">'
        f'▶&nbsp; Jump to topic <span aria-hidden="true">↗</span></a>'
        if timestamp_url
        else '<span class="no-link">Timestamp link unavailable</span>'
    )

    st.markdown(
        f"""
        <article class="lecture-card" aria-label="Result {rank_index}: {title}">
            <div class="lecture-media">
                {image}
                <div class="time-badge"><span aria-hidden="true">◷</span>
                    {start_time} – {end_time}</div>
            </div>
            <div class="lecture-body">
                <div class="lecture-meta">
                    <span class="rank-badge">#{rank_index}</span>
                    <span class="channel-name">{channel}</span>
                </div>
                <h3 class="lecture-title">{title}</h3>
                <div class="segment-duration">Matched portion · {duration}</div>
                <div class="excerpt-label">TRANSCRIPT MOMENT</div>
                <blockquote class="transcript-excerpt">“{excerpt}”</blockquote>
                {jump_link}
            </div>
        </article>
        <style>
        .lecture-card {{
            display: grid; grid-template-columns: minmax(230px, 34%) 1fr;
            gap: 1.4rem; margin: 1rem 0 1.2rem; padding: 1rem;
            overflow: hidden; border: 1px solid rgba(166, 183, 220, .15);
            border-radius: 1.2rem; background: linear-gradient(135deg,
                rgba(19, 29, 51, .97), rgba(13, 21, 39, .98));
            box-shadow: 0 14px 35px rgba(0, 0, 0, .13);
        }}
        .lecture-media {{ position: relative; min-height: 190px; }}
        .lecture-thumb {{
            display: block; width: 100%; height: 100%; min-height: 190px;
            object-fit: cover; border-radius: .85rem; background: #18233a;
        }}
        .thumb-placeholder {{
            display: grid; place-items: center; color: #c4b5fd; font-size: 2rem;
        }}
        .time-badge {{
            position: absolute; left: .65rem; bottom: .65rem;
            padding: .35rem .55rem; border: 1px solid rgba(255,255,255,.16);
            border-radius: .6rem; background: rgba(5, 10, 22, .86);
            color: #f3f5ff; font-size: .83rem; font-weight: 650;
        }}
        .lecture-body {{ display: flex; flex-direction: column; align-items: flex-start; }}
        .lecture-meta {{ display: flex; align-items: center; gap: .65rem; }}
        .rank-badge {{
            padding: .22rem .52rem; border-radius: 999px; color: #d8ccff;
            background: rgba(139, 92, 246, .17); font-size: .75rem; font-weight: 700;
        }}
        .channel-name {{ color: #aebbd3; font-size: .87rem; }}
        .lecture-title {{
            margin: .65rem 0 .35rem; color: #f3f5ff;
            font-size: 1.22rem; line-height: 1.35; font-weight: 700;
        }}
        .segment-duration {{ color: #aebbd3; font-size: .84rem; }}
        .excerpt-label {{
            margin-top: .95rem; color: #b9a6ff; font-size: .68rem;
            font-weight: 750; letter-spacing: .12em;
        }}
        .transcript-excerpt {{
            margin: .35rem 0 .9rem; padding: 0 0 0 .8rem;
            border-left: 2px solid rgba(147, 130, 255, .7);
            color: #d6def0; font-size: .93rem; line-height: 1.55;
        }}
        .jump-link {{
            display: inline-flex; align-items: center; gap: .45rem;
            margin-top: auto; padding: .68rem .9rem; border-radius: .75rem;
            color: #fff !important; background: linear-gradient(100deg, #7c3aed, #2563eb);
            font-size: .88rem; font-weight: 700; text-decoration: none !important;
            transition: filter .18s ease, transform .18s ease;
        }}
        .jump-link:hover {{ filter: brightness(1.13); transform: translateY(-1px); }}
        .no-link {{ color: #aebbd3; font-size: .85rem; }}
        @media (max-width: 700px) {{
            .lecture-card {{ grid-template-columns: 1fr; gap: .9rem; padding: .75rem; }}
            .lecture-media, .lecture-thumb {{ min-height: 180px; max-height: 250px; }}
            .lecture-title {{ font-size: 1.1rem; }}
            .jump-link {{ min-height: 44px; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .jump-link {{ transition: none; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
