"""
Search Components
Source cards and citation display.
"""

import streamlit as st
from typing import List, Dict, Optional


def render_source_card(source: Dict):
    """Render a single source as an expandable card."""
    source_type = source.get("type", "unknown")
    title = source.get("title", "Untitled")
    url = source.get("url")
    snippet = source.get("snippet", "")
    score = source.get("score")

    # Icon by source type
    icons = {
        "local_doc": "local",
        "web": "web",
        "research": "research",
    }
    icon = icons.get(source_type, "source")
    label = f"[{icon}] {title}"
    if score is not None:
        label += f" ({score:.0%})"

    with st.expander(label, expanded=False):
        if snippet:
            st.markdown(snippet)
        if url:
            st.markdown(f"[Open source]({url})")
        if score is not None:
            st.progress(min(score, 1.0))


def render_sources_panel(sources: List[Dict]):
    """Render all sources in a sidebar or panel."""
    if not sources:
        st.info("No sources for this query.")
        return

    st.subheader(f"Sources ({len(sources)})")
    for source in sources:
        render_source_card(source)
