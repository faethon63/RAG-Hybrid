"""
Chat Components
Message display, input handling, and history management.
"""

import streamlit as st
from typing import Dict, List, Optional


def render_chat_message(role: str, content: str, metadata: Optional[Dict] = None):
    """Render a single chat message."""
    with st.chat_message(role):
        st.markdown(content)
        if metadata:
            if metadata.get("mode"):
                st.caption(f"Mode: {metadata['mode']}")
            if metadata.get("processing_time"):
                st.caption(f"Time: {metadata['processing_time']:.2f}s")
            if metadata.get("confidence") is not None:
                confidence = metadata["confidence"]
                bar_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                st.progress(confidence, text=f"Confidence: {confidence:.0%}")


def render_chat_input() -> Optional[str]:
    """Render the chat input box and return the user's message."""
    return st.chat_input("Ask anything...")


def render_chat_history(messages: List[Dict]):
    """Render the full chat history."""
    for msg in messages:
        render_chat_message(
            role=msg["role"],
            content=msg["content"],
            metadata=msg.get("metadata"),
        )
