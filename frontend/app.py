"""
RAG-Hybrid Streamlit Frontend
Chat interface, mode selector, document upload, project switcher, system status.
"""

import sys
import os
import time

import streamlit as st
import httpx

# Allow importing components from this directory
sys.path.insert(0, os.path.dirname(__file__))
from components.chat import render_chat_history, render_chat_input, render_chat_message
from components.search import render_sources_panel
from components.projects import render_project_selector, render_document_manager

# --- Page config ---

st.set_page_config(
    page_title="RAG-Hybrid",
    page_icon="magnifying_glass",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session state defaults ---

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "api_url" not in st.session_state:
    st.session_state["api_url"] = os.getenv("API_URL", "http://localhost:8000")
if "auth_token" not in st.session_state:
    st.session_state["auth_token"] = ""
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "current_project" not in st.session_state:
    st.session_state["current_project"] = None
if "last_sources" not in st.session_state:
    st.session_state["last_sources"] = []

API_URL = st.session_state["api_url"]


def do_login(username: str, password: str) -> bool:
    """Authenticate and store token."""
    try:
        resp = httpx.post(
            f"{API_URL}/api/v1/login",
            json={"username": username, "password": password},
            timeout=10.0,
        )
        if resp.status_code == 200:
            st.session_state["auth_token"] = resp.json()["token"]
            st.session_state["logged_in"] = True
            return True
    except Exception:
        pass
    return False


# --- Auto-login or show login form ---

if not st.session_state["logged_in"]:
    st.title("RAG-Hybrid Login")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password", value="admin")
            submitted = st.form_submit_button("Login")

            if submitted:
                if do_login(username, password):
                    st.success("Logged in!")
                    st.rerun()
                else:
                    st.error("Login failed. Check credentials.")

        st.caption("Default: admin / admin")
    st.stop()


# --- Sidebar ---

st.sidebar.title("RAG-Hybrid")

# Mode selector
mode = st.sidebar.radio(
    "Search mode",
    ["hybrid", "local", "web", "research"],
    index=0,
    help=(
        "hybrid: Local + Web combined (default)\n"
        "local: Private docs only (ChromaDB + Ollama)\n"
        "web: Claude API\n"
        "research: Perplexity deep search"
    ),
)

# Auth token input
with st.sidebar.expander("Settings"):
    token_input = st.text_input("API Token", value=st.session_state["auth_token"], type="password")
    if token_input != st.session_state["auth_token"]:
        st.session_state["auth_token"] = token_input

    api_input = st.text_input("Backend URL", value=API_URL)
    if api_input != API_URL:
        st.session_state["api_url"] = api_input
        API_URL = api_input

st.sidebar.divider()

# Project selector & document manager
selected_project = render_project_selector()
st.session_state["current_project"] = selected_project

st.sidebar.divider()
render_document_manager()

# System status
st.sidebar.divider()
if st.sidebar.button("Check system health"):
    try:
        resp = httpx.get(f"{API_URL}/api/v1/health", timeout=5.0)
        if resp.status_code == 200:
            health = resp.json()
            st.sidebar.success(f"Status: {health['status']}")
            for svc, ok in health.get("services", {}).items():
                icon = "+" if ok else "x"
                st.sidebar.text(f"  [{icon}] {svc}")
        else:
            st.sidebar.error(f"Health check returned {resp.status_code}")
    except Exception as e:
        st.sidebar.error(f"Cannot reach backend: {e}")


# --- Main area ---

st.title("RAG-Hybrid Search")
st.caption(f"Mode: **{mode}** | Project: **{selected_project or 'all'}**")

# Render chat history
render_chat_history(st.session_state["messages"])

# Chat input
user_input = render_chat_input()

if user_input:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    render_chat_message("user", user_input)

    # Query the backend
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/api/v1/query",
                    headers={
                        "Authorization": f"Bearer {st.session_state['auth_token']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": user_input,
                        "mode": mode,
                        "project": selected_project,
                        "max_results": 5,
                        "include_sources": True,
                    },
                    timeout=120.0,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer received.")
                    sources = data.get("sources", [])
                    processing_time = data.get("processing_time", 0)
                    confidence = data.get("confidence")

                    st.markdown(answer)
                    if processing_time:
                        st.caption(f"Time: {processing_time:.2f}s")
                    if confidence is not None:
                        st.progress(confidence, text=f"Confidence: {confidence:.0%}")

                    # Store for sidebar display
                    st.session_state["last_sources"] = sources

                    # Add assistant message to history
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "mode": mode,
                            "processing_time": processing_time,
                            "confidence": confidence,
                        },
                    })

                elif resp.status_code == 401:
                    st.error("Authentication failed. Check your API token in Settings.")
                elif resp.status_code == 429:
                    st.warning("Rate limit exceeded. Please wait a moment.")
                else:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")

            except httpx.ConnectError:
                st.error(
                    "Cannot connect to backend. "
                    f"Is it running at {API_URL}?"
                )
            except Exception as e:
                st.error(f"Error: {e}")

# Sources panel (right column)
if st.session_state["last_sources"]:
    with st.sidebar:
        st.divider()
        render_sources_panel(st.session_state["last_sources"])

# Clear chat button
if st.session_state["messages"]:
    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.session_state["last_sources"] = []
        st.rerun()
