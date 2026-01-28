"""
Project Components
Project selector and document manager.
"""

import streamlit as st
import httpx
from typing import Optional, Dict, List


def _api_url() -> str:
    """Get backend API URL from session state."""
    return st.session_state.get("api_url", "http://localhost:8000")


def _auth_headers() -> Dict[str, str]:
    """Get auth headers from session state."""
    token = st.session_state.get("auth_token", "")
    return {"Authorization": f"Bearer {token}"}


def render_project_selector() -> Optional[str]:
    """Render project selector dropdown. Returns selected project name."""
    st.sidebar.subheader("Project")

    # Fetch projects from API
    projects = st.session_state.get("projects", [])

    if not projects:
        try:
            resp = httpx.get(
                f"{_api_url()}/api/v1/projects",
                headers=_auth_headers(),
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                projects = data.get("projects", [])
                st.session_state["projects"] = projects
        except Exception:
            pass

    project_names = ["(all)"] + [p.get("name", "?") for p in projects]
    selected = st.sidebar.selectbox("Active project", project_names, index=0)

    # Show doc count for selected project
    if selected != "(all)":
        for p in projects:
            if p.get("name") == selected:
                st.sidebar.caption(f"Documents: {p.get('document_count', 0)}")
                break

    return None if selected == "(all)" else selected


def render_document_manager():
    """Render document upload and management UI."""
    st.sidebar.subheader("Documents")

    uploaded = st.sidebar.file_uploader(
        "Upload document",
        type=["txt", "md", "pdf", "json"],
        accept_multiple_files=True,
    )

    if uploaded and st.sidebar.button("Index uploaded files"):
        documents = []
        for f in uploaded:
            content = f.read().decode("utf-8", errors="ignore")
            documents.append({
                "content": content,
                "title": f.name,
                "path": f.name,
            })

        if documents:
            project = st.session_state.get("current_project")
            try:
                resp = httpx.post(
                    f"{_api_url()}/api/v1/index",
                    headers={
                        **_auth_headers(),
                        "Content-Type": "application/json",
                    },
                    json={"documents": documents, "project": project},
                    timeout=60.0,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.sidebar.success(f"Indexed {result.get('indexed_count', 0)} chunks")
                    # Refresh project list
                    st.session_state.pop("projects", None)
                else:
                    st.sidebar.error(f"Indexing failed: {resp.status_code}")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    # Re-index button
    if st.sidebar.button("Refresh project list"):
        st.session_state.pop("projects", None)
        st.rerun()
