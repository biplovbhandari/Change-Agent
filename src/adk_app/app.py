import os
import sys
from pathlib import Path
from typing import Dict, Any

import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent       # .../src/adk_app
SRC_DIR = CURRENT_DIR.parent                         # .../src
sys.path.append(str(CURRENT_DIR))                    # for agent.py
sys.path.append(str(SRC_DIR))                        # (predict.py lives in src/)

from agent import ADKChangeAgent

PAGE_TITLE = "Change Detection Agent"
PAGE_ICON = "🔍"


# ---------------------------
# UI helpers
# ---------------------------
def render_action(action: Dict[str, Any]):
    with st.expander(action["type"], expanded=True):
        st.markdown(f"**Tool:** `{action['type']}`")
        st.markdown(f"**Thought:** {action.get('thought', '')}")

        args = action.get("args") or {}
        if args:
            st.markdown("**Arguments:**")
            for k, v in args.items():
                st.markdown(f"- `{k}`: `{v}`")

        result = action.get("result") or {}
        if result:
            st.markdown("**Result:**")

            # free-form text (from our tool)
            if "text" in result:
                st.markdown(result["text"])

            # show saved mask image if present
            mask_path = result.get("mask_path") or result.get("saved_to")
            if mask_path and os.path.exists(mask_path):
                st.image(mask_path, caption=Path(mask_path).name, width="content")

            # metrics if available
            stats = result.get("statistics")
            if stats:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Roads Changed", stats.get("roads", 0))
                with c2:
                    st.metric("Buildings Changed", stats.get("buildings", 0))


def render_assistant(agent_return: Dict[str, Any]):
    with st.chat_message("assistant"):
        for action in agent_return.get("actions", []):
            if action:
                render_action(action)
        st.markdown(agent_return.get("response", ""))


def save_uploaded(file, dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / file.name
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return str(path)


# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.set_page_config(layout="wide", page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.header(f"{PAGE_ICON} :blue[{PAGE_TITLE}]", divider="rainbow")

    # --- Sidebar ---
    st.sidebar.title("Configuration")

    # Model swap (Gemini family)
    default_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    model_options = os.environ.get("GEMINI_MODEL_OPTIONS", "gemini-2.5-flash,gemini-2.5-pro").split(",")
    if default_model not in model_options:
        model_options.insert(0, default_model)
    model_name = st.sidebar.selectbox(
        "LLM Model",
        options=model_options,
        index=model_options.index(default_model),
        help="Google ADK model name",
    )

    if st.sidebar.button("Clear conversation"):
        for k in ["assistant", "user", "adk_agent"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    uploaded_a = st.sidebar.file_uploader("Upload Image A", type=["png", "jpg", "jpeg"])
    uploaded_b = st.sidebar.file_uploader("Upload Image B", type=["png", "jpg", "jpeg"])

    # --- Initialize agent once ---
    if "adk_agent" not in st.session_state:
        with st.spinner("Loading ADK agent..."):
            st.session_state["adk_agent"] = ADKChangeAgent(
                model_name=model_name,
            )
            st.session_state["adk_agent"].ensure_session()
        st.session_state["assistant"] = []
        st.session_state["user"] = []

    agent: ADKChangeAgent = st.session_state["adk_agent"]

    # --- If the user uploaded images, store + register them with the agent ---
    context_prefix = ""
    if uploaded_a and uploaded_b:
        tmp_dir = Path("tmp_uploads")
        file_a = save_uploaded(uploaded_a, tmp_dir)
        file_b = save_uploaded(uploaded_b, tmp_dir)

        # Show the images
        c1, c2 = st.columns(2)
        with c1:
            st.image(file_a, caption="Image A", width="stretch")
        with c2:
            st.image(file_b, caption="Image B", width="stretch")

        # Register with ADK agent so follow-ups can omit paths
        agent.register_images(file_a, file_b)

        # Also pass paths in the message once; the agent will call set_image_paths
        context_prefix = f"Image A path: {file_a}\nImage B path: {file_b}\n\n"

    # --- Replay history ---
    for u, a in zip(st.session_state["user"], st.session_state["assistant"]):
        with st.chat_message("user"):
            st.markdown(u)
        render_assistant(a)

    # --- Chat box ---
    if user_msg := st.chat_input("Ask about the changes, counts, or request saving the road map…"):
        with st.chat_message("user"):
            st.markdown(user_msg)
        st.session_state["user"].append(user_msg)

        # Prepend context with paths (if any)
        full_msg = f"{context_prefix}{user_msg}"

        with st.spinner("Thinking…"):
            agent_return = agent.chat(full_msg)

        st.session_state["assistant"].append(agent_return)
        render_assistant(agent_return)


if __name__ == "__main__":
    main()
