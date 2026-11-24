from datetime import datetime
from pathlib import Path
import base64
import html

import streamlit as st

from utils.helper import generate_greeting

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "logo.png"


@st.cache_data
def _get_logo_base64(path: str, modified_at: float) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


LOGO_B64 = _get_logo_base64(str(LOGO_PATH), LOGO_PATH.stat().st_mtime)


def configure_page():
    st.set_page_config(page_title="Magnus AI", page_icon=str(LOGO_PATH))
    st.logo(str(LOGO_PATH), icon_image=str(LOGO_PATH), size="large")

    # Streamlit caps the built-in sidebar logo and chat avatars at a small size, so scale them up
    st.markdown(
        """
        <style>
            [data-testid="stLogo"] { height: 2rem; }
            [data-testid="stChatMessageAvatarCustom"] {
                width: 2rem;
                height: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.75rem;">
            <img src="data:image/png;base64,{LOGO_B64}" width="48" height="48" alt="Magnus logo" />
            <h1 style="margin: 0; padding: 0; font-size: 2.5rem; font-weight: 700;">Magnus AI</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_time_based_greeting():
    current_time = datetime.now().strftime("%I:%M %p, %A")
    return generate_greeting(current_time)


def render_greeting_screen():
    """Draw the empty-chat welcome panel and return its placeholder so it can be cleared."""
    greeting_placeholder = st.empty()

    if st.session_state["message_history"]:
        return greeting_placeholder

    thread_id = st.session_state["thread_id"]
    if thread_id not in st.session_state["thread_greetings"]:
        st.session_state["thread_greetings"][thread_id] = get_time_based_greeting()

    greeting = html.escape(st.session_state["thread_greetings"][thread_id])

    with greeting_placeholder.container():
        st.markdown(
            f"""
            <div style="text-align: center; padding: 12vh 1rem 4vh 1rem;">
                <img src="data:image/png;base64,{LOGO_B64}" width="96" height="96" alt="Magnus logo" style="margin-bottom: 1rem;" />
                <h2 style="font-weight: 500; margin-bottom: 0.4rem;">{greeting}</h2>
                <p style="opacity: 0.65; font-size: 1rem;">How can I help you today?</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return greeting_placeholder


def render_message_history():
    for message in st.session_state["message_history"]:
        avatar = str(LOGO_PATH) if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
