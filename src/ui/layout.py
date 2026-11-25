from datetime import datetime
from pathlib import Path
import base64

import streamlit as st

from ui import components
from ui.theme import inject_styles
from ui.tool_panel import render_tool_panel
from utils.helper import generate_greeting

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "logo.png"
ASSISTANT_AVATAR = str(LOGO_PATH)


@st.cache_data
def _get_logo_base64(path: str, modified_at: float) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


LOGO_B64 = _get_logo_base64(str(LOGO_PATH), LOGO_PATH.stat().st_mtime)

# Starter prompts on the empty chat. A card with no prompt cannot act on its own
# (a file can only be attached by the user), so it shows its hint as a toast instead.
SUGGESTIONS = [
    {
        "key": "weather",
        "icon": "cloud-sun",
        "tone": "violet",
        "title": "Check the weather",
        "text": "What's the weather in London right now?",
        "prompt": "What's the weather in London right now?",
    },
    {
        "key": "stock",
        "icon": "trending-up",
        "tone": "violet",
        "title": "Look up a stock",
        "text": "What's Apple trading at today?",
        "prompt": "What's Apple trading at today?",
    },
    {
        "key": "pdf",
        "icon": "file-text",
        "tone": "coral",
        "title": "Ask about a PDF",
        "text": "Attach a paper and ask for its key findings.",
        "hint": "Attach a PDF with the paperclip in the message box, then ask your question.",
    },
    {
        "key": "trade",
        "icon": "shield-alert",
        "tone": "amber",
        "title": "Place a practice trade",
        "text": "Buy 5 shares of NVDA. Magnus asks before it buys.",
        "prompt": "Buy 5 shares of NVDA",
    },
]


def configure_page():
    st.set_page_config(
        page_title="Magnus AI",
        page_icon=str(LOGO_PATH),
        layout="centered",
        # Open on desktop, collapsed on phones where it would cover the chat
        initial_sidebar_state="auto",
    )
    st.logo(str(LOGO_PATH), icon_image=str(LOGO_PATH), size="large")
    inject_styles()


def render_header(title, docs):
    """Conversation title, plus the PDFs this chat can answer from."""
    st.markdown(components.topbar(title, docs), unsafe_allow_html=True)


def get_time_based_greeting():
    current_time = datetime.now().strftime("%I:%M %p, %A")
    return generate_greeting(current_time)


def render_greeting_screen():
    """Draw the empty-chat welcome panel.

    Returns its placeholder, so it can be cleared once a turn starts, and the prompt of
    the suggestion card the user clicked on this run (or None).
    """
    greeting_placeholder = st.empty()

    if st.session_state["message_history"]:
        return greeting_placeholder, None

    thread_id = st.session_state["thread_id"]
    if thread_id not in st.session_state["thread_greetings"]:
        st.session_state["thread_greetings"][thread_id] = get_time_based_greeting()

    chosen_prompt = None

    with greeting_placeholder.container(key="mg-welcome"):
        st.markdown(
            components.welcome(st.session_state["thread_greetings"][thread_id], LOGO_B64),
            unsafe_allow_html=True,
        )

        for row in (SUGGESTIONS[:2], SUGGESTIONS[2:]):
            for column, suggestion in zip(st.columns(2, gap="small"), row):
                with column, st.container(key=f"mg-sugcard-{suggestion['key']}"):
                    st.markdown(
                        components.suggestion_card(
                            suggestion["icon"],
                            suggestion["tone"],
                            suggestion["title"],
                            suggestion["text"],
                        ),
                        unsafe_allow_html=True,
                    )
                    # Invisible button stretched over the card, so the whole card is clickable
                    if st.button(suggestion["title"], key=f"mg-sugbtn-{suggestion['key']}"):
                        if suggestion.get("prompt"):
                            chosen_prompt = suggestion["prompt"]
                        else:
                            st.toast(suggestion["hint"], icon=":material/attach_file:")

    return greeting_placeholder, chosen_prompt


def render_user_message(entry):
    with st.chat_message("user"):
        if entry.get("files"):
            st.markdown(components.file_chips(entry["files"]), unsafe_allow_html=True)
        if entry.get("content"):
            st.markdown(entry["content"])


def render_assistant_body(entry, key):
    """The parts of a saved assistant reply, in the order they appeared while live."""
    if entry.get("decision"):
        decision = entry["decision"]
        st.markdown(
            components.decision_chip(decision["tool"], decision["approved"]),
            unsafe_allow_html=True,
        )

    if entry.get("indexed"):
        st.markdown(components.file_chips(entry["indexed"], "indexed"), unsafe_allow_html=True)

    steps = entry.get("tools") or []
    state = entry.get("tool_state", "done")
    if steps or state == "awaiting":
        render_tool_panel(steps, state, key)

    if entry.get("content"):
        st.markdown(entry["content"])


def render_message_history(history):
    for index, entry in enumerate(history):
        if entry["role"] == "user":
            render_user_message(entry)
        else:
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                render_assistant_body(entry, f"h{index}")
