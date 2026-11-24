import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agentic_chatbot_backend import chatbot_workflow
from db.database import delete_thread, get_all_threads, record_thread_created
from utils.helper import generate_title


def generate_thread_id():
    """Generate a unique thread ID for each chat session."""
    return str(uuid.uuid4())


def init_session_state():
    defaults = {
        "message_history": [],
        "thread_id": generate_thread_id(),
        "chat_threads": None,
        "thread_titles": {},
        "thread_greetings": {},
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            # Reading the thread list hits the database, so only do it on first load
            st.session_state[key] = get_all_threads() if key == "chat_threads" else default


def add_thread(thread_id):
    """Add a thread to the conversation list if it isn't there already."""
    record_thread_created(thread_id)
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    """Create a new chat thread only if the current thread has messages."""
    if not st.session_state["message_history"]:
        return
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []


def load_conversation(thread_id):
    """Load a previous conversation from the langgraph checkpointer."""
    state = chatbot_workflow.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    return state.values.get("messages", [])


def get_thread_title(thread_id):
    title = st.session_state["thread_titles"].get(thread_id)
    if title:
        return title

    messages = load_conversation(thread_id)
    first_human = next(
        (m.text for m in messages if isinstance(m, HumanMessage) and m.text), None
    )
    if first_human:
        first_ai = next(
            (m.text for m in messages if isinstance(m, AIMessage) and m.text), ""
        )
        title = generate_title(first_human, first_ai)
        st.session_state["thread_titles"][thread_id] = title
        return title

    st.session_state["thread_titles"][thread_id] = "New Chat"
    return "New Chat"


@st.dialog("Delete conversation?")
def confirm_delete(thread_id):
    st.write(f"**{get_thread_title(thread_id)}** will be removed from your conversations.")

    cancel_column, delete_column = st.columns(2)

    # Closing happens by rerunning without calling this dialog function again
    if cancel_column.button("Cancel", key="cancel_delete", width="stretch"):
        st.rerun()

    if delete_column.button(
        "Delete", key="confirm_delete", type="primary", width="stretch"
    ):
        delete_thread(thread_id)

        st.session_state["chat_threads"] = [
            tid for tid in st.session_state["chat_threads"] if tid != thread_id
        ]
        st.session_state["thread_titles"].pop(thread_id, None)
        st.session_state["thread_greetings"].pop(thread_id, None)

        # Deleting the open conversation leaves the user on a fresh empty chat
        if st.session_state["thread_id"] == thread_id:
            st.session_state["thread_id"] = generate_thread_id()
            st.session_state["message_history"] = []

        st.rerun()


def _select_thread(thread_id):
    st.session_state["thread_id"] = thread_id

    history = []
    for message in load_conversation(thread_id):
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        history.append({"role": role, "content": message.text})

    st.session_state["message_history"] = history


def render_sidebar():
    st.sidebar.title("My Conversations")

    if st.sidebar.button("New Chat", width="stretch"):
        if st.session_state["message_history"]:
            reset_chat()
            st.rerun()

    # Keep only threads that actually have messages so no empty duplicate threads appear
    st.session_state["chat_threads"] = [
        tid for tid in st.session_state["chat_threads"] if load_conversation(tid)
    ]

    for thread_id in st.session_state["chat_threads"][::-1]:
        is_active = thread_id == st.session_state["thread_id"]

        select_column, delete_column = st.sidebar.columns(
            [0.8, 0.2], gap="small", vertical_alignment="center"
        )

        if select_column.button(
            get_thread_title(thread_id),
            key=thread_id,
            type="primary" if is_active else "secondary",
            width="stretch",
        ):
            _select_thread(thread_id)
            st.rerun()

        # Opens the confirmation modal, which does the actual deleting
        if delete_column.button(
            "",
            key=f"delete-{thread_id}",
            icon=":material/delete:",
            help="Delete this conversation",
            width="stretch",
        ):
            confirm_delete(thread_id)
