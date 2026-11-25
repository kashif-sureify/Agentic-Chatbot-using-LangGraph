import re
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentic_chatbot_backend import chatbot_workflow
from db.database import delete_thread, get_all_threads, record_thread_created
from ui import components
from ui.theme import render_theme_switch
from utils.helper import generate_title, has_rag_documents

# Tools that pause for the user's approval; their result says whether it was given
APPROVAL_TOOLS = {"purchase_stock", "get_current_location"}

# The open conversation is kept in the URL so a refresh, a shared link or a theme
# change (which reloads the page) comes back to the same chat
THREAD_QUERY_PARAM = "c"


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
        # PDF names per thread, for the header; filled from uploads and from saved
        # `rag_tool` results
        "thread_docs": {},
    }

    first_load = "chat_threads" not in st.session_state

    for key, default in defaults.items():
        if key not in st.session_state:
            # Reading the thread list hits the database, so only do it on first load
            st.session_state[key] = get_all_threads() if key == "chat_threads" else default

    if first_load:
        linked = st.query_params.get(THREAD_QUERY_PARAM)
        if linked and linked in st.session_state["chat_threads"]:
            _select_thread(linked)


def sync_thread_url():
    """Point the URL at the open conversation, or clear it for a new, unsaved chat."""
    thread_id = st.session_state["thread_id"]
    saved = thread_id in st.session_state["chat_threads"]

    if saved:
        # Set on every run, even when unchanged: an update made just before st.rerun()
        # can be dropped, and the server would otherwise think the URL is already right
        st.query_params[THREAD_QUERY_PARAM] = thread_id
    elif THREAD_QUERY_PARAM in st.query_params:
        del st.query_params[THREAD_QUERY_PARAM]


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


def remember_docs(thread_id, names):
    docs = st.session_state["thread_docs"].setdefault(thread_id, [])
    for name in names:
        if name not in docs:
            docs.append(name)


def get_thread_docs(thread_id):
    """Names of the PDFs this conversation can answer from, for the header."""
    docs = st.session_state["thread_docs"].get(thread_id, [])
    if docs:
        return docs
    # An index exists but no name has been seen yet (nothing asked about it so far)
    return ["PDF attached"] if has_rag_documents(thread_id) else []


def _was_declined(tool_output):
    text = tool_output.lower()
    return '"status": "rejected"' in text or "rejected by the user" in text


def _history_from_checkpoint(thread_id, messages, paused):
    """Rebuild the chat as the UI shows it: one assistant entry per user turn.

    Tool calls and results are folded into that entry's tool panel, and an approval
    tool's result tells whether the user approved it. If the graph is paused, the last
    turn is marked as waiting so the approval card can attach to it.
    """
    history = []
    turn = None

    def close_turn():
        if turn and (turn["texts"] or turn["tools"]):
            entry = {
                "role": "assistant",
                "content": "\n\n".join(turn["texts"]),
                "tools": turn["tools"],
                "tool_state": "done",
            }
            if turn["decision"]:
                entry["decision"] = turn["decision"]
            history.append(entry)

    for message in messages:
        if isinstance(message, HumanMessage):
            close_turn()
            history.append({"role": "user", "content": message.text})
            turn = {"texts": [], "tools": [], "decision": None}
        elif turn is None:
            continue
        elif isinstance(message, ToolMessage):
            output = message.text
            name = message.name or "tool"
            turn["tools"].append({"name": name, "result": components.result_preview(output)})
            if name in APPROVAL_TOOLS:
                turn["decision"] = {"tool": name, "approved": not _was_declined(output)}
            if name == "rag_tool":
                remember_docs(thread_id, re.findall(r"Source:\s*([^\n]+?)\s*(?:\n|$)", output))
        elif isinstance(message, AIMessage) and message.text:
            turn["texts"].append(message.text)

    if paused and turn is not None:
        # The paused turn has no answer yet, only the tool call waiting on the user
        history.append(
            {"role": "assistant", "content": "", "tools": turn["tools"], "tool_state": "awaiting"}
        )
    else:
        close_turn()

    return history


def _select_thread(thread_id):
    st.session_state["thread_id"] = thread_id

    snapshot = chatbot_workflow.get_state(config={"configurable": {"thread_id": thread_id}})
    st.session_state["message_history"] = _history_from_checkpoint(
        thread_id, snapshot.values.get("messages", []), paused=bool(snapshot.interrupts)
    )


@st.dialog("Delete conversation?", icon=":material/delete:")
def confirm_delete(thread_id):
    st.markdown(
        f"**{get_thread_title(thread_id)}** will be removed from your conversations."
    )

    with st.container(horizontal=True, gap="small", key="delete_actions"):
        # Closing happens by rerunning without calling this dialog function again
        if st.button("Cancel", key="cancel_delete", width="stretch"):
            st.rerun()

        if st.button(
            "Delete",
            key="confirm_delete",
            type="primary",
            icon=":material/delete:",
            width="stretch",
        ):
            delete_thread(thread_id)

            st.session_state["chat_threads"] = [
                tid for tid in st.session_state["chat_threads"] if tid != thread_id
            ]
            st.session_state["thread_titles"].pop(thread_id, None)
            st.session_state["thread_greetings"].pop(thread_id, None)
            st.session_state["thread_docs"].pop(thread_id, None)

            # Deleting the open conversation leaves the user on a fresh empty chat
            if st.session_state["thread_id"] == thread_id:
                st.session_state["thread_id"] = generate_thread_id()
                st.session_state["message_history"] = []

            st.rerun()


def render_sidebar():
    with st.sidebar:
        if st.button("New chat", key="new_chat", icon=":material/add:", width="stretch"):
            if st.session_state["message_history"]:
                reset_chat()
                st.rerun()

        query = st.text_input(
            "Search conversations",
            key="thread_search",
            placeholder="Search conversations",
            label_visibility="collapsed",
            icon=":material/search:",
            live=True,
        )
        query = (query or "").strip()

        # Keep only threads that actually have messages so no empty duplicate threads appear
        st.session_state["chat_threads"] = [
            tid for tid in st.session_state["chat_threads"] if load_conversation(tid)
        ]

        threads = st.session_state["chat_threads"][::-1]
        if query:
            threads = [tid for tid in threads if query.lower() in get_thread_title(tid).lower()]

        st.markdown(components.overline("Recent"), unsafe_allow_html=True)

        if not threads:
            st.markdown(components.empty_threads(query), unsafe_allow_html=True)

        # Not "mg-thread-…": that prefix is how the CSS finds a single row, and the list
        # sharing it would make hovering anywhere highlight every row
        with st.container(key="mg-conversations", gap=None):
            for thread_id in threads:
                is_active = thread_id == st.session_state["thread_id"]
                row_key = f"mg-thread-{'active-' if is_active else ''}{thread_id}"

                with st.container(
                    horizontal=True, key=row_key, gap=None, vertical_alignment="center"
                ):
                    if st.button(
                        get_thread_title(thread_id),
                        key=thread_id,
                        type="tertiary",
                        width="stretch",
                    ):
                        _select_thread(thread_id)
                        st.rerun()

                    # Opens the confirmation modal, which does the actual deleting
                    if st.button(
                        "",
                        key=f"delete-{thread_id}",
                        icon=":material/delete:",
                        type="tertiary",
                        help="Delete this conversation",
                    ):
                        confirm_delete(thread_id)

        with st.container(key="sidebar_footer", gap="small"):
            st.markdown('<div class="mg-foot-label">Theme</div>', unsafe_allow_html=True)
            render_theme_switch()
            st.markdown(
                '<p class="mg-fine">Magnus can make mistakes. Check important info.</p>',
                unsafe_allow_html=True,
            )
