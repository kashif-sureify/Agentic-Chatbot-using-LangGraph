import streamlit as st

from ui import components
from ui.approval import APPROVE, get_pending_approval, render_approval_card
from ui.documents import ingest_uploaded_pdfs
from ui.layout import (
    ASSISTANT_AVATAR,
    configure_page,
    render_greeting_screen,
    render_header,
    render_message_history,
    render_user_message,
)
from ui.streaming import resume_assistant_response, stream_assistant_response
from ui.threads import (
    add_thread,
    get_thread_docs,
    get_thread_title,
    init_session_state,
    remember_docs,
    render_sidebar,
    sync_thread_url,
)
from ui.tool_panel import render_tool_panel
from utils.helper import generate_title


def finish_turn(thread_id, user_input=None, assistant_response="", after_decision=False):
    """Record the thread, then redraw when the sidebar, input or an approval card must change."""
    add_thread(thread_id)
    sync_thread_url()

    titles = st.session_state["thread_titles"]
    needs_title = (
        user_input and assistant_response and titles.get(thread_id) in (None, "New Chat")
    )
    if needs_title:
        titles[thread_id] = generate_title(user_input, assistant_response)

    # A paused tool leaves nothing on screen, so redraw to put the approval card in view.
    # After a decision the chat input was drawn locked earlier in this run, so redraw to
    # unlock it.
    if needs_title or after_decision or get_pending_approval(thread_id):
        st.rerun()


def run_user_turn(text, files, thread_id, greeting_placeholder):
    """Show the user's message, index any PDFs, then stream the answer in one bubble."""
    greeting_placeholder.empty()
    history = st.session_state["message_history"]

    user_entry = {"role": "user", "content": text or ""}
    if files:
        user_entry["files"] = [uploaded.name for uploaded in files]
    history.append(user_entry)
    render_user_message(user_entry)

    bubble = st.chat_message("assistant", avatar=ASSISTANT_AVATAR)

    indexed = []
    if files:
        indexed = ingest_uploaded_pdfs(files, thread_id, bubble)
        if indexed:
            remember_docs(thread_id, indexed)
            with bubble:
                st.markdown(components.file_chips(indexed, "indexed"), unsafe_allow_html=True)

    if not text:
        if indexed:
            names = ", ".join(f"**{name}**" for name in indexed)
            reply = f"Indexed {names}. Ask me anything about it."
            with bubble:
                st.markdown(reply)
            history.append({"role": "assistant", "content": reply, "indexed": indexed})
        return

    result = stream_assistant_response(text, thread_id, bubble)
    if result["had_error"]:
        return

    entry = {
        "role": "assistant",
        "content": result["text"],
        "tools": result["steps"],
        "tool_state": result["state"],
    }
    if indexed:
        entry["indexed"] = indexed
    history.append(entry)

    finish_turn(thread_id, text, result["text"])


configure_page()
init_session_state()

# ============================ Sidebar threading feature ============================

render_sidebar()
sync_thread_url()

# =================================== Main chat interface ============================

thread_id = st.session_state["thread_id"]
history = st.session_state["message_history"]

render_header(
    get_thread_title(thread_id) if history else "New chat",
    get_thread_docs(thread_id),
)

greeting_placeholder, suggested_prompt = render_greeting_screen()

# A tool is paused mid-run, so this conversation cannot move until the user answers
pending_approval = get_pending_approval(thread_id)

# The paused turn is drawn together with its approval card, so the answer that follows
# the decision streams into the same bubble
paused_entry = None
if (
    pending_approval
    and history
    and history[-1]["role"] == "assistant"
    and history[-1].get("tool_state") == "awaiting"
):
    paused_entry = history[-1]

render_message_history(history[:-1] if paused_entry else history)

decision = None
if pending_approval:
    paused_steps = paused_entry["tools"] if paused_entry else []
    bubble = st.chat_message("assistant", avatar=ASSISTANT_AVATAR)
    with bubble:
        decision_slot = st.empty()
        tool_slot = st.empty()
        with tool_slot.container():
            render_tool_panel(paused_steps, "awaiting", "paused")
        decision = render_approval_card(pending_approval)

submission = st.chat_input(
    "Approve or decline the request above to continue"
    if pending_approval
    else "Ask Magnus anything…",
    accept_file=True,
    file_type=["pdf"],
    disabled=pending_approval is not None,
)

user_input = suggested_prompt
uploaded_files = []

if submission:
    user_input = (submission.text or "").strip() or None
    uploaded_files = list(submission.files or [])

if decision:
    approved = decision == APPROVE
    decision_slot.markdown(
        components.decision_chip(pending_approval["tool"], approved), unsafe_allow_html=True
    )
    result = resume_assistant_response(
        decision, thread_id, bubble, tool_slot, paused_steps
    )

    if not result["had_error"]:
        entry = paused_entry or {"role": "assistant"}
        entry.update(
            content=result["text"],
            tools=result["steps"],
            tool_state=result["state"],
            decision={"tool": pending_approval["tool"], "approved": approved},
        )
        if paused_entry is None:
            history.append(entry)
        finish_turn(thread_id, after_decision=True)

elif user_input or uploaded_files:
    run_user_turn(user_input, uploaded_files, thread_id, greeting_placeholder)
