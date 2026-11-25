import streamlit as st

from ui.approval import get_pending_approval, render_approval_card
from ui.documents import ingest_uploaded_pdfs
from ui.layout import (
    configure_page,
    render_greeting_screen,
    render_header,
    render_message_history,
)
from ui.streaming import resume_assistant_response, stream_assistant_response
from ui.threads import add_thread, init_session_state, render_sidebar
from utils.helper import generate_title


def finish_turn(thread_id, assistant_response, user_input=None):
    """Save the reply, then redraw when the sidebar or an approval card needs to change."""
    if assistant_response:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": assistant_response}
        )

    add_thread(thread_id)

    titles = st.session_state["thread_titles"]
    needs_title = (
        user_input and assistant_response and titles.get(thread_id) in (None, "New Chat")
    )
    if needs_title:
        titles[thread_id] = generate_title(user_input, assistant_response)

    # A paused tool leaves nothing on screen, so redraw to put the approval card in view
    if needs_title or get_pending_approval(thread_id):
        st.rerun()


configure_page()
render_header()
init_session_state()

# ============================ Sidebar threading feature ============================

render_sidebar()

# =================================== Main chat interface ============================

thread_id = st.session_state["thread_id"]

greeting_placeholder = render_greeting_screen()
render_message_history()

# A tool is paused mid-run, so this conversation cannot move until the user answers
pending_approval = get_pending_approval(thread_id)
decision = render_approval_card(pending_approval) if pending_approval else None

submission = st.chat_input(
    "Ask Magnus...",
    accept_file=True,
    file_type=["pdf"],
    disabled=pending_approval is not None,
)

# A question typed alongside an upload is answered on the rerun that follows indexing
user_input = st.session_state.pop("pending_input", None)

if submission:
    user_input = submission.text or user_input

    if submission.files:
        indexed_names = ingest_uploaded_pdfs(submission.files, thread_id)

        if indexed_names:
            file_list = ", ".join(f"`{name}`" for name in indexed_names)
            st.session_state["message_history"].append(
                {
                    "role": "assistant",
                    "content": f"📄 Indexed {file_list}. Ask me anything about it.",
                }
            )
            st.session_state["pending_input"] = user_input
            st.rerun()

if decision:
    assistant_response, had_error = resume_assistant_response(decision, thread_id)

    if not had_error:
        finish_turn(thread_id, assistant_response)

elif user_input:
    greeting_placeholder.empty()
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    assistant_response, had_error = stream_assistant_response(user_input, thread_id)

    if not had_error:
        finish_turn(thread_id, assistant_response, user_input)
