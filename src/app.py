import streamlit as st

from ui.documents import ingest_uploaded_pdfs
from ui.layout import (
    configure_page,
    render_greeting_screen,
    render_header,
    render_message_history,
)
from ui.streaming import stream_assistant_response
from ui.threads import add_thread, init_session_state, render_sidebar
from utils.helper import generate_title

configure_page()
render_header()
init_session_state()

# ============================ Sidebar threading feature ============================

render_sidebar()

# =================================== Main chat interface ============================

greeting_placeholder = render_greeting_screen()
render_message_history()

submission = st.chat_input("Ask Magnus...", accept_file=True, file_type=["pdf"])

# A question typed alongside an upload is answered on the rerun that follows indexing
user_input = st.session_state.pop("pending_input", None)

if submission:
    user_input = submission.text or user_input

    if submission.files:
        indexed_names = ingest_uploaded_pdfs(submission.files)

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

if user_input:
    greeting_placeholder.empty()
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    thread_id = st.session_state["thread_id"]
    assistant_response, had_error = stream_assistant_response(user_input, thread_id)

    if not had_error and assistant_response:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": assistant_response}
        )

        add_thread(thread_id)
        if st.session_state["thread_titles"].get(thread_id) in (None, "New Chat"):
            st.session_state["thread_titles"][thread_id] = generate_title(
                user_input, assistant_response
            )
            st.rerun()
