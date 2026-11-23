from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage, ToolMessage
from agentic_chatbot_backend import chatbot_workflow
from utils.helper import (
    generate_title,
    generate_greeting,
    get_all_threads,
    get_friendly_error_message,
)
from datetime import datetime
from pathlib import Path
import streamlit as st
import base64
import html
import logging
import uuid

logger = logging.getLogger(__name__)


LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

st.set_page_config(page_title="Magnus AI", page_icon=str(LOGO_PATH))
st.logo(str(LOGO_PATH), icon_image=str(LOGO_PATH), size="large")


@st.cache_data
def get_logo_base64(path: str, modified_at: float) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


LOGO_B64 = get_logo_base64(str(LOGO_PATH), LOGO_PATH.stat().st_mtime)

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


# Generate a unique thread ID for each chat session
def generate_thread_id():
    return str(uuid.uuid4())


# Add a new thread ID for the conversation list if it doesn't exist
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# Creates a new chat thread only if the current thread has messages
def reset_chat():
    if not st.session_state["message_history"]:
        return
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []


# Load a previous conversation from the langgraph Checkpointer
def load_conversation(thread_id):

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


def get_time_based_greeting():
    current_time = datetime.now().strftime("%I:%M %p, %A")
    return generate_greeting(current_time)


st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.75rem;">
        <img src="data:image/png;base64,{LOGO_B64}" width="48" height="48" alt="Magnus logo" />
        <h1 style="margin: 0; padding: 0; font-size: 2.5rem; font-weight: 700;">Magnus AI</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# config = {"configurable": {"thread_id": thread_id}}

if "message_history" not in st.session_state:
    st.session_state.message_history = []
    
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

if "thread_greetings" not in st.session_state:
    st.session_state["thread_greetings"] = {}


# ============================ Sidebar threading feature ============================

st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat", width="stretch"):
    if st.session_state["message_history"]:
        reset_chat()
        st.rerun()

# Keep only threads that actually have messages so no empty duplicate threads appear
st.session_state["chat_threads"] = [
    tid for tid in st.session_state["chat_threads"] if load_conversation(tid)
]

# Display the list of existing chat threads in the sidebar
for thread_id in st.session_state["chat_threads"][::-1]:
    title = get_thread_title(thread_id)
    is_active = thread_id == st.session_state["thread_id"]
    
    # Creates a sidebar button for every existing chat thread
    if st.sidebar.button(
        title,
        key=thread_id,
        type="primary" if is_active else "secondary",
        width="stretch",
    ):
        
        # Set the selected thread as the current thread in the session state
        st.session_state["thread_id"] = thread_id
        
        # Load the conversation history for the selected thread
        messages = load_conversation(thread_id)
        
        # Temporary list for converting Langchain messages into Streamlit message format
        
        temp_messages = []
        
        for message in messages:
            
            # Check whether the message is from the user or the assistant
            
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue

            temp_messages.append({"role": role, "content": message.text})

        st.session_state["message_history"] = temp_messages
        
        st.rerun()
        
        
# =================================== Main chat interface ============================

greeting_placeholder = st.empty()

if not st.session_state.message_history:
    current_thread_id = st.session_state["thread_id"]
    if current_thread_id not in st.session_state["thread_greetings"]:
        st.session_state["thread_greetings"][current_thread_id] = get_time_based_greeting()
    greeting = html.escape(st.session_state["thread_greetings"][current_thread_id])
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

for message in st.session_state.message_history:
    avatar = str(LOGO_PATH) if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

user_input = st.chat_input("Ask Magnus...")

if user_input:
    greeting_placeholder.empty()
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    ##  Pass the current thread ID to the chatbot workflow
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace"
    }

    assistant_response = ""
    had_error = False

    with st.chat_message("assistant", avatar=str(LOGO_PATH)):
        # Reserved above the answer so tool activity always renders on top of the reply
        status_container = st.container()
        response_placeholder = st.empty()

        status_holder = {"box": None}
        used_tools = []

        def show_tool_status(label):
            if status_holder["box"] is None:
                with status_container:
                    status_holder["box"] = st.status(label, expanded=True)
            else:
                status_holder["box"].update(label=label, state="running", expanded=True)

        committed_text = ""
        step_text = ""
        current_step = None
        current_run_id = None

        try:
            for message_chunk, metadata in chatbot_workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # A tool finished running, so show what it returned
                if isinstance(message_chunk, ToolMessage):
                    tool_name = message_chunk.name or "tool"
                    show_tool_status(f"🔧 Using `{tool_name}` …")
                    result = message_chunk.text
                    with status_holder["box"]:
                        st.markdown(
                            f"**{tool_name}** → {result[:300]}{'…' if len(result) > 300 else ''}"
                        )
                    if tool_name not in used_tools:
                        used_tools.append(tool_name)
                    continue

                if not isinstance(message_chunk, (AIMessageChunk, AIMessage)):
                    continue

                # The model announced a tool call, so surface it before the tool actually runs
                announced_calls = (
                    getattr(message_chunk, "tool_call_chunks", None)
                    or getattr(message_chunk, "tool_calls", None)
                    or []
                )
                for tool_call in announced_calls:
                    if tool_call.get("name"):
                        show_tool_status(f"🔧 Using `{tool_call['name']}` …")

                if not message_chunk.text:
                    continue

                step = metadata.get("langgraph_step")
                run_id = message_chunk.id

                if step != current_step:
                    committed_text += step_text
                    step_text = ""
                    current_step = step
                    current_run_id = run_id
                elif run_id and current_run_id and run_id != current_run_id:
                    # A fallback model took over mid-stream in the same step; discard partial output
                    step_text = ""
                    current_run_id = run_id

                step_text += message_chunk.text
                response_placeholder.markdown(committed_text + step_text + "▌")

            assistant_response = committed_text + step_text
            if assistant_response:
                response_placeholder.markdown(assistant_response)
            else:
                had_error = True
                response_placeholder.warning(
                    "I couldn't generate a response right now. Please try again."
                )

            if status_holder["box"] is not None:
                tool_list = ", ".join(f"`{name}`" for name in used_tools)
                status_holder["box"].update(
                    label=f"✅ Tools Used {tool_list}" if tool_list else "✅ Tools finished",
                    state="complete",
                    expanded=False,
                )
        except Exception as error:
            had_error = True
            logger.exception("Error while generating chatbot response")
            response_placeholder.empty()
            if status_holder["box"] is not None:
                status_holder["box"].update(state="error", expanded=False)
            st.warning(get_friendly_error_message(error))

    if not had_error and assistant_response:
        st.session_state.message_history.append(
            {"role": "assistant", "content": assistant_response}
        )

        current_thread_id = st.session_state["thread_id"]
        add_thread(current_thread_id)
        if st.session_state["thread_titles"].get(current_thread_id) in (None, "New Chat"):
            st.session_state["thread_titles"][current_thread_id] = generate_title(
                user_input, assistant_response
            )
            st.rerun()
