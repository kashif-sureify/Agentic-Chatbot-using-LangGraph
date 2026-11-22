from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage
from agentic_chatbot_backend import chatbot_workflow
from utils.helper import generate_title, generate_greeting, get_all_threads
from datetime import datetime
from pathlib import Path
import streamlit as st
import base64
import html
import uuid


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

    with st.chat_message("assistant", avatar=str(LOGO_PATH)):
        assistant_response = st.write_stream(
            message_chunk.text
            for message_chunk, metadata in chatbot_workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
            if isinstance(message_chunk, AIMessageChunk) and message_chunk.text
        )

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
