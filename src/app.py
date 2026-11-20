from langchain_core.messages import AIMessageChunk, HumanMessage
from agentic_chatbot_backend import chatbot_workflow
import streamlit as st

st.title("Agentic Chatbot")

thread_id = "1"
config = {"configurable": {"thread_id": thread_id}}

if "message_history" not in st.session_state:
    st.session_state.message_history = []

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Enter your message: ")

if user_input:
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        assistant_response = st.write_stream(
            message_chunk.text
            for message_chunk, metadata in chatbot_workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            )
            if isinstance(message_chunk, AIMessageChunk) and message_chunk.text
        )

    st.session_state.message_history.append(
        {"role": "assistant", "content": assistant_response}
    )