import logging

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from agentic_chatbot_backend import chatbot_workflow
from ui.layout import LOGO_PATH
from utils.helper import get_friendly_error_message

logger = logging.getLogger(__name__)


def _build_config(thread_id):
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_trace",
    }


def stream_assistant_response(user_input, thread_id):
    """Stream one assistant reply, showing tool activity as it happens.

    Returns the final answer text and whether the turn failed.
    """
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
                config=_build_config(thread_id),
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

    return assistant_response, had_error
