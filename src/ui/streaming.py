import logging

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command

from agentic_chatbot_backend import chatbot_workflow
from ui import components
from ui.approval import get_pending_approval
from ui.tool_panel import LiveToolPanel
from utils.helper import get_friendly_error_message

logger = logging.getLogger(__name__)


def _build_config(thread_id, run_name):
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": run_name,
    }


def stream_assistant_response(user_input, thread_id, bubble):
    """Answer a new user message inside `bubble`, an assistant chat message."""
    return _stream_turn(
        {"messages": [HumanMessage(content=user_input)]}, thread_id, "chat_trace", bubble
    )


def resume_assistant_response(decision, thread_id, bubble, tool_slot, prior_steps):
    """Hand the user's decision back to the tool that is waiting on `interrupt()`.

    The decision is the whole input: re-sending the original message would start a
    second turn instead of restarting the paused one. The reply continues in the same
    bubble and tool panel as the paused turn, so it reads as one answer.
    """
    return _stream_turn(
        Command(resume=decision),
        thread_id,
        "approval_resume_trace",
        bubble,
        tool_slot=tool_slot,
        prior_steps=prior_steps,
    )


def _stream_turn(graph_input, thread_id, run_name, bubble, tool_slot=None, prior_steps=()):
    """Stream one assistant reply, showing tool activity as it happens.

    Returns a dict with the answer `text`, the tool `steps`, the panel `state`
    ("done", "awaiting" or "error") and whether the turn failed (`had_error`).
    """
    with bubble:
        # Reserved above the answer so tool activity always renders on top of the reply
        if tool_slot is None:
            tool_slot = st.empty()
        response_placeholder = st.empty()

    panel = LiveToolPanel(tool_slot, prior_steps)

    # The model is called with `invoke`, so nothing streams until it answers.
    # Without this the turn shows an empty bubble for several seconds.
    response_placeholder.markdown(components.thinking(), unsafe_allow_html=True)

    committed_text = ""
    step_text = ""
    current_step = None
    current_run_id = None
    result = {"text": "", "steps": panel.steps, "state": "done", "had_error": False}

    try:
        for message_chunk, metadata in chatbot_workflow.stream(
            graph_input,
            config=_build_config(thread_id, run_name),
            stream_mode="messages",
        ):
            # A tool finished running, so show what it returned
            if isinstance(message_chunk, ToolMessage):
                panel.add_step(message_chunk.name or "tool", message_chunk.text)
                # The model now reads the result and writes again, which takes a moment
                if not (committed_text or step_text):
                    response_placeholder.markdown(components.thinking(), unsafe_allow_html=True)
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
                    if not (committed_text or step_text):
                        # The panel now shows progress, so drop the thinking indicator
                        response_placeholder.empty()
                    panel.running(tool_call["name"])

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

        result["text"] = committed_text + step_text

        # A tool that called `interrupt()` ends the turn with no answer to show,
        # which is a pause rather than the failure the warning below describes
        awaiting_approval = get_pending_approval(thread_id) is not None

        if result["text"]:
            response_placeholder.markdown(result["text"])
        elif awaiting_approval:
            response_placeholder.empty()
        else:
            result["had_error"] = True
            response_placeholder.warning(
                "I couldn't generate a response right now. Please try again.",
                icon=":material/warning:",
            )

        result["state"] = "awaiting" if awaiting_approval else "done"
        panel.finish(result["state"])

    except Exception as error:
        result["had_error"] = True
        result["state"] = "error"
        logger.exception("Error while generating chatbot response")
        panel.finish("error")
        response_placeholder.warning(get_friendly_error_message(error), icon=":material/warning:")

    result["steps"] = panel.steps
    return result
