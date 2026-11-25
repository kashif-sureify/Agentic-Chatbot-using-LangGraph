import html

import streamlit as st

from agentic_chatbot_backend import chatbot_workflow
from ui.layout import LOGO_PATH

# The strings `purchase_stock` and `get_current_location` compare their resume value against
APPROVE = "yes"
DECLINE = "no"

CARD_STYLE = """
<style>
    .st-key-approval_card {
        border: 1px solid rgba(240, 173, 78, 0.45);
        border-left: 4px solid #f0ad4e;
        border-radius: 0.75rem;
        background: rgba(240, 173, 78, 0.07);
        padding: 1rem 1.25rem;
    }
    .approval-head {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 0.65rem;
    }
    .approval-badge {
        background: #f0ad4e;
        color: #1a1a1a;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
    }
    .approval-tool { font-size: 0.85rem; opacity: 0.75; }
    .approval-question { font-size: 1.05rem; font-weight: 600; }
    .approval-detail {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f0ad4e;
        margin-top: 0.45rem;
    }
    .approval-args {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin-top: 0.55rem;
    }
    .approval-arg {
        font-size: 0.78rem;
        padding: 0.2rem 0.55rem;
        border-radius: 0.4rem;
        background: rgba(128, 128, 128, 0.18);
    }
    .approval-reason {
        font-size: 0.85rem;
        opacity: 0.7;
        margin: 0.7rem 0 0.9rem 0;
    }
</style>
"""


def get_pending_approval(thread_id):
    """Return the request a paused tool is waiting on, or None when nothing is paused.

    `interrupt()` stops the graph mid-tool and writes the pause into the checkpoint, so the
    saved state is what survives a browser refresh or a switch between conversations.
    """
    try:
        snapshot = chatbot_workflow.get_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        # A brand new thread has no checkpoint to read yet
        return None

    interrupts = snapshot.interrupts or ()
    if not interrupts:
        return None

    request = interrupts[0].value
    if not isinstance(request, dict):
        request = {"question": str(request)}

    # The tool call the graph stopped inside, so the card can name what is about to run
    messages = snapshot.values.get("messages", [])
    tool_calls = getattr(messages[-1], "tool_calls", None) if messages else None
    pending_call = tool_calls[0] if tool_calls else {}

    return {
        "question": request.get("question", "Approve this action?"),
        "details": request.get("details", ""),
        "reason": request.get("reason", ""),
        "tool": pending_call.get("name", ""),
        "args": pending_call.get("args", {}),
    }


def _card_body(request):
    tool = (
        f"<code>{html.escape(request['tool'])}</code>" if request["tool"] else "a tool"
    )

    chips = "".join(
        f"<span class='approval-arg'>{html.escape(str(name))}: "
        f"<b>{html.escape(str(value))}</b></span>"
        for name, value in request["args"].items()
    )
    # Each block is dropped when empty so a tool without arguments or a cost
    # line does not leave a gap in the card
    arguments = f'<div class="approval-args">{chips}</div>' if chips else ""

    details = (
        f'<div class="approval-detail">{html.escape(request["details"])}</div>'
        if request["details"]
        else ""
    )

    # Kept flush left: Markdown reads indented lines as a code block
    return f"""
<div class="approval-head">
<span class="approval-badge">Approval required</span>
<span class="approval-tool">Magnus wants to run {tool}</span>
</div>
<div class="approval-question">{html.escape(request["question"])}</div>
{details}
{arguments}
<div class="approval-reason">{html.escape(request["reason"])}</div>
"""


def render_approval_card(request):
    """Draw the approval prompt and return the decision, or None while the user decides."""
    placeholder = st.empty()
    decision = None

    with placeholder.container():
        with st.chat_message("assistant", avatar=str(LOGO_PATH)):
            with st.container(key="approval_card"):
                st.markdown(CARD_STYLE + _card_body(request), unsafe_allow_html=True)

                approve_column, decline_column = st.columns(2)

                if approve_column.button(
                    "Approve",
                    key="approve_action",
                    type="primary",
                    icon=":material/check:",
                    width="stretch",
                ):
                    decision = APPROVE

                if decline_column.button(
                    "Decline",
                    key="decline_action",
                    icon=":material/close:",
                    width="stretch",
                ):
                    decision = DECLINE

    # Clear the card straight away so it is not left on screen while the answer streams in
    if decision:
        placeholder.empty()

    return decision
