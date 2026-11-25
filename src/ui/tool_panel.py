"""The collapsible panel that shows which tools Magnus used in a turn."""

import uuid

import streamlit as st

from ui import components

STATE_ICONS = {
    "done": ":material/check_circle:",
    "awaiting": ":material/pause_circle:",
    "error": ":material/error:",
}


def panel_label(steps, state):
    if state == "awaiting":
        return "Paused · waiting for your approval"
    if state == "error":
        return "A tool ran into a problem"

    names = list(dict.fromkeys(step["name"] for step in steps))
    return "Used " + ", ".join(f"`{name}`" for name in names) if names else "Tools finished"


def render_tool_panel(steps, state, key):
    """Draw a finished panel. The container key carries the state so CSS can color it."""
    with st.container(key=f"mg-tools-{state}-{key}"):
        with st.expander(panel_label(steps, state), expanded=False, icon=STATE_ICONS[state]):
            if steps:
                st.markdown(components.tool_steps(steps), unsafe_allow_html=True)
            else:
                st.caption("Magnus is waiting for your decision before it runs the tool.")


class LiveToolPanel:
    """Tool activity for a turn that is still streaming.

    It shows a running `st.status` while tools are called and is swapped for the same
    static panel the history uses once the turn ends, so a reply looks identical live
    and after a rerun.
    """

    def __init__(self, slot, steps=()):
        self.slot = slot
        self.steps = list(steps)
        self._status = None
        self._body = None

    @property
    def started(self):
        return self._status is not None or bool(self.steps)

    def running(self, tool_name):
        label = f"Using `{tool_name}` …"
        if self._status is None:
            with self.slot.container(key=f"mg-tools-running-{uuid.uuid4().hex[:8]}"):
                self._status = st.status(label, expanded=True, state="running")
            if self.steps:
                self._render_steps()
        else:
            self._status.update(label=label, state="running", expanded=True)

    def add_step(self, tool_name, raw_result):
        self.steps.append({"name": tool_name, "result": components.result_preview(raw_result)})
        self.running(tool_name)
        self._render_steps()

    def _render_steps(self):
        if self._body is None:
            with self._status:
                self._body = st.empty()
        self._body.markdown(components.tool_steps(self.steps), unsafe_allow_html=True)

    def finish(self, state):
        if not self.started and state != "awaiting":
            return
        with self.slot.container():
            render_tool_panel(self.steps, state, uuid.uuid4().hex[:8])
