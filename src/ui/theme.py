from pathlib import Path

import streamlit as st

STYLES_PATH = Path(__file__).resolve().parent / "styles.css"


def inject_styles():
    """Load the Aurora stylesheet. Style-only HTML takes no space in the layout."""
    st.html(STYLES_PATH)


# Icons come from styles.css (CSS masks): st.html strips inline SVG
_THEMES = ["System", "Light", "Dark"]

# Streamlit keeps the viewer's theme choice in localStorage as "System", "Light" or
# "Dark" under this key and reads it on page load. It is not a public API, so if a
# Streamlit upgrade renames it the buttons only reload the page and the theme is
# still available from the app menu (⋮ → Settings).
#
# st.html sanitizes with DOMPurify, which drops a <script> whose text contains "<"
# followed by a letter, so keep the script free of anything that looks like a tag.
_THEME_SCRIPT = """
<script>
(() => {
  const KEY = `stActiveTheme-${window.location.pathname}-v2`;
  const read = () => {
    try { return JSON.parse(window.localStorage.getItem(KEY)) || "System"; }
    catch (error) { return "System"; }
  };
  const sync = () => {
    const active = read();
    document.querySelectorAll("[data-mg-theme]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.mgTheme === active));
    });
    // Dialogs and toasts render outside .stApp, where Streamlit's color-scheme does not
    // reach, so copy it to the root element to keep light-dark() tokens right for them too
    const app = document.querySelector(".stApp");
    if (app) document.documentElement.style.colorScheme = getComputedStyle(app).colorScheme;
  };
  document.querySelectorAll("[data-mg-theme]").forEach((button) => {
    if (button.dataset.bound) return;
    button.dataset.bound = "1";
    button.addEventListener("click", () => {
      if (button.dataset.mgTheme === read()) return;
      try { window.localStorage.setItem(KEY, JSON.stringify(button.dataset.mgTheme)); }
      catch (error) { return; }
      window.location.reload();
    });
  });
  sync();
  // The app menu can change the theme without a reload, so keep the buttons in step
  if (!window.__mgThemeSync) window.__mgThemeSync = window.setInterval(sync, 700);
})();
</script>
"""


def render_theme_switch():
    buttons = "".join(
        f'<button type="button" data-mg-theme="{name}" aria-pressed="{str(name == "System").lower()}">'
        f"{name}</button>"
        for name in _THEMES
    )
    st.html(
        f'<div class="mg-theme" role="group" aria-label="Theme">{buttons}</div>{_THEME_SCRIPT}',
        unsafe_allow_javascript=True,
    )
