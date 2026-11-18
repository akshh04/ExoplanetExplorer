"""Register the browser bridge once, outside Streamlit's rerun script."""

from pathlib import Path

import streamlit as st

browser_state = st.components.v2.component(
    "explorer_browser_state",
    js=Path(__file__).with_name("browser_state.js").read_text(encoding="utf-8"),
)

route_writer = st.components.v2.component(
    "explorer_route_writer",
    js="""
    export default function ({ data }) {
        const url = new URL(location.href);
        for (const key of data.managed) url.searchParams.delete(key);
        for (const [key, value] of Object.entries(data.params)) url.searchParams.set(key, value);
        if (url.search !== location.search) {
            if (data.replace) history.replaceState(history.state, "", url);
            else history.pushState(history.state, "", url);
        }
    }
    """,
)
