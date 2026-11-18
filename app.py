"""Exoplanet Explorer — an interactive dashboard over the NASA Exoplanet Archive."""

import hashlib
import html
import inspect
import json
import logging
import threading
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import StringIO
from typing import Optional
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlsplit

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from streamlit.runtime.scriptrunner import add_script_run_ctx

import archive_cache
from data_contract import validate_archive_columns
from explorer_browser import browser_state, route_writer

LOGGER = logging.getLogger("exoplanet_explorer")

st.set_page_config(
    page_title="Exoplanet Explorer",
    layout="wide",
    page_icon="🪐",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": (
            "Interactive exploration of the NASA Exoplanet Archive "
            "(Planetary Systems Composite Parameters table)."
        )
    },
)

# Browser preferences are loaded once per session and never overwrite URL state.
PREFERENCE_DEFAULTS = {
    "dark_mode": False, "distance_unit": "Light-years", "size_unit": "Earth",
    "catalogue_layout": "Cards", "favorites": [], "filter_sets": {}, "collections": {},
}
for preference, default in PREFERENCE_DEFAULTS.items():
    st.session_state.setdefault(preference, default)
    # Keep preferences whose widgets are only mounted on one page.
    st.session_state[preference] = st.session_state[preference]
with st.container(key="browser-state"):
    browser = browser_state(
        key="explorer_browser",
        data={"initialized": st.session_state.get("preferences_loaded", False),
              "preferences": {key: st.session_state[key] for key in PREFERENCE_DEFAULTS},
              "view": st.session_state.get("active_view", "Explore"),
              "preview": st.session_state.get("preview_planet")},
        on_preferences_change=lambda: None,
        on_route_change=lambda: None,
    )
if not st.session_state.get("preferences_loaded") and browser.preferences is not None:
    if isinstance(browser.preferences, dict):
        for preference, default in PREFERENCE_DEFAULTS.items():
            value = browser.preferences.get(preference)
            if type(value) is type(default) and (
                not isinstance(default, str) or value in {
                    "Earth", "Jupiter", "Light-years", "Parsecs", "Cards", "Table",
                }
            ):
                st.session_state[preference] = value
    st.session_state["preferences_loaded"] = True
    st.rerun()

# Keep the route writer in the same location as the page controls change.
browser_route = st.container(key="browser-route")

# --- THEME ---------------------------------------------------------------
# Streamlit's real theme (the one that also reskins native canvas-rendered
# widgets like st.dataframe) can only be changed via .streamlit/config.toml,
# which needs a server restart to take effect — no good for a per-user,
# click-to-toggle control. So this toggle is a self-contained light/dark skin:
# session-state-driven CSS for the page chrome, plus explicit Plotly colours
# for the charts (rather than theme="streamlit", which always follows the
# server's static config and would otherwise go unreadable in dark mode).
# The one thing this can't reach is st.dataframe's own canvas grid, which
# keeps rendering with the server's light theme — a light table on a dark
# page, still fully legible, just not colour-matched.
st.session_state.setdefault("dark_mode", False)
IS_DARK = st.session_state["dark_mode"]

if IS_DARK:
    PAGE_BG, SIDEBAR_BG, CARD_BG = "#0e1117", "#161b22", "#1c2129"
    TEXT_COLOR, MUTED_TEXT, BORDER = "#e6edf3", "#9aa4b2", "rgba(230,237,243,0.16)"
    INPUT_BG = "#20262f"
    CHART_FONT, CHART_GRID = "#e6edf3", "rgba(230,237,243,0.15)"
    ACCENT_SOFT, SHADOW = "rgba(167,139,250,0.55)", "rgba(0,0,0,0.45)"
    HOVER_TINT, FOCUS_RING = "rgba(230,237,243,0.07)", "#a78bfa"
    LINK_COLOR = "#7dd3fc"
    OVERVIEW_TINT, HABITABLE_ACCENT, HABITABLE_TINT = "#1c1730", "#5eead4", "#15312e"
else:
    PAGE_BG, SIDEBAR_BG, CARD_BG = "#ffffff", "#f7f8fb", "#ffffff"
    TEXT_COLOR, MUTED_TEXT, BORDER = "#31333f", "#5c6470", "rgba(49,51,63,0.16)"
    INPUT_BG = "#ffffff"
    CHART_FONT, CHART_GRID = "#31333f", "rgba(49,51,63,0.12)"
    ACCENT_SOFT, SHADOW = "rgba(124,58,237,0.45)", "rgba(15,23,42,0.10)"
    HOVER_TINT, FOCUS_RING = "rgba(49,51,63,0.05)", "#7c3aed"
    LINK_COLOR = "#0284c7"
    OVERVIEW_TINT, HABITABLE_ACCENT, HABITABLE_TINT = "#f7f3ff", "#0f766e", "#e6f6f3"

_dark_overrides = f"""
    html, body, .stApp, [data-testid="stHeader"], [data-testid="stAppViewContainer"] {{
        background-color: {PAGE_BG} !important;
        color: {TEXT_COLOR} !important;
    }}
    [data-testid="stSidebar"], [data-testid="stPopoverBody"], [role="dialog"] {{
        background-color: {SIDEBAR_BG} !important;
        color: {TEXT_COLOR} !important;
    }}
    h1, h2, h3, h4, h5, h6, p, span, label, li, [data-testid="stMarkdownContainer"] {{
        color: {TEXT_COLOR} !important;
    }}
    a {{ color: #7dd3fc !important; }}
    [data-testid="stCaptionContainer"], .hero-sub {{ color: {MUTED_TEXT} !important; }}
    [data-testid="stMetric"] {{ background-color: {CARD_BG} !important; border-color: {BORDER} !important; }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED_TEXT} !important; }}
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{ color: {TEXT_COLOR} !important; }}
    .st-key-home-overview [data-testid="stMetricDelta"] {{ background: #2b3141 !important; }}
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"],
    textarea, input[type="text"] {{
        background-color: {INPUT_BG} !important;
        color: {TEXT_COLOR} !important;
        border-color: {BORDER} !important;
    }}
    input::placeholder {{ color: {MUTED_TEXT} !important; opacity: 1 !important; }}
    [data-testid="stSelectbox"] [role="group"],
    [data-testid="stSelectbox"] [role="group"] button,
    [data-testid="stSelectbox"] input,
    [role="listbox"], [role="option"] {{
        background-color: {INPUT_BG} !important;
        color: {TEXT_COLOR} !important;
    }}
    [data-testid="stTooltipIcon"] button {{ color: {MUTED_TEXT} !important; }}
    .st-key-home-overview [data-testid="stTooltipIcon"] svg {{ stroke: {MUTED_TEXT} !important; }}
    [data-testid="stDataFrame"] canvas {{ filter: invert(0.9) hue-rotate(180deg); }}
    [role="option"]:hover, [role="option"][data-focused="true"] {{ background-color: #334155 !important; }}
    [data-baseweb="tag"] {{ background-color: #334155 !important; }}
    [role="listbox"], [data-baseweb="popover"] [role="listbox"] {{ background-color: {INPUT_BG} !important; }}
    [data-testid="stDataFrame"], [data-testid="stDataFrame"] > div {{
        border-radius: 10px;
        border: 1px solid {BORDER};
    }}
    hr, [data-testid="stDivider"] {{ border-color: {BORDER} !important; }}
    [data-testid="stExpander"], [data-testid="stStatusWidget"] {{
        background-color: {CARD_BG} !important;
        border-color: {BORDER} !important;
    }}
    button[kind], [data-testid^="stBaseButton"] {{
        background-color: {INPUT_BG} !important;
        color: {TEXT_COLOR} !important;
        border-color: {BORDER} !important;
    }}
""" if IS_DARK else ""

st.markdown(
    f"""
<style>
    .st-key-browser-state, [data-testid="stLayoutWrapper"]:has(> .st-key-browser-route) {{
        display: none;
    }}
    .st-key-catalogue-preview {{ scroll-margin-top: 10rem; }}
    .st-key-catalogue-preview h4 {{ padding-top: 0; }}
    .st-key-results-controls p {{ margin-bottom: 0; }}
    [data-testid="stMainBlockContainer"]:has(.st-key-home-toolbar) {{ max-width: 1120px; padding-top: 2.75rem; }}
    .st-key-home-toolbar [data-baseweb="input"] {{ min-height: 44px; }}
    .st-key-home-toolbar [data-testid="stPopover"] button {{ min-height: 44px; }}
    .hero-title .brand-accent {{ color: {FOCUS_RING} !important; }}
    .st-key-home-toolbar .st-key-open_filters button[kind] {{
        color: {FOCUS_RING} !important; border-color: {ACCENT_SOFT} !important;
    }}
    .st-key-home-workspace [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-home-rail) {{
        display: grid; grid-template-columns: minmax(0, 1fr); gap: 1rem;
    }}
    .st-key-home-workspace [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-home-rail) > [data-testid="stColumn"] {{
        min-width: 0 !important; width: 100% !important;
    }}
    .st-key-home-discovery {{ display: none; }}
    .st-key-home-overview {{
        background: linear-gradient(120deg, {OVERVIEW_TINT}, {CARD_BG});
        border-color: {ACCENT_SOFT};
    }}
    .st-key-home-overview .overview-heading {{
        font-size: 1.1rem; font-weight: 650; padding: 0; margin: 0; line-height: 1.25;
    }}
    .st-key-home-overview .st-key-headline-metrics [data-testid="stHorizontalBlock"] {{
        display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0.6rem;
    }}
    .st-key-home-overview .st-key-headline-metrics [data-testid="stColumn"] {{
        min-width: 0 !important; width: 100% !important;
    }}
    .st-key-home-overview [data-testid="stMetric"] {{
        height: auto; min-height: 0; padding: 0; border: 0;
        background: transparent !important;
    }}
    .st-key-home-overview [data-testid="stMetric"]:hover {{ transform: none; box-shadow: none; }}
    .st-key-home-overview [data-testid="stMetric"] > div {{
        grid-template-columns: minmax(0, 1fr); grid-template-rows: auto; align-items: start;
    }}
    .st-key-home-overview [data-testid="stMetricLabel"] p {{
        white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.3;
    }}
    .st-key-home-overview [data-testid="stMetricValue"] {{ font-size: clamp(1.35rem, 2.5vw, 1.85rem); }}
    .st-key-home-overview [data-testid="stMetric"] > div > div:has([data-testid="stMetricDelta"]) {{
        grid-column: 1; grid-row: 3;
    }}
    .st-key-home-overview [data-testid="stMetricDelta"] p {{
        white-space: normal; overflow: visible; text-overflow: clip; overflow-wrap: anywhere;
    }}
    .st-key-home-overview [data-testid="stMetricIcon"] {{
        color: {FOCUS_RING} !important;
    }}
    .st-key-home-overview [data-testid="stColumn"]:nth-child(3) [data-testid="stMetricIcon"] {{
        color: {HABITABLE_ACCENT} !important;
    }}
    .st-key-catalogue-results {{
        display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem;
    }}
    .st-key-catalogue-results [class*="st-key-planet-card-"] {{
        display: grid; grid-template-columns: minmax(0, 1fr) auto;
        grid-template-areas: "name preview" "details details";
        align-items: center; gap: 0.25rem 0.65rem;
        background: {CARD_BG}; border-color: {BORDER};
    }}
    .st-key-catalogue-results [class*="st-key-planet-card-"] > :first-child {{ grid-area: name; }}
    .st-key-catalogue-results [class*="st-key-planet-card-"] > :nth-child(2) {{ grid-area: details; }}
    .st-key-catalogue-results [class*="st-key-planet-card-"] > :last-child {{ grid-area: preview; }}
    .st-key-catalogue-results [class*="st-key-planet-card-"] > * {{ min-width: 0; }}
    .st-key-catalogue-results [class*="st-key-card-preview-"] {{ justify-self: end; }}
    .st-key-home-catalogue .st-key-home-toolbar [data-testid="stColumn"] {{ min-width: 0; }}
    .st-key-home-catalogue .st-key-home-toolbar [data-testid="stPopover"] button {{
        white-space: nowrap;
    }}
    .st-key-catalogue-results [data-testid="stButton"] button[kind="tertiary"] {{
        justify-content: flex-start; padding: 0.25rem 0; text-align: left;
        font-size: 1.05rem; font-weight: 650;
        color: {TEXT_COLOR} !important; border: 0 !important; background: transparent !important;
    }}
    .st-key-catalogue-results [data-testid="stButton"] button[kind="tertiary"]:hover {{
        color: {FOCUS_RING} !important;
    }}
    .st-key-catalogue-results button[kind="tertiary"] > div,
    .st-key-catalogue-results button[kind="tertiary"] [data-has-shortcut] {{ width: 100%; }}
    .st-key-catalogue-results button[kind="tertiary"] [data-has-shortcut] {{
        display: flex; justify-content: space-between;
    }}
    .st-key-catalogue-results button[kind="tertiary"] span:has(> [data-testid="stIconMaterial"]) {{
        order: 1; color: {MUTED_TEXT};
    }}
    .st-key-catalogue-results button[kind="tertiary"] p {{
        font-size: 1.05rem; font-weight: 650; overflow-wrap: anywhere;
    }}
    .st-key-catalogue-results [class*="st-key-card-preview-"] button[kind="tertiary"] {{
        min-width: 44px; justify-content: center; border-radius: 10px;
        background: {OVERVIEW_TINT} !important;
    }}
    .st-key-catalogue-results [class*="st-key-card-preview-"] button[kind="tertiary"] > div,
    .st-key-catalogue-results [class*="st-key-card-preview-"] button[kind="tertiary"] [data-has-shortcut] {{
        width: auto;
    }}
    .st-key-catalogue-results [class*="st-key-card-preview-"] button [data-testid="stIconMaterial"] {{
        color: {FOCUS_RING} !important;
    }}
    .st-key-catalogue-results [class*="st-key-card-preview-"] button p {{
        position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0;
        overflow: hidden; clip-path: inset(50%); white-space: nowrap;
    }}
    .catalogue-meta {{ color: {MUTED_TEXT} !important; font-size: 0.875rem; margin: 0; overflow-wrap: anywhere; }}
    .catalogue-zone {{ color: {MUTED_TEXT} !important; font-size: 0.8rem; margin: 0.35rem 0 0; }}
    .catalogue-zone.is-habitable {{
        display: inline-block; padding: 0.16rem 0.45rem; border-radius: 5px;
        background: {HABITABLE_TINT}; color: {HABITABLE_ACCENT} !important;
    }}
    [data-testid="stPlotlyChart"] {{ padding-top: 1.75rem; }}
    [data-testid="stPlotlyChart"] .modebar {{ top: -1.5rem !important; }}
    .stApp, [data-testid="stAppViewContainer"] {{ background: {PAGE_BG}; }}
    [data-testid="stMainBlockContainer"] {{ max-width: 1440px; padding-top: 3.5rem; }}
    h2, h3 {{ letter-spacing: -0.025em; }}
    .st-key-explorer-navigation {{
        position: sticky; top: 2.8rem; z-index: 10;
        background: {PAGE_BG}; padding: 0.65rem 0;
        border-bottom: 1px solid {BORDER};
    }}
    .st-key-explorer-navigation [role="radiogroup"] {{
        display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.35rem; width: 100%;
    }}
    .st-key-explorer-navigation [role="radiogroup"] label {{
        display: flex; align-items: center; justify-content: center;
        min-height: 44px; margin: 0; padding: 0.45rem 0.25rem;
        color: {MUTED_TEXT}; border-radius: 0;
        cursor: pointer; border: 0; border-bottom: 2px solid transparent;
    }}
    .st-key-explorer-navigation .st-key-nav_group [role="radiogroup"] {{ max-width: 30rem; }}
    .st-key-explorer-navigation .st-key-chart_view [role="radiogroup"] {{
        grid-template-columns: repeat(6, minmax(0, 1fr));
    }}
    .st-key-explorer-navigation [role="radiogroup"] label {{ width: 100%; }}
    .st-key-explorer-navigation [role="radiogroup"] label > div > div:first-child {{ display: none; }}
    .st-key-explorer-navigation [role="radiogroup"] label:has(input:checked) {{
        border-bottom-color: {FOCUS_RING}; font-weight: 650;
    }}
    .st-key-explorer-navigation label:has(input:checked) p {{ color: {FOCUS_RING} !important; }}
    .st-key-explorer-navigation label:has(input:focus-visible) {{ outline: 2px solid {FOCUS_RING}; outline-offset: 2px; }}
    [data-testid="stButton"] button, [data-testid="stLinkButton"] a {{ min-height: 44px; border-radius: 10px; }}
    @media (min-width: 1100px) {{
        [data-testid="stMainBlockContainer"]:has(.st-key-home-workspace) {{
            max-width: none; padding-inline: clamp(2rem, 3.25vw, 4.5rem);
        }}
        .st-key-home-workspace [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-home-rail) {{
            grid-template-columns: clamp(16.75rem, 23%, 22rem) minmax(0, 1fr);
            gap: clamp(1.5rem, 2.75vw, 3.25rem);
        }}
        .st-key-home-overview .st-key-headline-metrics [data-testid="stHorizontalBlock"] {{
            grid-template-columns: minmax(0, 1fr); gap: 0.55rem;
        }}
        .st-key-home-overview .st-key-headline-metrics [data-testid="stColumn"] + [data-testid="stColumn"] {{
            border-top: 1px solid {BORDER}; padding-top: 0.55rem;
        }}
        .st-key-home-overview [data-testid="stMetric"] > div {{
            grid-template-columns: minmax(0, 1fr) auto; align-items: center;
            gap: 0.2rem 0.75rem;
        }}
        .st-key-home-overview [data-testid="stMetricLabel"] {{ grid-column: 1; grid-row: 1; }}
        .st-key-home-overview [data-testid="stMetricValue"] {{
            grid-column: 2; grid-row: 1; justify-self: end; font-size: 1.25rem;
            font-variant-numeric: tabular-nums;
        }}
        .st-key-home-overview [data-testid="stMetric"] > div > div:has([data-testid="stMetricDelta"]) {{
            grid-column: 1 / -1; grid-row: 2; justify-self: start;
        }}
        .st-key-home-discovery {{ display: flex; }}
        .st-key-home-toolbar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
            flex: 0 0 7.25rem;
        }}
        .st-key-home-toolbar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {{
            flex: 1 1 0;
        }}
        .st-key-home-toolbar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {{
            display: none;
        }}
        .st-key-home-catalogue .st-key-catalogue-results {{
            grid-template-columns: minmax(0, 1fr); gap: 0;
            border: 1px solid {BORDER}; border-radius: 12px; background: {CARD_BG};
        }}
        .st-key-home-catalogue .st-key-catalogue-results [class*="st-key-planet-card-"] {{
            grid-template-columns: minmax(8rem, 1fr) minmax(13rem, 2fr) auto;
            grid-template-areas: "name details preview";
            gap: 0.35rem 1rem; padding: 0.45rem 0.85rem;
            border: 0; border-radius: 0; background: transparent;
        }}
        .st-key-home-catalogue .st-key-catalogue-results > [data-testid="stLayoutWrapper"] + [data-testid="stLayoutWrapper"] {{
            border-top: 1px solid {BORDER};
        }}
        .st-key-home-catalogue .catalogue-details {{
            display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
            gap: 0.15rem 0.8rem;
        }}
        .st-key-home-catalogue .catalogue-zone {{ margin: 0; }}
    }}
    @media (max-width: 950px) {{
        .st-key-explorer-navigation .st-key-chart_view [role="radiogroup"] {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
        .st-key-home-overview .st-key-headline-metrics [data-testid="stHorizontalBlock"] {{
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }}
    }}
    @media (max-width: 640px) {{
        [data-testid="stMainBlockContainer"]:has(.st-key-home-toolbar) {{ padding-top: 2.25rem; }}
        .st-key-explorer-navigation {{ top: 2.7rem; padding: 0.45rem 0; }}
        .st-key-explorer-navigation label p {{ font-size: 0.76rem; }}
        .st-key-home-toolbar [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: 0.5rem; }}
        .st-key-home-toolbar [data-testid="stColumn"] {{ min-width: 0 !important; flex: 1 1 0 !important; }}
        .st-key-home-toolbar [data-testid="stColumn"]:first-child {{ flex-basis: 100% !important; }}
        .st-key-site-heading [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap; gap: 0.65rem; }}
        .st-key-site-heading [data-testid="stColumn"] {{ min-width: 0 !important; }}
        .st-key-site-heading [data-testid="stColumn"]:last-child {{ flex: 0 0 5.25rem !important; }}
        .st-key-site-heading [data-testid="stToggle"] p {{ font-size: 0.8rem; }}
        .st-key-home-overview .st-key-headline-metrics [data-testid="stHorizontalBlock"] {{
            gap: 0.75rem;
        }}
        .st-key-home-overview [data-testid="stMetricLabel"] span:has(> span > [data-testid="stMetricIcon"]) {{
            display: none;
        }}
        .st-key-catalogue-results {{ grid-template-columns: minmax(0, 1fr); }}
        [role="dialog"] {{ max-width: 100vw !important; }}
    }}
    @media (max-width: 360px) {{
        .st-key-site-heading .hero-title {{ font-size: 1.38rem !important; }}
    }}
    /* Metric cards. NOTE: the test id is `stMetric`, not `metric-container`. */
    [data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        min-height: 6.75rem;
    }}
    /* Label on its own row; value and delta share the row below it, both
       anchored to the card's bottom edge so cards line up regardless of
       whether a given metric has a delta. Direct-child selectors matter
       here — grid-row/grid-column placement only applies to grid items
       that are immediate children of the grid container. */
    [data-testid="stMetric"] > div {{
        display: grid;
        grid-template-columns: auto auto;
        grid-template-rows: auto 1fr;
        align-items: end;
        column-gap: 0.6rem;
        row-gap: 0.2rem;
        height: 100%;
    }}
    [data-testid="stMetricLabel"] {{
        grid-column: 1 / -1;
        grid-row: 1;
    }}
    [data-testid="stMetricValue"] {{
        grid-row: 2;
        grid-column: 1;
        font-size: 1.9rem;
    }}
    [data-testid="stMetric"] > div > div:has([data-testid="stMetricDelta"]) {{
        grid-row: 2;
        grid-column: 2;
        align-self: end;
    }}
    [data-testid="stMetricDelta"] {{
        display: flex;
        align-items: center;
        gap: 0.15rem;
    }}
    [data-testid="stMetricLabel"] p {{
        font-size: 0.82rem;
        opacity: 0.75;
        letter-spacing: 0.02em;
    }}

    h1.hero-title {{
        color: {TEXT_COLOR};
        font-size: clamp(1.75rem, 4vw, 2.75rem) !important;
        font-weight: 750;
        line-height: 1.15;
        letter-spacing: -0.045em;
        margin: 0;
        padding: 0;
    }}
    .hero-sub {{
        color: {MUTED_TEXT};
        font-size: 0.95rem;
        line-height: 1.45;
        margin: 0.5rem 0 0;
    }}

    /* Sidebar heading, styled rather than using st.header so it can sit
       tight above the filter groups. */
    .sidebar-brand {{
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.1rem;
    }}
    .sidebar-brand-sub {{
        font-size: 0.83rem;
        color: {MUTED_TEXT};
        margin-bottom: 0.9rem;
    }}
    /* Filter groups read as one stack, not four floating boxes. */
    [data-testid="stSidebar"] [data-testid="stExpander"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
        font-weight: 600;
        font-size: 0.92rem;
    }}

    /* Planet profile header. */
    .profile-head {{
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin: 0.5rem 0 0.15rem 0;
    }}
    .profile-name {{
        font-size: 1.85rem;
        font-weight: 750;
        letter-spacing: -0.015em;
        line-height: 1.2;
    }}
    .profile-chip {{
        font-size: 0.78rem;
        font-weight: 650;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        white-space: nowrap;
    }}
    .profile-sub {{
        font-size: 0.97rem;
        color: {MUTED_TEXT};
        margin-bottom: 1.1rem;
    }}

    /* Section sub-headings. */
    [data-testid="stMarkdownContainer"] h5 {{
        margin-top: 1.4rem;
        margin-bottom: 0.35rem;
        font-size: 1.02rem;
        font-weight: 650;
        opacity: 0.9;
    }}

    /* Subtle hover feedback for metric cards. */
    [data-testid="stMetric"] {{
        transition: transform 140ms ease, border-color 140ms ease,
                    box-shadow 140ms ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        border-color: {ACCENT_SOFT};
        box-shadow: 0 6px 18px {SHADOW};
    }}

    /* Five metric cards crush badly on a narrow window; let that one row
       wrap. Scoped by container key so other column layouts are untouched. */
    .st-key-headline-metrics [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap;
        gap: 0.6rem 0.75rem;
    }}
    /* With the sidebar open a 1280px window leaves ~810px of content, so five
       cards get ~149px each. Keep the floor below that or they wrap to 4 + 1;
       genuinely narrow windows still wrap, which is the point. */
    .st-key-headline-metrics [data-testid="stColumn"] {{
        min-width: 140px;
    }}

    /* Buttons and inputs: visible focus ring for keyboard users. */
    button:focus-visible,
    input:focus-visible,
    [data-baseweb="select"] div:focus-visible {{
        outline: 2px solid {FOCUS_RING};
        outline-offset: 2px;
    }}

    /* Links: readable in light mode, not just dark. */
    .hero-sub a {{
        color: {LINK_COLOR};
        text-decoration-color: {ACCENT_SOFT};
        text-underline-offset: 2px;
    }}

    /* Tablets and narrow desktop windows: five cards no longer fit across,
       and letting them fall 4 + 1 looks accidental. Three per row gives a
       deliberate 3 + 2. */
    @media (min-width: 641px) and (max-width: 950px) {{
        .st-key-headline-metrics [data-testid="stColumn"] {{
            flex: 1 1 calc(33.333% - 11px) !important;
            min-width: calc(33.333% - 11px) !important;
        }}
    }}

    {_dark_overrides}

    /* ---------------- Mobile ---------------- */
    @media (max-width: 640px) {{
        /* Streamlit reserves desktop-sized chrome; claim it back. */
        [data-testid="stMainBlockContainer"] {{
            padding-top: 3rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
            padding-bottom: 3rem;
        }}

        /* Streamlit stacks columns on mobile by forcing min-width:100%, which
           turned 5 headline cards into 5 full-width slabs (and 8 profile ones
           into 8). Two compact cards per row instead; the flex row already
           wraps, so only the width floor needs overriding. */
        [data-testid="stColumn"]:has([data-testid="stMetric"]) {{
            flex: 1 1 calc(50% - 8px) !important;
            min-width: calc(50% - 8px) !important;
            width: calc(50% - 8px) !important;
        }}
        /* min-height:0 alone left cards carrying a delta at 107px and the
           rest at 80px, so each 2-up row came out ragged. Stretch every card
           to the height of its row. */
        [data-testid="stColumn"]:has([data-testid="stMetric"]) {{
            align-self: stretch;
        }}
        [data-testid="stColumn"]:has([data-testid="stMetric"]) > div,
        [data-testid="stColumn"]:has([data-testid="stMetric"])
            > div > [data-testid="stElementContainer"] {{
            height: 100%;
        }}
        [data-testid="stMetric"] {{
            min-height: 0;
            height: 100%;
            padding: 0.6rem 0.7rem;
        }}
        [data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
        [data-testid="stMetricLabel"] p {{ font-size: 0.7rem; }}
        [data-testid="stMetricLabel"] p {{
            white-space: normal; overflow: visible; text-overflow: clip;
            line-height: 1.35;
        }}
        [data-testid="stMetricDelta"] {{ font-size: 0.68rem; }}
        /* Stack value over delta: side by side is too tight at this width. */
        [data-testid="stMetric"] > div {{
            grid-template-columns: 1fr;
        }}
        [data-testid="stMetric"] > div > div:has([data-testid="stMetricDelta"]) {{
            grid-row: 3;
            grid-column: 1;
        }}

        /* Sub-headings and section spacing scale down with everything else. */
        [data-testid="stMarkdownContainer"] h5 {{ margin-top: 1rem; }}
        .profile-name {{ font-size: 1.35rem; }}
        .profile-sub {{ font-size: 0.88rem; margin-bottom: 0.8rem; }}
        .profile-head {{ gap: 0.45rem; }}

        /* Tables: scroll inside their own box rather than stretching the page. */
        [data-testid="stDataFrame"] {{ max-width: 100%; overflow-x: auto; }}
        [data-testid="stDataFrame"] > div {{
            max-width: 100%;
            overflow-x: auto;
        }}

        /* Plotly keeps the desktop figure height on narrow screens, which
           makes the first chart dominate the phone viewport. Cap the canvas
           itself rather than only its wrapper so the responsive SVG follows. */
        [data-testid="stPlotlyChart"] {{
            max-width: 100%;
            overflow: hidden;
        }}
        [data-testid="stPlotlyChart"] .js-plotly-plot,
        [data-testid="stPlotlyChart"] .plot-container,
        [data-testid="stPlotlyChart"] .svg-container {{
            height: min(390px, 108vw) !important;
            max-height: 390px !important;
        }}
        [data-testid="stPlotlyChart"] .gl-container {{
            height: min(410px, 114vw) !important;
            max-height: 410px !important;
        }}

        /* Full-width tap targets for the download/link buttons. */
        [data-testid="stDownloadButton"] button,
        [data-testid="stLinkButton"] a {{ width: 100%; }}

        /* Keep filter controls comfortable for thumbs without letting the
           sidebar consume the whole first screen. */
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
            min-height: 44px;
            display: flex;
            align-items: center;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] input {{
            min-height: 44px;
        }}
    }}

    /* Comfortable tap targets on any touch device. */
    @media (hover: none) and (pointer: coarse) {{
        [role="tab"] {{ min-height: 44px; }}
        [data-testid="stMetric"]:hover {{ transform: none; box-shadow: none; }}
    }}

    /* Respect the OS "reduce motion" setting. */
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
            scroll-behavior: auto !important;
        }}
        [data-testid="stMetric"]:hover {{ transform: none; }}
    }}
</style>
""",
    unsafe_allow_html=True,
)

# --- CONSTANTS -----------------------------------------------------------
ARCHIVE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
ARCHIVE_HOME = "https://exoplanetarchive.ipac.caltech.edu/"

RADIUS = "Planet Radius (Earth Radii)"
MASS = "Planet Mass (Earth Mass)"
PERIOD = "Orbital Period (Days)"
TEMP = "Equilibrium Temp (K)"
DISTANCE = "Distance (parsecs)"
YEAR = "Discovery Year"
METHOD = "Discovery Method"
NAME = "Planet Name"
HOST = "Host Star"
METHOD_SHORT = "Method"

# Legend-only aliases. The archive's full wording is kept everywhere the user
# reads or filters values; only the colour key uses these.
SHORT_METHOD = {
    "Transit Timing Variations": "Transit Timing",
    "Eclipse Timing Variations": "Eclipse Timing",
    "Pulsation Timing Variations": "Pulsation Timing",
    "Orbital Brightness Modulation": "Orbital Brightness",
}

# Colour-blind-friendly palette shared by every chart so a discovery method
# keeps the same colour across the whole dashboard.
PALETTE = [
    "#4C9BE8", "#F2836B", "#69C7A5", "#B07AE8", "#F5C065",
    "#4FC3D9", "#E86B9A", "#9BB86A", "#B0895E", "#8C93A8", "#D96BD9",
]

SOLAR_SYSTEM = pd.DataFrame(
    {
        "Name": ["Mercury", "Earth", "Neptune", "Jupiter"],
        "Period": [87.97, 365.25, 60182.0, 4332.59],
        "Radius": [0.383, 1.0, 3.883, 11.209],
        "Color": ["#b0895e", "#2ecc71", "#3498db", "#e67e22"],
    }
)


# --- HABITABLE ZONE & EARTH SIMILARITY -----------------------------------
# Kopparapu et al. (2014) habitable-zone limits. The effective stellar flux
# at each boundary is a quartic in (Teff - 5780 K):
#     Seff = Seff_sun + a*T + b*T^2 + c*T^3 + d*T^4
# The published fit covers 2600-7200 K. We allow down to 2500 K so that late
# M-dwarfs such as TRAPPIST-1 (2566 K) get classified instead of dropped --
# a mild extrapolation, flagged in the UI.
HZ_COEFFS = {
    "recent_venus":       (1.776, 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15),
    "runaway_greenhouse": (1.107, 1.332e-4, 1.580e-8, -8.308e-12, -1.931e-15),
    "maximum_greenhouse": (0.356, 6.171e-5, 1.698e-9, -3.198e-12, -5.575e-16),
    "early_mars":         (0.320, 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16),
}
HZ_TEFF_MIN, HZ_TEFF_MAX = 2500.0, 7200.0
EARTH_SYM = "⊕"  # the Earth symbol used in unit labels
EARTH_EQ_TEMP = 255.0  # Earth's *equilibrium* temperature, not its 288 K surface

STAR_TEMP = "Stellar Effective Temp (K)"
STAR_LUM = "Stellar Luminosity (log(L/L_sun))"
SEMI_MAJOR = "Semi-Major Axis (au)"
HZ_STATUS = "Habitable Zone"
ESI = "Earth Similarity Index"
INSOLATION = "Insolation (S-Earth)"
HZ_INNER = "HZ Inner (au)"
HZ_OUTER = "HZ Outer (au)"
HZ_INNER_OPT = "HZ Inner Optimistic (au)"
HZ_OUTER_OPT = "HZ Outer Optimistic (au)"

HZ_ORDER = ["Conservative", "Optimistic", "Too hot", "Too cold", "Unknown"]
HZ_COLORS = {
    "Conservative": "#22c55e",
    "Optimistic": "#a3c53a",
    "Too hot": "#f97316",
    "Too cold": "#38bdf8",
    "Unknown": "#94a3b8",
}
def hz_slug(status: str) -> str:
    """CSS-safe suffix for a habitable-zone status name."""
    return status.lower().replace(" ", "-")


# One chip class per status. These live in our stylesheet rather than in an
# inline style attribute: Streamlit's sanitiser strips inline `!important`
# colours, and the dark-mode rules repaint every <span>, so an inline colour
# would be overridden in dark mode.
st.markdown(
    "<style>"
    + "".join(
        f".profile-chip.chip-{hz_slug(_name)}{{"
        f"color:{_color} !important;"
        f"background:{_color}1f;"
        f"border:1px solid {_color}59;}}"
        for _name, _color in HZ_COLORS.items()
    )
    + "</style>",
    unsafe_allow_html=True,
)

HZ_HELP = (
    "Kopparapu et al. (2014) limits from the star's luminosity and temperature. "
    "**Conservative** = between the runaway- and maximum-greenhouse limits; "
    "**Optimistic** = the wider recent-Venus to early-Mars band."
)


def _hz_flux(teff, boundary: str):
    """Effective stellar flux (relative to Earth's) at one HZ boundary."""
    s, a, b, c, d = HZ_COEFFS[boundary]
    t = teff - 5780.0
    return s + a * t + b * t**2 + c * t**3 + d * t**4


def add_habitability_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive insolation, Kopparapu habitable-zone class, and a two-parameter
    Earth Similarity Index from the stellar and orbital columns."""
    teff = df[STAR_TEMP]
    lum = 10 ** df[STAR_LUM]  # the archive stores log10(L / L_sun)
    a_au = df[SEMI_MAJOR].where(df[SEMI_MAJOR] > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        df[INSOLATION] = lum / a_au**2
        # np.maximum guards the sqrt: the quartic can go negative outside the
        # temperature range the fit was built for.
        bounds = {
            key: np.sqrt(np.maximum(lum / _hz_flux(teff, key), 0)) for key in HZ_COEFFS
        }

    df[HZ_INNER] = bounds["runaway_greenhouse"]
    df[HZ_OUTER] = bounds["maximum_greenhouse"]
    df[HZ_INNER_OPT] = bounds["recent_venus"]
    df[HZ_OUTER_OPT] = bounds["early_mars"]

    usable = teff.between(HZ_TEFF_MIN, HZ_TEFF_MAX) & lum.notna() & a_au.notna()
    inner_c, outer_c = bounds["runaway_greenhouse"], bounds["maximum_greenhouse"]
    inner_o, outer_o = bounds["recent_venus"], bounds["early_mars"]

    status = pd.Series("Unknown", index=df.index, dtype=object)
    status[usable & (a_au < inner_o)] = "Too hot"
    status[usable & (a_au > outer_o)] = "Too cold"
    status[usable & (a_au >= inner_o) & (a_au < inner_c)] = "Optimistic"
    status[usable & (a_au > outer_c) & (a_au <= outer_o)] = "Optimistic"
    status[usable & (a_au >= inner_c) & (a_au <= outer_c)] = "Conservative"
    df[HZ_STATUS] = status

    df[METHOD_SHORT] = df[METHOD].map(SHORT_METHOD).fillna(df[METHOD])

    # Two-parameter Earth Similarity Index (radius + equilibrium temperature),
    # exponents from Schulze-Makuch et al. (2011).
    radius, temp = df[RADIUS].where(df[RADIUS] > 0), df[TEMP].where(df[TEMP] > 0)
    with np.errstate(invalid="ignore"):
        esi_r = (1 - ((radius - 1.0) / (radius + 1.0)).abs()) ** 0.57
        esi_t = (1 - ((temp - EARTH_EQ_TEMP) / (temp + EARTH_EQ_TEMP)).abs()) ** 0.70
        df[ESI] = (esi_r * esi_t) ** (1 / 1.27)
    return df


# --- DATA ----------------------------------------------------------------
# Any edit to the habitability maths changes this digest, which is part of the
# cache key below. Without it a warm 24-hour cache would keep serving rows
# computed by the *previous* version of the code -- which, on a deploy, means
# a KeyError for every user whose cache is still warm.
HAB_VERSION = hashlib.sha1(
    "".join(
        [
            inspect.getsource(_hz_flux),
            inspect.getsource(add_habitability_columns),
            repr(sorted(HZ_COEFFS.items())),
            repr((HZ_TEFF_MIN, HZ_TEFF_MAX, EARTH_EQ_TEMP)),
            repr(sorted(SHORT_METHOD.items())),
        ]
    ).encode()
).hexdigest()[:12]


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_exoplanet_table(_sink=None) -> pd.DataFrame:
    """Fetch the Planetary Systems Composite Parameters table from NASA.

    `_sink` is an optional plain dict that receives progress as it goes. It
    must never be a Streamlit element: Streamlit replays whatever a cached
    function draws, and replaying a widget bound to a container from an
    earlier run raises CacheReplayClosureError. The leading underscore also
    keeps the dict out of the cache key.
    """

    def report(message: str) -> None:
        if _sink is not None:
            _sink["message"] = message

    query = """
    select pl_name, hostname, discoverymethod, disc_year, pl_rade, pl_bmasse,
           pl_orbper, pl_orbsmax, pl_eqt, st_teff, st_lum, sy_dist,
           pl_orbeccen, ra, dec, disc_facility, disc_pubdate,
           pl_radeerr1, pl_radeerr2, pl_bmasseerr1, pl_bmasseerr2,
           pl_orbpererr1, pl_orbpererr2, pl_eqterr1, pl_eqterr2,
           pl_orbsmaxerr1, pl_orbsmaxerr2, sy_disterr1, sy_disterr2,
           pl_rade_reflink, pl_bmasse_reflink, pl_orbper_reflink,
           pl_eqt_reflink, pl_orbsmax_reflink, sy_dist_reflink
    from pscomppars
    """
    report("Opening a connection to the archive…")
    response = None
    for attempt in range(3):
        try:
            response = requests.get(
                ARCHIVE_URL,
                params={"query": query, "format": "csv"},
                timeout=(10, 120),  # (connect, read) — without this the app can hang forever
                stream=True,
            )
            response.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError) as exc:
            if isinstance(exc, requests.exceptions.ProxyError) or attempt == 2:
                LOGGER.exception("NASA archive request failed after %d attempts", attempt + 1)
                raise
            delay = 2 ** attempt
            LOGGER.warning(
                "NASA archive request failed on attempt %d; retrying in %ds: %s",
                attempt + 1,
                delay,
                exc,
            )
            time.sleep(delay)
        except requests.HTTPError as exc:
            if response is None or response.status_code < 500 or attempt == 2:
                LOGGER.exception("NASA archive returned an unrecoverable HTTP error")
                raise
            response.close()
            delay = 2 ** attempt
            LOGGER.warning(
                "NASA archive returned HTTP %s; retrying in %ds",
                response.status_code,
                delay,
            )
            time.sleep(delay)
    if response is None:
        raise RuntimeError("The NASA archive request did not produce a response.")

    # The server runs the query before it sends anything, so the first bytes
    # arrive several seconds in; that wait is covered by the step above.
    report("Query accepted — receiving the table…")
    parts, received, last_tick = [], 0, 0.0
    with response:
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            parts.append(chunk)
            received += len(chunk)
            now = time.monotonic()
            # Throttle: the table arrives in thousands of chunks and every update
            # is a websocket round trip.
            if now - last_tick >= 0.12:
                last_tick = now
                report(f"Receiving… {received / 1024:,.0f} KB")

    report(f"Received {received / 1024:,.0f} KB — parsing…")
    text = b"".join(parts).decode(response.encoding or "utf-8", errors="replace")
    if not text.lstrip().lower().startswith("pl_name"):
        # TAP reports query errors with a 200 and a plain-text body.
        raise RuntimeError(f"Archive returned an unexpected response: {text[:300]}")

    fetched_at = datetime.now(timezone.utc)
    frame = parse_archive(text)
    try:
        archive_cache.write(text, fetched_at)
    except OSError:
        LOGGER.warning("Could not write the archive snapshot", exc_info=True)
    frame.attrs["fetched_at"] = fetched_at.isoformat()
    return frame


def parse_archive(text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(text))
    validate_archive_columns(df.columns)
    if df.empty:
        raise RuntimeError("The archive returned an empty table.")

    df = df.rename(
        columns={
            "pl_name": NAME,
            "hostname": HOST,
            "discoverymethod": METHOD,
            "disc_year": YEAR,
            "pl_rade": RADIUS,
            "pl_bmasse": MASS,
            "pl_orbper": PERIOD,
            "pl_orbsmax": "Semi-Major Axis (au)",
            "pl_eqt": TEMP,
            "st_teff": "Stellar Effective Temp (K)",
            "st_lum": "Stellar Luminosity (log(L/L_sun))",
            "sy_dist": DISTANCE,
        }
    )

    years = pd.to_numeric(df[YEAR], errors="coerce")
    df[YEAR] = years.where(years.between(1900, datetime.now(timezone.utc).year + 1) & years.mod(1).eq(0)).astype("Int64")
    df = df.dropna(subset=[NAME, HOST, YEAR]).drop_duplicates(NAME).reset_index(drop=True)
    if df.empty:
        raise ValueError("The archive contains no usable planet records.")
    df[METHOD] = df[METHOD].fillna("Unknown")
    for column in ("disc_facility", "disc_pubdate"):
        if column not in df:
            df[column] = pd.NA
    for column in (RADIUS, MASS, PERIOD, SEMI_MAJOR, TEMP, STAR_TEMP, STAR_LUM, DISTANCE, "ra", "dec", "pl_orbeccen"):
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def load_exoplanet_data(version: str = HAB_VERSION, _sink=None) -> pd.DataFrame:
    """The archive table plus the derived habitability columns.

    Split from the fetch so the slow network call survives a change to the
    habitability maths, while the derived columns never go stale. `version` is
    unused inside the body -- it exists purely to key the cache, and callers
    must pass it positionally: Streamlit hashes only the arguments actually
    supplied, so relying on the default would silently disable versioning.
    """
    df = fetch_exoplanet_table(_sink=_sink)
    if _sink is not None:
        _sink["message"] = "Computing habitable zones…"
    df = add_habitability_columns(df)
    if _sink is not None:
        _sink["message"] = f"Ready: {len(df):,} planets"
    return df


@st.cache_data(show_spinner=False)
def cluster_planets(features: pd.DataFrame, n_clusters: int) -> pd.Series:
    """K-Means on log10(mass, radius), with clusters renumbered by median mass.

    Raw K-Means labels are arbitrary and shift whenever the data changes, so
    they can never be mapped to fixed names. Sorting by median mass makes the
    numbering (and therefore the naming and colouring) reproducible.
    """
    scaled = StandardScaler().fit_transform(np.log10(features.to_numpy()))
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)

    median_mass = [np.median(features.iloc[labels == i, 0]) for i in range(n_clusters)]
    rank = {old: new for new, old in enumerate(np.argsort(median_mass))}
    return pd.Series([rank[label] for label in labels], index=features.index)


def describe_cluster(median_radius: float, median_mass: float) -> str:
    """Plain-language name for a cluster, keyed off its median radius."""
    if median_radius < 1.25:
        return "Terrestrial"
    if median_radius < 2.0:
        return "Super-Earths"
    if median_radius < 4.0:
        return "Sub-Neptunes"
    if median_radius < 8.0:
        return "Neptunians"
    if median_mass < 1000:
        return "Gas giants"
    return "Super-Jupiters"


def method_colors(methods) -> dict:
    """Stable colour per discovery method, independent of the active filters."""
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(methods))}


def transparent(fig, **layout):
    """Transparent chart background with axis/legend text tuned to the active
    page skin — chart text is baked into the figure, so it doesn't follow the
    dark-mode CSS overrides the way native Streamlit chrome does."""
    # Flat, top-level magic-underscore kwargs — passing the equivalent as a
    # nested dict value (e.g. xaxis=dict(title=dict(font=dict(...)))) silently
    # fails to reach axis title/tick fonts in this Plotly version.
    for axis in ("xaxis", "yaxis"):
        layout.setdefault(f"{axis}_gridcolor", CHART_GRID)
        layout.setdefault(f"{axis}_zerolinecolor", CHART_GRID)
        layout.setdefault(f"{axis}_linecolor", CHART_GRID)
        layout.setdefault(f"{axis}_tickfont_color", CHART_FONT)
        layout.setdefault(f"{axis}_title_font_color", CHART_FONT)
    # A vertical legend on the right steals width from the plot -- ruinous at
    # 348px. Lay the colour key out horizontally under the chart instead, and
    # reserve bottom margin for it (Plotly does not grow the margin itself for
    # an out-of-plot legend).
    if layout.get("showlegend", True):
        # Above the plot rather than below it: underneath, Plotly's automargin
        # packs the key hard against the x-axis title with a fixed ~4px gap
        # that no margin or standoff setting will widen.
        layout.setdefault("legend_orientation", "h")
        layout.setdefault("legend_yanchor", "bottom")
        layout.setdefault("legend_y", 1.09)
        layout.setdefault("legend_xanchor", "left")
        layout.setdefault("legend_x", 0)
        layout.setdefault("legend_font_size", 10)
        # A proportional entry width lays the keys out on a grid instead of
        # letting them flow ragged -- 11 methods previously wrapped into seven
        # rows of one to three items. Left on paper refs so automargin still
        # reserves the space above the plot.
        layout.setdefault("legend_entrywidthmode", "fraction")
        # 0.45 of the legend width: wide enough that the longest label
        # ("Orbital Brightness") clears the next column, narrow enough to
        # still lay out in two columns rather than one tall stack.
        layout.setdefault("legend_entrywidth", 0.45)
        layout.setdefault("legend_itemsizing", "constant")

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=CHART_FONT,
        legend_title_text="",
        legend_font_color=CHART_FONT,
        **layout,
    )
    return fig


CHART_CONFIG = {
    "displayModeBar": True, "displaylogo": False, "scrollZoom": False,
    "responsive": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "exoplanet-explorer", "scale": 2},
}

# Plotly bakes a pixel height into the figure and CSS cannot resize the SVG,
# so these are single values chosen to stay comfortable on a phone while still
# looking generous on a desktop.
CHART_H_TALL = 460
CHART_H_SCATTER = 510
CHART_H_3D = 520


# --- HEADER --------------------------------------------------------------
with st.container(key="site-heading"):
    brand, theme = st.columns([5, 1.5], vertical_alignment="center")
    brand.html(
        '<header><h1 class="hero-title">Exoplanet <span class="brand-accent">Explorer</span></h1>'
        f'<p class="hero-sub">Confirmed planets from the <a href="{ARCHIVE_HOME}">NASA Exoplanet Archive</a>.</p></header>'
    )
    theme.toggle("Dark", key="dark_mode")


def load_with_progress():
    """Fetch on a worker thread while the script updates the loading status.

    Every Streamlit call stays here on the script thread; the worker only
    updates the message in `shared`.
    """
    shared = {"message": "Contacting the archive…"}
    result = {}

    def worker():
        try:
            # HAB_VERSION must be passed explicitly: Streamlit keys the
            # cache on the arguments actually supplied, so a default value
            # never reaches the key and the version would do nothing.
            result["df"] = load_exoplanet_data(HAB_VERSION, _sink=shared)
        except BaseException as exc:  # surfaced on this thread below
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    add_script_run_ctx(thread)  # the worker needs the ctx to reach the cache
    thread.start()
    # A cache hit finishes inside this first wait, so nothing is ever drawn
    # and the page does not flicker on ordinary reruns.
    thread.join(timeout=0.4)

    holder = None
    while thread.is_alive():
        if holder is None:
            status = st.status(
                "Fetching live data from the NASA Exoplanet Archive", expanded=True
            )
            status.caption(
                "The archive runs the query before it starts sending, so the "
                "first few seconds are the server working rather than the "
                "download."
            )
            holder = status.empty()
        holder.caption(str(shared["message"]))
        thread.join(timeout=0.1)

    if "error" in result:
        raise result["error"]
    return result["df"]


loader = st.empty()
archive_problem = None
try:
    last_failure = st.session_state.get("archive_failure")
    if last_failure and time.monotonic() - last_failure[0] < 90:
        raise last_failure[1]
    with loader.container():
        df = load_with_progress()
    st.session_state.pop("archive_failure", None)
except Exception as exc:
    if not last_failure or exc is not last_failure[1]:
        st.session_state["archive_failure"] = (time.monotonic(), exc)
    archive_problem = exc
    try:
        cached_csv, fetched_at = archive_cache.read()
        df = add_habitability_columns(parse_archive(cached_csv))
        df.attrs["fetched_at"] = fetched_at.isoformat()
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        df = None
loader.empty()


def refresh_archive():
    st.session_state.pop("archive_failure", None)
    fetch_exoplanet_table.clear()
    load_exoplanet_data.clear()


if archive_problem is not None:
    if isinstance(archive_problem, requests.Timeout):
        problem = "The NASA Exoplanet Archive did not respond in time."
    elif isinstance(archive_problem, requests.exceptions.ProxyError):
        problem = "The network proxy could not connect to the NASA Exoplanet Archive."
    elif isinstance(archive_problem, requests.RequestException):
        problem = "Could not load data from the NASA Exoplanet Archive."
    else:
        problem = "Could not read the latest NASA catalogue."
    if df is None:
        st.error(problem)
    else:
        age = datetime.fromisoformat(str(df.attrs["fetched_at"])).strftime("%d %b %Y, %H:%M UTC")
        st.warning(f"{problem} Showing a saved NASA snapshot fetched {age}.")
    st.button("Try again", type="primary", on_click=refresh_archive)
    with st.expander("Connection details"):
        st.text(type(archive_problem).__name__)
    if df is None:
        st.stop()

LOGGER.info("Dashboard data ready: %d planets", len(df))

planet_names = frozenset(df[NAME])
COLOR_MAP = method_colors(df[METHOD_SHORT].unique())
YEAR_MIN, YEAR_MAX = int(df[YEAR].min()), int(df[YEAR].max())
ALL_METHODS = sorted(df[METHOD].unique())
MAX_DIST = max(50.0, float(np.ceil(df[DISTANCE].max() / 100) * 100))

FILTER_DEFAULTS = {
    "year_range": (YEAR_MIN, YEAR_MAX),
    "methods": [],
    "dist_limit": MAX_DIST,
    "keep_unknown_dist": True,
    "hz_filter": [],
    "esi_floor": 0.0,
    "complete_only": False,
}
VIEWS = ["Explore", "Saved", "Trends", "Orbits", "Classes", "3D", "Habitable", "Sky", "Profile", "Learn"]
CORE_COLUMNS = [RADIUS, MASS, PERIOD, TEMP, DISTANCE, SEMI_MAJOR]
CORE_LABELS = ["Radius", "Mass", "Orbital period", "Equilibrium temperature", "Distance", "Semi-major axis"]
COMPARISON_LIMIT = 4
DISTANCE_SCALE = {"Light-years": 3.26156, "Parsecs": 1.0}
DISTANCE_SUFFIX = {"Light-years": "ly", "Parsecs": "pc"}
SIZE_SCALE = {"Earth": (1.0, 1.0, "⊕"), "Jupiter": (11.209, 317.828, "♃")}
distance_factor = DISTANCE_SCALE.get(st.session_state["distance_unit"], 3.26156)
distance_suffix = DISTANCE_SUFFIX.get(st.session_state["distance_unit"], "ly")
radius_divisor, mass_divisor, size_suffix = SIZE_SCALE.get(st.session_state["size_unit"], SIZE_SCALE["Earth"])


def measurement(value, spec, suffix=""):
    return "—" if pd.isna(value) else f"{value:{spec}}{suffix}"


def zone_label(status):
    return {
        "Conservative": "Conservative zone", "Optimistic": "Optimistic zone",
        "Too hot": "Inside inner limit", "Too cold": "Beyond outer limit",
        "Unknown": "Zone unknown",
    }[status]


route_visit = browser.route is not None or "route_initialized" not in st.session_state
route_query = ({key: values[-1] for key, values in parse_qs(browser.route.lstrip("?")).items()}
               if isinstance(browser.route, str) else dict(st.query_params))


def _query_text(key: str) -> Optional[str]:
    """Read one scalar query parameter without trusting malformed URLs."""
    value = route_query.get(key)
    return value if isinstance(value, str) and value else None


def _comparison_from_query() -> list[str]:
    """Read the shared tray in order, keeping only known, unique planet names."""
    encoded = _query_text("tray")
    if not encoded:
        return []
    try:
        names = json.loads(encoded)
    except (ValueError, RecursionError):
        return []
    if not isinstance(names, list):
        return []
    known_names = dict.fromkeys(name for name in names if isinstance(name, str) and name in planet_names)
    return list(known_names)[:COMPARISON_LIMIT]


def _filters_from_query() -> dict:
    """Decode shareable filters, falling back safely for invalid values."""
    parsed = {}
    year_text = _query_text("years")
    if year_text and "-" in year_text:
        try:
            start, end = (int(part) for part in year_text.split("-", 1))
            parsed["year_range"] = (start, end)
        except ValueError:
            pass
    methods = _query_text("methods")
    if methods:
        parsed["methods"] = [item for item in methods.split(",") if item]
    distance = _query_text("distance")
    if distance:
        try:
            parsed["dist_limit"] = float(distance)
        except ValueError:
            pass
    keep_unknown = _query_text("unknown")
    if keep_unknown in {"0", "1"}:
        parsed["keep_unknown_dist"] = keep_unknown == "1"
    hz = _query_text("hz")
    if hz:
        parsed["hz_filter"] = [item for item in hz.split(",") if item]
    esi = _query_text("esi")
    if esi:
        try:
            parsed["esi_floor"] = float(esi)
        except ValueError:
            pass
    parsed["complete_only"] = _query_text("complete") == "1"
    return parsed


def _sync_filters_to_query() -> None:
    """Keep the current slice bookmarkable without adding default noise."""
    params = {}
    if tuple(st.session_state["year_range"]) != tuple(FILTER_DEFAULTS["year_range"]):
        params["years"] = "-".join(str(value) for value in st.session_state["year_range"])
    if st.session_state["methods"] and set(st.session_state["methods"]) != set(ALL_METHODS):
        params["methods"] = ",".join(st.session_state["methods"])
    if st.session_state["dist_limit"] != FILTER_DEFAULTS["dist_limit"]:
        params["distance"] = f'{st.session_state["dist_limit"]:.2f}'.rstrip("0").rstrip(".")
    if st.session_state["keep_unknown_dist"] != FILTER_DEFAULTS["keep_unknown_dist"]:
        params["unknown"] = "1" if st.session_state["keep_unknown_dist"] else "0"
    if st.session_state["hz_filter"]:
        params["hz"] = ",".join(st.session_state["hz_filter"])
    if st.session_state["esi_floor"] > 0:
        params["esi"] = f'{st.session_state["esi_floor"]:.2f}'
    if st.session_state["complete_only"]:
        params["complete"] = "1"
    params["view"] = st.session_state.get("active_view", "Explore")
    if st.session_state.get("profile_pick") in planet_names:
        params["planet"] = st.session_state["profile_pick"]
    if st.session_state.get("profile_rival") in planet_names:
        params["compare"] = st.session_state["profile_rival"]
    if st.session_state["comparison_tray"]:
        params["tray"] = json.dumps(st.session_state["comparison_tray"], ensure_ascii=False, separators=(",", ":"))
    st.session_state["route_params"] = params
    with browser_route:
        route_writer(
            key="route_writer",
            data={"params": params, "replace": route_visit,
                  "managed": ["years", "methods", "distance", "unknown", "hz",
                              "esi", "complete", "view", "planet", "compare", "tray"]},
        )



def reset_filters() -> None:
    """Restore every filter to its default.

    This has to run as a button callback: assigning to widget keys during the
    normal script body (or popping them and rerunning) leaves the rendered
    slider thumbs showing their old positions.
    """
    st.session_state.update(FILTER_DEFAULTS)


# Seed session state once, then build the widgets from `key` alone. Passing
# both a `value=`/`default=` and a session-state entry makes Streamlit log a
# warning on every rerun that follows a reset.
_query_filters = _filters_from_query()
if route_visit:
    st.session_state["route_initialized"] = True
    st.session_state.pop("card_preview_planet", None)
    st.session_state.update({
        key: _query_filters.get(key, default) for key, default in FILTER_DEFAULTS.items()
    })
    st.session_state["active_view"] = _query_text("view") or "Explore"
    st.session_state["comparison_tray"] = _comparison_from_query()
    for state_key, parameter in [("profile_pick", "planet"), ("profile_rival", "compare")]:
        st.session_state[state_key] = _query_text(parameter)
else:
    for _key, _default in FILTER_DEFAULTS.items():
        st.session_state.setdefault(_key, _default)
    st.session_state["comparison_tray"] = [
        name for name in st.session_state.get("comparison_tray", []) if name in planet_names
    ]
st.session_state.update(st.session_state.pop("pending_filters", {}))

# A data refresh can move the bounds under stored state (a new discovery year,
# a newly catalogued far-away planet), so re-fit what is already there.
_lo, _hi = st.session_state["year_range"]
_lo, _hi = sorted((min(YEAR_MAX, max(YEAR_MIN, _lo)), min(YEAR_MAX, max(YEAR_MIN, _hi))))
st.session_state["year_range"] = (_lo, _hi)
_distance = st.session_state["dist_limit"]
st.session_state["dist_limit"] = min(max(0.0, _distance), MAX_DIST) if np.isfinite(_distance) else MAX_DIST
_esi = st.session_state["esi_floor"]
st.session_state["esi_floor"] = min(max(0.0, _esi), 1.0) if np.isfinite(_esi) else 0.0
st.session_state["methods"] = [m for m in st.session_state["methods"] if m in ALL_METHODS]
st.session_state["hz_filter"] = [h for h in st.session_state["hz_filter"] if h in HZ_ORDER]
st.session_state.setdefault("active_view", _query_text("view") or "Explore")
if st.session_state["active_view"] not in VIEWS:
    st.session_state["active_view"] = "Explore"
# Keep profile selections when their widgets are absent from the current page.
for _state_key in ("active_view", "profile_pick", "profile_rival", "orbit_period_limit",
                   "catalogue_sort", "catalogue_collection", "catalogue_columns",
                   "catalogue_search", "saved_search", "saved_collection"):
    if _state_key in st.session_state:
        st.session_state[_state_key] = st.session_state[_state_key]


def open_profile(name, clear_filters=False):
    """Keep the current selection when opening a planet from it."""
    if clear_filters:
        reset_filters()
    st.session_state.pop("card_preview_planet", None)
    current_view = st.session_state["active_view"]
    if current_view != "Profile":
        st.session_state["profile_return_view"] = "Saved" if current_view == "Saved" else "Explore"
    st.session_state["profile_pick"] = name
    st.session_state["highlighted_planet"] = name
    st.session_state["active_view"] = "Profile"


def search_planet():
    if st.session_state.get("global_search"):
        open_profile(st.session_state["global_search"], clear_filters=True)
        st.session_state["global_search"] = None


def apply_preset(preset):
    reset_filters()
    if preset == "Habitable candidates":
        st.session_state["hz_filter"] = ["Conservative", "Optimistic"]
    elif preset == "Nearby worlds":
        st.session_state["dist_limit"] = min(50.0, MAX_DIST)
        st.session_state["keep_unknown_dist"] = False
    elif preset == "Recent discoveries":
        st.session_state["year_range"] = (max(YEAR_MIN, YEAR_MAX - 2), YEAR_MAX)
    elif preset == "Short years":
        st.session_state["active_view"] = "Orbits"
        st.session_state["orbit_period_limit"] = 87.97
        return
    st.session_state["active_view"] = "Explore"


def render_guided_discoveries(source):
    st.markdown("**Shorter than Mercury**")
    st.caption("Find a planet with an orbit shorter than Mercury’s 87.97-day year.")
    st.button("Explore short orbits", on_click=apply_preset, args=("Short years",),
              key=f"short-orbits-{source}")
    st.markdown("**Look for neighbours**")
    st.caption("Nearby worlds uses a distance limit of 50 parsecs.")
    st.markdown("**Explore a multi-planet system**")
    st.caption("Choose Multi-planet systems under Options, then open a profile to compare the orbits.")


def render_discovery_shortcuts(source):
    for preset in ("Habitable candidates", "Nearby worlds", "Recent discoveries"):
        st.button(preset, on_click=apply_preset, args=(preset,), width="stretch",
                  key=f"preset-{source}-{preset}")


def render_global_search():
    st.selectbox(
        "Find a planet", options=sorted(planet_names), index=None,
        placeholder="Search all planets",
        key="global_search", on_change=search_planet,
        help="Search the entire archive. Opening a planet resets filters.",
    )


if st.session_state["active_view"] != "Explore":
    render_global_search()


# --- FILTERS AND NAVIGATION ----------------------------------------------
def filter_fields(prefix="", values=None):
    values = values or st.session_state
    keyed = lambda name: prefix + name
    # Live sidebar values come from their widget keys. The drawer stages a
    # separate copy and applies it in one full rerun on submit.
    def initial(name):
        return {"value": values[name]} if prefix else {}

    filters = {}
    with st.expander("Discovery", expanded=True):
        filters["year_range"] = (
            st.slider("Discovery year", YEAR_MIN, YEAR_MAX, key=keyed("year_range"), **initial("year_range"))
            if YEAR_MIN < YEAR_MAX else (YEAR_MIN, YEAR_MAX)
        )
        filters["methods"] = st.multiselect(
            "Discovery method", ALL_METHODS, key=keyed("methods"),
            **({"default": values["methods"]} if prefix else {}),
        )
    with st.expander("Distance", expanded=True):
        distance_label = f"Max distance ({distance_suffix})"
        if prefix:
            displayed = st.number_input(distance_label, min_value=0.0,
                                        max_value=MAX_DIST * distance_factor,
                                        value=values["dist_limit"] * distance_factor,
                                        key=keyed("display_distance"), step=1.0)
            filters["dist_limit"] = min(MAX_DIST, displayed / distance_factor)
        else:
            # Keep the canonical filter and shared URLs in parsecs.
            filters["dist_limit"] = st.slider(
                "Max distance (parsecs)", 0.0, MAX_DIST, step=50.0, key="dist_limit",
            )
        if prefix:
            excluded = st.checkbox("Exclude planets with unknown distance",
                                   key=keyed("exclude_unknown"), value=not values["keep_unknown_dist"])
            filters["keep_unknown_dist"] = not excluded
        else:
            st.session_state["exclude_unknown"] = not values["keep_unknown_dist"]

            def update_distance_coverage():
                st.session_state["keep_unknown_dist"] = not st.session_state["exclude_unknown"]

            st.checkbox("Exclude planets with unknown distance", key="exclude_unknown",
                        on_change=update_distance_coverage)
            filters["keep_unknown_dist"] = values["keep_unknown_dist"]
    with st.expander("Habitability and coverage"):
        filters["hz_filter"] = st.multiselect(
            "Habitable-zone class", HZ_ORDER, key=keyed("hz_filter"), help=HZ_HELP,
            **({"default": values["hz_filter"]} if prefix else {}),
        )
        filters["esi_floor"] = st.slider(
            "Minimum Earth Similarity Index", 0.0, 1.0, step=0.05,
            key=keyed("esi_floor"), **initial("esi_floor"),
        )
        filters["complete_only"] = st.checkbox(
            "Require all six core measurements", key=keyed("complete_only"),
            help=", ".join(CORE_LABELS), **initial("complete_only"),
        )
    return filters


def save_filter_set():
    name = st.session_state.get("filter_name", "").strip()
    if name:
        st.session_state["filter_sets"] = {
            **st.session_state["filter_sets"],
            name: {key: st.session_state[key] for key in FILTER_DEFAULTS},
        }


def restore_filter_set():
    preset = st.session_state["named_filter"]
    if preset in st.session_state["filter_sets"]:
        st.session_state.update(st.session_state["filter_sets"][preset])


def dismiss_filters():
    st.session_state["filter_dialog_open"] = False


@st.dialog("Filters", position="right", on_dismiss=dismiss_filters)
def filters_drawer():
    revision = st.session_state.get("drawer_revision", 0)
    with st.form(f"filter_form_{revision}"):
        staged = filter_fields(f"draft_{revision}_", st.session_state)
        apply, reset = st.columns(2)
        with apply:
            submitted = st.form_submit_button("Apply filters", type="primary", width="stretch")
        with reset:
            cleared = st.form_submit_button("Reset filters", width="stretch")
    if submitted or cleared:
        st.session_state["pending_filters"] = FILTER_DEFAULTS if cleared else staged
        dismiss_filters()
        st.rerun()


def show_filters():
    st.session_state["drawer_revision"] = st.session_state.get("drawer_revision", 0) + 1
    st.session_state["filter_dialog_open"] = True


with st.sidebar:
    st.subheader("Filters")
    if st.session_state["filter_sets"]:
        st.selectbox("Saved filters", sorted(st.session_state["filter_sets"]), index=None,
                     key="named_filter", on_change=restore_filter_set)
        if st.session_state["named_filter"] and st.button("Delete saved filter"):
            del st.session_state["filter_sets"][st.session_state["named_filter"]]
            st.rerun()
    live_filters = filter_fields()
    st.button("Reset filters", width="stretch", on_click=reset_filters)
    with st.expander("Save this filter set"):
        st.text_input("Filter set name", key="filter_name", max_chars=40)
        st.button("Save filter set", on_click=save_filter_set)
    with st.expander("Display preferences"):
        st.radio("Distance", list(DISTANCE_SCALE), key="distance_unit", horizontal=True)
        st.radio("Planet size and mass", list(SIZE_SCALE), key="size_unit", horizontal=True)
    st.button("Refresh data from NASA", width="stretch", on_click=refresh_archive)

CHART_VIEWS = ["Trends", "Orbits", "Classes", "3D", "Habitable", "Sky"]
NAV_GROUPS = ["Explore", "Charts", "Saved", "Learn"]


def navigate_chart():
    st.session_state["active_view"] = st.session_state["chart_view"]
    st.session_state["last_chart"] = st.session_state["chart_view"]


def navigate_group():
    group = st.session_state["nav_group"]
    st.session_state["active_view"] = st.session_state.get("last_chart", "Trends") if group == "Charts" else group


view = st.session_state["active_view"]
group = "Charts" if view in CHART_VIEWS else "Explore" if view == "Profile" else view
st.session_state["nav_group"] = group
with st.container(key="explorer-navigation"):
    st.radio("Main navigation", NAV_GROUPS, key="nav_group", horizontal=True,
             label_visibility="collapsed", width="stretch", on_change=navigate_group)
    if group == "Charts":
        st.session_state["chart_view"] = view
        st.radio("Chart view", CHART_VIEWS, key="chart_view", horizontal=True,
                 label_visibility="collapsed", width="stretch", on_change=navigate_chart)
    if view == "Profile":
        return_view = st.session_state.get("profile_return_view", "Explore")
        return_label = "Back to saved planets" if return_view == "Saved" else "Back to catalogue"
        st.button(return_label, on_click=lambda: st.session_state.update(active_view=return_view))

year_range = live_filters["year_range"]
selected_methods = live_filters["methods"]
dist_limit = live_filters["dist_limit"]
keep_unknown_dist = live_filters["keep_unknown_dist"]
hz_filter = live_filters["hz_filter"]
esi_floor = live_filters["esi_floor"]

# Apply the global filters.
methods_filter = selected_methods or ALL_METHODS
mask = (
    df[YEAR].between(year_range[0], year_range[1])
    & df[METHOD].isin(methods_filter)
)
within_distance = df[DISTANCE] <= dist_limit
if keep_unknown_dist:
    within_distance |= df[DISTANCE].isna()
mask &= within_distance
if hz_filter:
    mask &= df[HZ_STATUS].isin(hz_filter)
if esi_floor > 0:
    # NaN >= x is False, so unscored planets drop out here by design.
    mask &= df[ESI] >= esi_floor
if live_filters["complete_only"]:
    mask &= df[CORE_COLUMNS].notna().all(axis=1)
fdf = df[mask].copy()

# Human-readable summary of everything currently narrowing the dataset.
active_filters = []
if tuple(year_range) != (YEAR_MIN, YEAR_MAX):
    active_filters.append(f"{year_range[0]}–{year_range[1]}")
if selected_methods and len(selected_methods) < len(ALL_METHODS):
    active_filters.append(
        selected_methods[0] if len(selected_methods) == 1
        else f"{len(selected_methods)} methods"
    )
if dist_limit < MAX_DIST:
    active_filters.append(f"within {dist_limit * distance_factor:,.0f} {distance_suffix}")
if not keep_unknown_dist:
    active_filters.append("known distance only")
if hz_filter:
    active_filters.append(" / ".join(hz_filter) + " HZ")
if esi_floor > 0:
    active_filters.append(f"ESI ≥ {esi_floor:.2f}")

if live_filters["complete_only"]:
    active_filters.append("Complete measurements")

def change_page(page_key, page):
    st.session_state[page_key] = page
    st.session_state.pop("preview_planet", None)
    st.session_state.pop("card_preview_planet", None)


def catalogue_search(is_saved):
    search_key = "saved_search" if is_saved else "catalogue_search"
    page_key = "saved_page" if is_saved else "catalogue_page"
    return st.text_input(
        "Search saved planets" if is_saved else "Search catalogue",
        placeholder="Search by planet or host star", key=search_key, max_chars=120,
        label_visibility="collapsed", on_change=change_page, args=(page_key, 0),
    ).strip()


def filter_button():
    label = f"Filters ({len(active_filters)})" if active_filters else "Filters"
    st.button(label, icon=":material/tune:", key="open_filters", on_click=show_filters,
              help="Keyboard shortcut: F", width="stretch")


def render_selection_metrics():
    latest_year = int(fdf[YEAR].max())
    this_year = int((fdf[YEAR] == latest_year).sum())
    prev_year = int((fdf[YEAR] == latest_year - 1).sum())

    hz_count = int(fdf[HZ_STATUS].isin(["Conservative", "Optimistic"]).sum())
    hz_conservative = int((fdf[HZ_STATUS] == "Conservative").sum())

    best_esi_label, best_esi_help = "—", "No planet in this selection has a published radius and temperature."
    if fdf[ESI].notna().any():
        best = fdf.loc[fdf[ESI].idxmax()]
        best_esi_label = f"{best[ESI]:.3f}"
        best_esi_help = f"{best[NAME]} — the closest match to Earth in this selection."

    metric_row = st.container(key="headline-metrics")
    m1, m2, m3, m4, m5 = metric_row.columns(5)
    m1.metric(
        "Planets shown",
        f"{len(fdf):,}",
        delta=(f"{len(fdf) - len(df):,} from filters" if len(fdf) < len(df) else None),
        delta_color="off",
        delta_arrow="off",
        help=f"Out of {len(df):,} confirmed planets in the archive.",
        icon=":material/public:",
    )
    m2.metric("Planetary systems", f"{fdf[HOST].nunique():,}",
              help="Distinct host stars in the current selection.",
              icon=":material/hub:")
    m3.metric(
        "In the habitable zone",
        f"{hz_count:,}",
        delta=f"{hz_conservative:,} conservative" if hz_conservative else None,
        delta_color="off",
        delta_arrow="off",
        help=HZ_HELP,
        icon=":material/water_drop:",
    )
    m4.metric("Best Earth Similarity", best_esi_label, help=best_esi_help,
              icon=":material/star:")
    m5.metric(
        f"Discovered in {latest_year}",
        f"{this_year:,}",
        delta=f"{this_year - prev_year:+,} vs {latest_year - 1}" if prev_year else None,
        icon=":material/trending_up:",
    )


if view == "Explore":
    with st.container(key="home-workspace"):
        rail_col, catalogue_col = st.columns([1, 3])
        home_rail = rail_col.container(key="home-rail")
        home_catalogue = catalogue_col.container(key="home-catalogue")
    with home_rail:
        if not fdf.empty:
            with st.container(key="home-overview", border=True, gap="xsmall"):
                st.html('<h2 class="overview-heading">Selection overview</h2>')
                render_selection_metrics()
        with st.container(key="home-discovery", border=True, gap="xsmall"):
            st.markdown("#### Discover")
            render_discovery_shortcuts("rail")
            with st.popover("Guided discoveries", width="stretch"):
                render_guided_discoveries("rail")
    with home_catalogue:
        st.subheader("Planet catalogue")
        with st.container(key="home-toolbar"):
            search_col, filter_col, options_col, discover_col = st.columns(
                [5, 1.35, 1.3, 1.45], vertical_alignment="bottom",
            )
            with search_col:
                home_query = catalogue_search(is_saved=False)
            with filter_col:
                filter_button()
            home_options = options_col.popover("Options", width="stretch")
            with discover_col.popover("Discover", width="stretch"):
                st.markdown("**Discovery shortcuts**")
                render_discovery_shortcuts("toolbar")
                st.divider()
                render_guided_discoveries("toolbar")
        if active_filters:
            with st.container(key="active-filters", horizontal=True, vertical_alignment="center", gap="medium"):
                st.caption(" · ".join(active_filters))
                st.button("Clear all filters", on_click=reset_filters, type="tertiary")
else:
    with st.container(key="results-controls"):
        filter_col, count_col = st.columns([1, 3], vertical_alignment="center")
        with filter_col:
            filter_button()
        with count_col:
            st.markdown(f"**{len(fdf):,}** of {len(df):,} planets · {fdf[HOST].nunique():,} systems")
            if active_filters:
                st.caption(" · ".join(active_filters))

if st.session_state.get("filter_dialog_open"):
    filters_drawer()

if view == "Profile" and not fdf.empty:
    profile_names = set(fdf[NAME])
    if st.session_state.get("profile_pick") not in profile_names:
        st.session_state["profile_pick"] = fdf.sort_values(ESI, ascending=False)[NAME].iloc[0]
    if st.session_state.get("profile_rival") not in profile_names - {st.session_state["profile_pick"]}:
        st.session_state["profile_rival"] = "— none —"
_sync_filters_to_query()

if fdf.empty and view not in {"Saved", "Learn"}:
    message_area = home_catalogue if view == "Explore" else st
    message_area.warning("**No planets match the current filters.**")
    message_area.button("Relax filters", type="primary", on_click=reset_filters)
    if view != "Explore":
        st.stop()

if not fdf.empty and view in {*CHART_VIEWS, "Profile"}:
    with st.expander("Selection overview"):
        render_selection_metrics()

if active_filters and view != "Explore":
    st.button("Clear all filters", on_click=reset_filters, key="clear_active_filters")

with st.sidebar:
    st.caption(f"{len(fdf):,} of {len(df):,} planets match your filters.")
    st.download_button(
        "Download selection (CSV)", data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="exoplanets_filtered.csv", mime="text/csv", width="stretch",
    )

def preview_planet(name):
    st.session_state["preview_planet"] = name


def enqueue_comparison(name):
    tray = st.session_state["comparison_tray"]
    if name in tray:
        st.session_state["comparison_tray"] = [planet for planet in tray if planet != name]
    elif len(tray) < COMPARISON_LIMIT:
        st.session_state["comparison_tray"] = [*tray, name]


def open_comparison():
    selected = st.session_state["comparison_tray"]
    open_profile(selected[0], clear_filters=True)
    st.session_state["profile_rival"] = selected[1] if len(selected) > 1 else None


def dismiss_card_preview():
    st.session_state.pop("card_preview_planet", None)


def render_planet_preview(planet, from_saved):
    name = planet[NAME]
    st.subheader(name)
    st.caption(f"{planet[HOST]} · {int(planet[YEAR])} · {planet[METHOD]}")
    st.write(zone_label(planet[HZ_STATUS]))
    st.metric("Earth similarity", measurement(planet[ESI], ".3f"))
    if st.button("Open full profile", type="primary"):
        open_profile(name, clear_filters=from_saved)
        st.rerun()
    tray = st.session_state["comparison_tray"]
    selected = name in tray
    if st.button("Remove from comparison" if selected else "Add to comparison",
                 disabled=not selected and len(tray) >= COMPARISON_LIMIT):
        enqueue_comparison(name)
        st.rerun()


@st.dialog("Planet preview", on_dismiss=dismiss_card_preview)
def card_preview(planet, from_saved):
    render_planet_preview(planet, from_saved)


def show_summary_card(planet, from_saved):
    name = planet[NAME]
    radius = measurement(planet[RADIUS] / radius_divisor, ".2f", f" R{size_suffix}")
    distance = measurement(planet[DISTANCE] * distance_factor, ",.1f", f" {distance_suffix}")
    details = " · ".join((
        str(planet[HOST]),
        radius if radius != "—" else "Radius unknown",
        distance if distance != "—" else "Distance unknown",
    ))
    st.button(name, on_click=open_profile, args=(name, from_saved),
              key=f"card_open_{name}", type="tertiary")
    status = planet[HZ_STATUS]
    zone_class = "catalogue-zone is-habitable" if status in {"Conservative", "Optimistic"} else "catalogue-zone"
    st.html(f'<div class="catalogue-details"><p class="catalogue-meta">{html.escape(details)}</p>'
            f'<p class="{zone_class}">{zone_label(status)}</p></div>')
    st.button(f"Preview {name}", icon=":material/visibility:", help=f"Preview {name}",
              on_click=st.session_state.update, args=({"card_preview_planet": name},),
              key=f"card-preview-{name}", type="tertiary")


@st.dialog("Report a problem")
def feedback():
    st.write(f"Current view: **{st.session_state['active_view']}**")
    st.caption("You can describe what happened and include steps to reproduce it. Avoid sharing personal information.")
    title = f"Dashboard feedback: {st.session_state['active_view']}"
    st.link_button("Open GitHub issue form",
                   "https://github.com/akshh04/ExoplanetExplorer/issues/new?" + urlencode({"title": title}))


with st.sidebar:
    st.button("Report a problem", on_click=lambda: st.session_state.update(show_feedback=True))
feedback_requested = st.session_state.pop("show_feedback", False)
if feedback_requested and not st.session_state.get("filter_dialog_open"):
    feedback()

tray = st.session_state["comparison_tray"]
if tray and view in {"Explore", "Saved", "Profile"}:
    comparison_area = home_catalogue if view == "Explore" else st
    with comparison_area.container(border=True, key="compare-tray"):
        st.caption(f"Comparison · {len(tray)} of {COMPARISON_LIMIT} planets")
        st.write(" · ".join(tray))
        if view == "Profile":
            st.markdown("[Jump to comparison](#compare-with-earth)")
        else:
            st.button("Compare with Earth", on_click=open_comparison)
        with st.expander("Edit comparison"):
            for name in tray:
                st.button(f"Remove {name} from comparison", on_click=enqueue_comparison, args=(name,),
                          key=f"tray_remove_{name}")
            st.button("Clear comparison", on_click=lambda: st.session_state.update(comparison_tray=[]))

def render_catalogue(is_saved, home_query="", home_options=None):
    page_key = "saved_page" if is_saved else "catalogue_page"
    search_key = "saved_search" if is_saved else "catalogue_search"
    reset_page = {"on_change": change_page, "args": (page_key, 0)}
    if is_saved:
        st.subheader("Saved planets")
        query = catalogue_search(is_saved=True)
        collection_control, sort_control, layout_control = st.columns(3)
    else:
        query = home_query
        collection_control = sort_control = layout_control = home_options
    if is_saved:
        collection_options = ["All saved", *sorted(st.session_state["collections"])]
        if st.session_state.get("saved_collection") not in collection_options:
            st.session_state["saved_collection"] = "All saved"
        selection_name = collection_control.selectbox(
            "Collection", collection_options, key="saved_collection", **reset_page,
        )
        saved_names = st.session_state["favorites"]
        if selection_name != "All saved":
            saved_names = st.session_state["collections"][selection_name]
        catalogue = df[df[NAME].isin(saved_names)]
    else:
        collection = collection_control.selectbox(
            "Collection", ["All worlds", "Closest worlds", "Giant planets", "Unusual orbits", "Multi-planet systems"],
            key="catalogue_collection", **reset_page,
        )
        catalogue = fdf
        if collection == "Closest worlds":
            catalogue = catalogue[catalogue[DISTANCE].le(50)]
        elif collection == "Giant planets":
            catalogue = catalogue[catalogue[RADIUS].ge(8)]
        elif collection == "Unusual orbits":
            catalogue = catalogue[catalogue["pl_orbeccen"].ge(0.6) | catalogue[PERIOD].lt(1)]
        elif collection == "Multi-planet systems":
            catalogue = catalogue[catalogue[HOST].isin(df[HOST].value_counts().loc[lambda counts: counts > 1].index)]

    sort_options = {
        "Earth similarity": (ESI, False), "Nearest first": (DISTANCE, True),
        "Newest discoveries": (YEAR, False), "Latest publications": ("disc_pubdate", False),
        "Name": (NAME, True),
    }
    if query:
        name_match = catalogue[NAME].str.contains(query, case=False, regex=False, na=False)
        host_match = catalogue[HOST].str.contains(query, case=False, regex=False, na=False)
        catalogue = catalogue[name_match | host_match]

    sort_by = sort_control.selectbox("Sort planets", list(sort_options), key="catalogue_sort", **reset_page)
    sort_key, ascending = sort_options[sort_by]
    catalogue = catalogue.sort_values([sort_key, NAME] if sort_key != NAME else [NAME],
                                      ascending=[ascending, True] if sort_key != NAME else ascending)
    layout = layout_control.radio("Layout", ["Cards", "Table"], key="catalogue_layout", horizontal=True)
    if not is_saved:
        with home_options:
            st.divider()
            render_global_search()
    if sort_by == "Latest publications":
        sort_control.caption("Discovery publication date from NASA. Catalogue release dates are listed separately in Trends.")
    if catalogue.empty:
        if query:
            st.info("No planets match this search in the current selection.")
            st.button("Clear search", on_click=lambda: st.session_state.update({search_key: ""}))
        elif is_saved or not fdf.empty:
            st.info("Save planets from their profiles to fill this collection." if is_saved
                    else "No planets match this collection and the current filters.")
    else:
        st.html(f'<p class="catalogue-meta" role="status" data-testid="catalogue-count">'
                f'{len(catalogue):,} {"planet" if len(catalogue) == 1 else "planets"}</p>')
        if layout == "Cards":
            page_size = 12
            last_page = (len(catalogue) - 1) // page_size
            page = min(last_page, max(0, st.session_state.get(page_key, 0)))
            st.session_state[page_key] = page
            start = page * page_size
            with st.container(key="catalogue-results"):
                for _, planet_row in catalogue.iloc[start:start + page_size].iterrows():
                    with st.container(key=f"planet-card-{planet_row[NAME]}", border=True, gap="xxsmall"):
                        show_summary_card(planet_row, is_saved)
            if last_page:
                previous, position, next_page = st.columns(3, vertical_alignment="center")
                previous.button("Previous", disabled=page == 0, on_click=change_page, args=(page_key, page - 1))
                position.caption(f"Page {page + 1} of {last_page + 1}")
                next_page.button("Next", disabled=page == last_page, on_click=change_page, args=(page_key, page + 1))
            preview_name = st.session_state.get("card_preview_planet")
            if preview_name and not (feedback_requested or st.session_state.get("filter_dialog_open")):
                matching = catalogue[catalogue[NAME] == preview_name]
                if not matching.empty:
                    card_preview(matching.iloc[0], is_saved)
                else:
                    dismiss_card_preview()
        else:
            columns = [RADIUS, MASS, DISTANCE, YEAR, ESI, HZ_STATUS]
            with (st.expander("Table columns") if is_saved else home_options):
                visible = st.multiselect("Measurements", columns, default=columns, key="catalogue_columns")
                st.button("Restore default columns",
                          on_click=lambda: st.session_state.update(catalogue_columns=columns))
            display = catalogue[[NAME, *visible]].copy()
            for column, scale in [(RADIUS, 1 / radius_divisor), (MASS, 1 / mass_divisor), (DISTANCE, distance_factor)]:
                if column in display:
                    display[column] *= scale
            # A row index is only valid for the specific ordered result set.
            table_id = hashlib.sha256(pd.util.hash_pandas_object(catalogue[NAME], index=False).values.tobytes()).hexdigest()[:16]
            selection = st.dataframe(
                display, hide_index=True, height=465,
                key=f"planet_catalogue_{table_id}", on_select="rerun", selection_mode="single-row",
                column_config={
                    NAME: st.column_config.TextColumn("Planet", pinned=True),
                    RADIUS: st.column_config.NumberColumn(f"Radius (R{size_suffix})", format="%.2f"),
                    MASS: st.column_config.NumberColumn(f"Mass (M{size_suffix})", format="%.2f"),
                    DISTANCE: st.column_config.NumberColumn(f"Distance ({distance_suffix})", format="%.1f"),
                    ESI: st.column_config.NumberColumn("Earth similarity", format="%.3f"),
                },
            )
            if selection.selection.rows:
                preview_planet(catalogue.iloc[selection.selection.rows[0]][NAME])
            selected_name = st.session_state.get("preview_planet")
            matched = catalogue[catalogue[NAME] == selected_name]
            if not matched.empty:
                with st.container(border=True, key="catalogue-preview"):
                    render_planet_preview(matched.iloc[0], is_saved)
            else:
                st.caption("Select a row to open its profile.")
    if is_saved and st.session_state["favorites"]:
        with st.expander("Manage collections"):
            with st.form("planet_collection"):
                label = st.text_input("Collection name", max_chars=40).strip()
                names = st.multiselect("Saved planets", st.session_state["favorites"])
                if st.form_submit_button("Save collection") and label and label != "All saved":
                    st.session_state["collections"] = {**st.session_state["collections"], label: names}
                    st.rerun()
            if selection_name != "All saved" and st.button("Delete this collection"):
                del st.session_state["collections"][selection_name]
                st.rerun()
    freshness = datetime.fromisoformat(str(df.attrs["fetched_at"])).strftime("%d %b %Y, %H:%M UTC")
    st.caption(f"NASA archive · Fetched {freshness}")


if view == "Explore":
    with home_catalogue:
        render_catalogue(False, home_query, home_options)
elif view == "Saved":
    render_catalogue(True)


def chart_table(frame, label="View chart data"):
    panel = st.expander(label, on_change="rerun")
    if panel.open:
        panel.dataframe(frame, hide_index=True)


@st.cache_data(ttl=86400, show_spinner=False)
def archive_releases():
    query = """
        select pl_name, min(releasedate) as first_release
        from ps
        group by pl_name
    """
    with requests.get(ARCHIVE_URL, params={"query": query, "format": "csv"},
                      timeout=(8, 35)) as response:
        response.raise_for_status()
        releases = pd.read_csv(StringIO(response.text))
    if not {"pl_name", "first_release"}.issubset(releases.columns):
        raise ValueError("NASA did not return planet release dates.")
    releases["first_release"] = pd.to_datetime(releases["first_release"], errors="coerce")
    return (releases.dropna().sort_values("first_release", ascending=False).head(50)
            .rename(columns={"pl_name": NAME, "first_release": "First archive release"}))


if view == "Sky":
    st.subheader("Where known planets are in the sky")
    sky = fdf.dropna(subset=["ra", "dec"])
    if sky.empty:
        st.info("No sky coordinates are available in this selection.")
    else:
        fig = px.scatter(
            sky, x="ra", y="dec", color=DISTANCE, hover_name=NAME, custom_data=[NAME],
            labels={"ra": "Right ascension (°)", "dec": "Declination (°)", DISTANCE: "Distance (pc)"},
            color_continuous_scale="Viridis", opacity=0.65,
        )
        fig.update_xaxes(autorange="reversed")
        transparent(fig, height=CHART_H_TALL, uirevision="sky")
        selected = st.plotly_chart(fig, key="sky_map", on_select="rerun", selection_mode="points", config=CHART_CONFIG)
        if selected.selection.points:
            name = selected.selection.points[0]["customdata"][0]
            if name in set(fdf[NAME]):
                st.button(f"Open {name}", on_click=open_profile, args=(name,))
        st.caption(f"{len(sky):,} of {len(fdf):,} selected planets have sky coordinates. "
                   "Right ascension increases to the left. Surveys target different areas of the sky.")
        chart_table(sky[[NAME, HOST, "ra", "dec", DISTANCE]])

# --- TAB 1: DISCOVERY TRENDS --------------------------------------------
if st.session_state["active_view"] == "Trends":
    st.subheader("The pace of discovery")
    st.markdown(
        "Since the first confirmed planets in the 1990s, the rate of discovery has "
        "skyrocketed — driven largely by space telescopes using the transit method."
    )

    trend_mode = st.radio(
        "Trend view",
        ["Per year", "Cumulative", "Share of year (%)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    yearly = (
        fdf.groupby([YEAR, METHOD_SHORT], observed=True)
        .size()
        .reset_index(name="Count")
        .astype({YEAR: int})
    )

    if trend_mode == "Cumulative":
        pivot = yearly.pivot(index=YEAR, columns=METHOD_SHORT, values="Count").fillna(0)
        pivot = pivot.reindex(range(year_range[0], year_range[1] + 1), fill_value=0).cumsum()
        plot_df = pivot.reset_index().melt(
            id_vars=YEAR, var_name=METHOD_SHORT, value_name="Count"
        )
        fig = px.area(plot_df, x=YEAR, y="Count", color=METHOD_SHORT, color_discrete_map=COLOR_MAP, pattern_shape=METHOD_SHORT)
        y_title = "Planets known (cumulative)"
    else:
        plot_df = yearly.copy()
        y_title = "Planets discovered"
        if trend_mode == "Share of year (%)":
            totals = plot_df.groupby(YEAR)["Count"].transform("sum")
            plot_df["Count"] = plot_df["Count"] / totals * 100
            y_title = "Share of discoveries (%)"
        fig = px.bar(plot_df, x=YEAR, y="Count", color=METHOD_SHORT, color_discrete_map=COLOR_MAP, pattern_shape=METHOD_SHORT)

    transparent(
        fig,
        xaxis_title="Year",
        yaxis_title=y_title,
        hovermode="x unified",
        margin=dict(t=10, b=0, l=0, r=0),
        height=CHART_H_TALL,
    )
    fig.update_xaxes(dtick=5, tickformat="d")
    if st.checkbox("Show a second chart for discovery methods"):
        trend_panel, method_panel = st.columns([3, 2])
        trend_panel.plotly_chart(fig, config=CHART_CONFIG)
        methods = yearly.groupby(METHOD_SHORT)["Count"].sum().sort_values().reset_index()
        method_chart = px.bar(methods, x="Count", y=METHOD_SHORT, orientation="h",
                              color=METHOD_SHORT, color_discrete_map=COLOR_MAP)
        transparent(method_chart, height=CHART_H_TALL, showlegend=False,
                    xaxis_title="Planet discoveries", yaxis_title="Discovery method")
        method_panel.plotly_chart(method_chart, config=CHART_CONFIG)
    else:
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

    by_method = (
        fdf[METHOD].value_counts().rename_axis(METHOD).reset_index(name="Planets")
    )
    by_method["Share"] = by_method["Planets"] / by_method["Planets"].sum()
    st.dataframe(
        by_method,
        hide_index=True,
        width="stretch",
        column_config={
            "Planets": st.column_config.NumberColumn(format="%d"),
            "Share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        },
    )

    st.caption("Discovery methods favour different planet sizes, periods and viewing angles.")
    chart_table(plot_df, "View discovery chart data")
    with st.expander("Discovery facilities"):
        facilities = fdf.dropna(subset=["disc_facility"])
        counts = facilities["disc_facility"].value_counts().head(12)
        if not counts.empty:
            selected_facility = st.selectbox("Facility or mission", counts.index.tolist())
            timeline = (facilities[facilities["disc_facility"] == selected_facility]
                        .groupby(YEAR).size().rename("Discoveries").reset_index())
            cols = st.columns(2) if st.checkbox("Show chart and table side by side") else None
            figure_area, table_area = cols if cols else (st.container(), st.container())
            figure_area.line_chart(timeline, x=YEAR, y="Discoveries", y_label="Planet discoveries")
            table_area.dataframe(timeline, hide_index=True)
        else:
            st.info("NASA has not supplied facility data for this selection.")
    with st.expander("Recent catalogue releases"):
        st.caption("The first release date in NASA’s Planetary Systems table is distinct from its discovery year.")
        if st.button("Load NASA release dates"):
            st.session_state["show_release_dates"] = True
        if st.session_state.get("show_release_dates"):
            try:
                releases = archive_releases().merge(df[[NAME, YEAR]], on=NAME)
                st.dataframe(releases, hide_index=True)
            except (requests.RequestException, ValueError):
                st.warning("NASA release dates are currently unavailable. Try again later.")

# --- TAB 2: SIZE & ORBIT -------------------------------------------------
if st.session_state["active_view"] == "Orbits":
    st.subheader("Orbital period vs. planet size")
    st.markdown(
        "Each dot is a world; our own planets are overlaid as stars for scale. "
        "Look for the **hot Jupiters** at the top left and the dense band of "
        "**super-Earths** through the middle."
    )

    st.caption("Select a point to inspect it below and highlight it in supported charts.")
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        radius_range = st.slider(
            "Planet radius (Earth radii)", 0.0, 30.0, (0.0, 30.0), 0.5
        )
    with ctrl2:
        st.session_state.setdefault("orbit_period_limit", 10_000)
        period_limit = st.select_slider(
            "Max orbital period (days)",
            options=[10, 87.97, 100, 1_000, 5_000, 10_000, 100_000],
            key="orbit_period_limit",
            format_func=lambda v: f"{v:,}",
        )
    with ctrl3:
        show_solar = st.checkbox("Solar System", value=True)

    scatter_df = fdf[
        fdf[RADIUS].between(*radius_range)
        & (fdf[PERIOD] <= period_limit)
        & (fdf[RADIUS] > 0)
        & (fdf[PERIOD] > 0)
    ]

    if scatter_df.empty:
        st.info("No planets fall inside this radius and period window.")
    else:
        fig = px.scatter(
            scatter_df,
            x=PERIOD,
            y=RADIUS,
            color=METHOD_SHORT,
            color_discrete_map=COLOR_MAP,
            hover_name=NAME,
            custom_data=[NAME],
            hover_data={TEMP: ":.0f", DISTANCE: ":.1f"},
            log_x=True,
            log_y=True,
            opacity=0.55,
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        highlighted = scatter_df[scatter_df[NAME] == st.session_state.get("highlighted_planet")]
        if not highlighted.empty:
            fig.add_scatter(x=highlighted[PERIOD], y=highlighted[RADIUS], mode="markers",
                            marker=dict(size=14, symbol="diamond-open", color=CHART_FONT),
                            name="Selected planet", hoverinfo="skip")

        hidden = []
        if show_solar:
            for _, planet in SOLAR_SYSTEM.iterrows():
                visible = (
                    radius_range[0] <= planet["Radius"] <= radius_range[1]
                    and planet["Period"] <= period_limit
                )
                if not visible:
                    hidden.append(planet["Name"])
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[planet["Period"]],
                        y=[planet["Radius"]],
                        mode="markers+text",
                        marker=dict(size=17, color=planet["Color"], symbol="star",
                                    line=dict(width=1, color="rgba(128,128,128,0.7)")),
                        name=planet["Name"],
                        text=[planet["Name"]],
                        textposition="top center",
                        textfont=dict(size=13, color=planet["Color"]),
                        hovertemplate=f"<b>{planet['Name']}</b><br>Period: %{{x:,.1f}} d"
                                      f"<br>Radius: %{{y:.2f}} R⊕<extra></extra>",
                        showlegend=False,
                    )
                )

        transparent(fig, margin=dict(t=10, b=0, l=0, r=0), height=CHART_H_SCATTER, uirevision="orbits")
        fig.update_xaxes(title_text="Orbital period (days, log scale)")
        fig.update_yaxes(title_text="Radius (Earth radii, log scale)")
        selected = st.plotly_chart(
            fig, width="stretch", config=CHART_CONFIG, key="orbit_selection",
            on_select="rerun", selection_mode="points",
        )
        if selected.selection.points:
            point_data = selected.selection.points[0].get("customdata", [])
            if point_data and point_data[0] in set(scatter_df[NAME]):
                chosen = point_data[0]
                st.session_state["highlighted_planet"] = chosen
                row = scatter_df[scatter_df[NAME] == chosen].iloc[0]
                st.info(f"{chosen} · {row[PERIOD]:,.2f} days · {row[RADIUS]:.2f} Earth radii")
                st.button(f"Open {chosen}", on_click=open_profile, args=(chosen,))

        cap = f"{len(scatter_df):,} planets plotted."
        if hidden:
            cap += f" Outside the current window: {', '.join(hidden)}."
        st.caption(cap)
        chart_table(scatter_df[[NAME, PERIOD, RADIUS, METHOD, TEMP]])

# --- TAB 3: K-MEANS CLASSIFICATION ---------------------------------------
if st.session_state["active_view"] == "Classes":
    st.subheader("Unsupervised classification (K-Means)")
    st.markdown(
        "Rather than hand-defining what a *hot Jupiter* or *super-Earth* is, K-Means "
        "finds natural groupings in log mass–radius space. Groups are numbered by "
        "median mass, so the labels stay consistent as the archive grows."
    )

    ml_df = fdf.dropna(subset=[MASS, RADIUS]).copy()
    ml_df = ml_df[(ml_df[MASS] > 0) & (ml_df[RADIUS] > 0)]

    n_clusters = st.slider("Number of groups (k)", 2, 6, 3)

    if len(ml_df[[MASS, RADIUS]].drop_duplicates()) < n_clusters:
        st.warning(
            "Not enough planets with both a mass and a radius to run clustering. "
            "Widen the filters in the sidebar."
        )
    else:
        features = ml_df[[MASS, RADIUS]]
        with st.spinner("Fitting K-Means…"):
            ml_df["Group"] = cluster_planets(features, n_clusters)

        stats = (
            ml_df.groupby("Group")[[MASS, RADIUS]]
            .median()
            .join(ml_df.groupby("Group").size().rename("Planets"))
            .reset_index()
        )
        names = {
            int(row["Group"]): describe_cluster(row[RADIUS], row[MASS])
            for _, row in stats.iterrows()
        }
        # At higher k two clusters can land in the same size bracket; add the
        # median radius so every legend entry stays distinguishable.
        duplicated = {n for n in names.values() if list(names.values()).count(n) > 1}
        labels = {
            group: f"Group {group + 1} · {name}"
            + (f" ({stats.loc[stats['Group'] == group, RADIUS].iloc[0]:.1f} R⊕)"
               if name in duplicated else "")
            for group, name in names.items()
        }
        ml_df["Group Name"] = ml_df["Group"].map(labels)
        stats["Group Name"] = stats["Group"].map(labels)

        group_colors = {labels[i]: PALETTE[i % len(PALETTE)] for i in labels}

        fig = px.scatter(
            ml_df,
            x=MASS,
            y=RADIUS,
            color="Group Name",
            color_discrete_map=group_colors,
            category_orders={"Group Name": [labels[i] for i in sorted(labels)]},
            hover_name=NAME,
            hover_data={METHOD: True, "Group": False},
            log_x=True,
            log_y=True,
            opacity=0.7,
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        highlighted = ml_df[ml_df[NAME] == st.session_state.get("highlighted_planet")]
        if not highlighted.empty:
            fig.add_scatter(x=highlighted[MASS], y=highlighted[RADIUS], mode="markers",
                            marker=dict(size=14, symbol="diamond-open", color=CHART_FONT),
                            name="Selected planet", hoverinfo="skip")
        # One entry per row: the group names run long enough to overrun a
        # half-width column.
        transparent(
            fig,
            margin=dict(t=10, b=0, l=0, r=0),
            height=CHART_H_SCATTER - 20,
            legend_entrywidth=0.95,
        )
        fig.update_xaxes(title_text="Mass (Earth masses, log scale)")
        fig.update_yaxes(title_text="Radius (Earth radii, log scale)")
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

        st.dataframe(
            stats[["Group Name", "Planets", MASS, RADIUS]],
            hide_index=True,
            width="stretch",
            column_config={
                "Group Name": "Group",
                "Planets": st.column_config.NumberColumn(format="%d"),
                MASS: st.column_config.NumberColumn("Median mass (M⊕)", format="%.2f"),
                RADIUS: st.column_config.NumberColumn("Median radius (R⊕)", format="%.2f"),
            },
        )
        st.caption(
            f"{len(ml_df):,} planets have both a published mass and radius "
            f"({len(fdf) - len(ml_df):,} of the current selection do not and are excluded)."
        )

        chart_table(ml_df[[NAME, MASS, RADIUS, "Group Name"]])

# --- TAB 4: 3D EXPLORER --------------------------------------------------
if st.session_state["active_view"] == "3D":
    st.subheader("Three-dimensional view")
    st.markdown(
        "Rotate and zoom to see how radius, orbital period and equilibrium "
        "temperature relate. Hot, close-in planets sit at the front-left."
    )

    td_df = fdf.dropna(subset=[RADIUS, PERIOD, TEMP]).copy()
    td_df = td_df[(td_df[RADIUS] > 0) & (td_df[PERIOD] > 0)]

    if td_df.empty:
        st.info("No planets in the current selection have all three parameters.")
    elif st.radio("Display", ["3D", "2D"], horizontal=True) == "2D":
        flat = td_df.sample(2500, random_state=42) if len(td_df) > 2500 else td_df
        fig = px.scatter(flat, x=PERIOD, y=RADIUS, color=TEMP, hover_name=NAME,
                         log_x=True, log_y=True, color_continuous_scale="Inferno",
                         render_mode="svg", opacity=0.65)
        transparent(fig, height=CHART_H_TALL, uirevision="flat")
        st.plotly_chart(fig, config=CHART_CONFIG)
        st.caption(f"Colour shows temperature. Plotting {len(flat):,} of {len(td_df):,} planets with all three measurements.")
        chart_table(td_df[[NAME, PERIOD, RADIUS, TEMP]])
    else:
        # 3D scenes get sluggish past a few thousand markers, so sample.
        MAX_POINTS = 2500
        sampled = td_df
        if len(td_df) > MAX_POINTS:
            sampled = td_df.sample(MAX_POINTS, random_state=42)

        color_by = st.radio(
            "Colour by",
            [METHOD, TEMP],
            horizontal=True,
            format_func=lambda c: "Discovery method" if c == METHOD else "Equilibrium temperature",
            label_visibility="collapsed",
        )

        kwargs = (
            dict(color=METHOD_SHORT, color_discrete_map=COLOR_MAP)
            if color_by == METHOD
            else dict(color=TEMP, color_continuous_scale="Inferno")
        )
        fig = px.scatter_3d(
            sampled,
            x=PERIOD,
            y=RADIUS,
            z=TEMP,
            hover_name=NAME,
            log_x=True,
            log_y=True,
            opacity=0.7,
            **kwargs,
        )
        fig.update_traces(marker=dict(size=3.5, line=dict(width=0)))
        highlighted = td_df[td_df[NAME] == st.session_state.get("highlighted_planet")]
        if not highlighted.empty:
            fig.add_scatter3d(x=highlighted[PERIOD], y=highlighted[RADIUS], z=highlighted[TEMP],
                              mode="markers", marker=dict(size=9, color=CHART_FONT, symbol="diamond-open"),
                              text=highlighted[NAME], hoverinfo="text", name="Selected planet")

        # Flat, top-level magic-underscore kwargs — nesting these as dict
        # values (e.g. scene=dict(xaxis=dict(title=dict(...)))) silently
        # fails to reach the axis title/tick fonts in this Plotly version.
        axis_titles = {"xaxis": "Period (days)", "yaxis": "Radius (R⊕)", "zaxis": "Eq. temp (K)"}
        scene_kwargs = {"scene_bgcolor": "rgba(0,0,0,0)"}
        for axis, title in axis_titles.items():
            scene_kwargs[f"scene_{axis}_title_text"] = title
            scene_kwargs[f"scene_{axis}_title_font_color"] = CHART_FONT
            scene_kwargs[f"scene_{axis}_tickfont_color"] = CHART_FONT
            scene_kwargs[f"scene_{axis}_color"] = CHART_FONT
            # Thin guideline through every tick (division point) on each axis.
            scene_kwargs[f"scene_{axis}_showgrid"] = True
            scene_kwargs[f"scene_{axis}_gridwidth"] = 1
            scene_kwargs[f"scene_{axis}_gridcolor"] = CHART_GRID
            scene_kwargs[f"scene_{axis}_zerolinecolor"] = CHART_GRID
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=CHART_FONT,
            legend_font_color=CHART_FONT,
            legend_orientation="h",
            # A gl3d scene ignores automargin, so the legend is pinned to the
            # container's top edge and the scene's own domain is shortened to
            # leave room -- otherwise the key sits on top of the plot.
            legend_yref="container",
            legend_yanchor="top",
            legend_y=0.99,
            legend_xanchor="left",
            legend_x=0,
            legend_font_size=10,
            legend_entrywidthmode="fraction",
            legend_entrywidth=0.45,
            legend_itemsizing="constant",
            scene_domain_y=(0, 0.75),
            uirevision="3d",
            coloraxis_colorbar_orientation="h",
            coloraxis_colorbar_y=-0.02,
            coloraxis_colorbar_thickness=12,
            coloraxis_colorbar_len=0.9,
            coloraxis_colorbar_tickfont_color=CHART_FONT,
            coloraxis_colorbar_title_font_color=CHART_FONT,
            margin=dict(l=0, r=0, b=70, t=10),
            height=CHART_H_3D + 60,
            **scene_kwargs,
        )
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

        if len(sampled) < len(td_df):
            st.caption(f"Showing a random sample of {len(sampled):,} of {len(td_df):,} planets to keep the 3D view responsive.")
        else:
            st.caption(f"{len(sampled):,} planets plotted.")
        chart_table(sampled[[NAME, PERIOD, RADIUS, TEMP]])

# --- TAB 5: HABITABLE WORLDS ---------------------------------------------
if st.session_state["active_view"] == "Habitable":
    st.subheader("The search for another Earth")
    st.markdown(
        "Two ways to shortlist candidates: the **habitable zone** computed from each "
        "star's own luminosity and temperature, or a plain **equilibrium-temperature "
        "band**. The first is the more rigorous test; the second needs no stellar data "
        "and so covers more planets."
    )

    mode = st.radio(
        "Selection method",
        ["Habitable zone (Kopparapu)", "Temperature band"],
        horizontal=True,
        label_visibility="collapsed",
    )

    c1, c2 = st.columns(2)
    with c1:
        radius_cap = st.slider("Max radius (Earth radii)", 0.5, 5.0, 2.0, 0.1)

    if mode.startswith("Habitable"):
        with c2:
            zone_choice = st.radio(
                "Zone width",
                ["Conservative only", "Conservative + optimistic"],
                index=1,
                help=HZ_HELP,
            )
        wanted = (["Conservative"] if zone_choice == "Conservative only"
                  else ["Conservative", "Optimistic"])
        candidates = fdf[
            fdf[HZ_STATUS].isin(wanted) & (fdf[RADIUS] <= radius_cap) & (fdf[RADIUS] > 0)
        ].copy()
        criterion = (
            f"radius \u2264 {radius_cap:g} R\u2295 and an orbit inside the "
            f"{' / '.join(w.lower() for w in wanted)} habitable zone"
        )
        empty_msg = (
            "No planets in the current selection sit inside the habitable zone at "
            "this radius. Try the optimistic zone, a larger radius, or widen the "
            "sidebar filters."
        )
    else:
        with c2:
            temp_band = st.slider("Equilibrium temperature (K)", 150, 500, (273, 373), 5)
        candidates = fdf[
            fdf[TEMP].between(*temp_band)
            & (fdf[RADIUS] <= radius_cap)
            & (fdf[RADIUS] > 0)
        ].copy()
        criterion = (
            f"radius \u2264 {radius_cap:g} R\u2295 and an equilibrium temperature of "
            f"{temp_band[0]}\u2013{temp_band[1]} K"
        )
        empty_msg = "No planets match this radius and temperature window."

    candidates = candidates.sort_values(ESI, ascending=False, na_position="last")

    if candidates.empty:
        st.info(empty_msg)
    else:
        st.success(f"**{len(candidates):,} candidates** with {criterion}.")

        # Leaderboard of the closest matches to Earth.
        top = candidates.dropna(subset=[ESI]).head(12).iloc[::-1]
        if not top.empty:
            board = go.Figure(go.Bar(
                x=top[ESI], y=top[NAME], orientation="h",
                marker_color=[HZ_COLORS.get(s, HZ_COLORS["Unknown"])
                              for s in top[HZ_STATUS]],
                text=[f"{v:.3f}" for v in top[ESI]],
                textposition="outside", textfont=dict(color=CHART_FONT),
                customdata=top[[HZ_STATUS, RADIUS, TEMP]].to_numpy(),
                hovertemplate=(
                    "<b>%{y}</b><br>ESI %{x:.3f}<br>%{customdata[0]} zone"
                    "<br>%{customdata[1]:.2f} R\u2295 &middot; %{customdata[2]:.0f} K"
                    "<extra></extra>"
                ),
            ))
            transparent(
                board, height=max(260, 30 * len(top)),
                margin=dict(t=10, b=10, l=10, r=52), showlegend=False,
                xaxis_title="Earth Similarity Index",
            )
            board.update_xaxes(range=[0, 1.06])
            st.plotly_chart(board, width="stretch", config=CHART_CONFIG)
            st.caption(
                "Bar colour shows the habitable-zone class. ESI combines radius and "
                "equilibrium temperature; 1.000 would be an exact match for Earth."
            )

        search = st.text_input(
            "Search by planet or host star", placeholder="e.g. TRAPPIST, Kepler-452"
        )
        table = candidates
        if search:
            needle = search.strip()
            # regex=False: planet names and user input contain characters like
            # "(" and "+" that would otherwise raise a regex error.
            table = candidates[
                candidates[NAME].str.contains(needle, case=False, na=False, regex=False)
                | candidates[HOST].str.contains(needle, case=False, na=False, regex=False)
            ]
            if table.empty:
                st.info(f"No candidate matches “{needle}”.")

        columns = [NAME, HOST, YEAR, METHOD, RADIUS, TEMP, ESI, HZ_STATUS, PERIOD, DISTANCE]
        st.dataframe(
            table[columns],
            width="stretch",
            hide_index=True,
            height=430,
            column_config={
                YEAR: st.column_config.NumberColumn("Year", format="%d"),
                RADIUS: st.column_config.NumberColumn("Radius (R⊕)", format="%.2f"),
                TEMP: st.column_config.NumberColumn("Eq. temp (K)", format="%.0f"),
                ESI: st.column_config.ProgressColumn(
                    "ESI", format="%.3f", min_value=0.0, max_value=1.0
                ),
                HZ_STATUS: st.column_config.TextColumn("Zone"),
                PERIOD: st.column_config.NumberColumn("Period (days)", format="%.2f"),
                DISTANCE: st.column_config.NumberColumn("Distance (pc)", format="%.1f"),
            },
        )

        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Download these candidates (CSV)",
            data=candidates[columns].to_csv(index=False).encode("utf-8"),
            file_name="temperate_candidates.csv",
            mime="text/csv",
            width="stretch",
        )
        d2.download_button(
            "⬇️ Download full filtered dataset (CSV)",
            data=fdf.drop(columns=[HZ_INNER_OPT, HZ_OUTER_OPT])
                    .to_csv(index=False).encode("utf-8"),
            file_name="exoplanets_filtered.csv",
            mime="text/csv",
            width="stretch",
        )

        st.caption(
            "A caveat on both methods: equilibrium temperature ignores atmosphere, "
            "albedo and greenhouse effects, and the habitable-zone limits assume an "
            "Earth-like atmosphere around a single star. Treat this as a shortlist "
            "for follow-up, not a list of habitable planets."
        )

class ArchiveReferenceParser(HTMLParser):
    href = None

    def handle_starttag(self, tag, attrs):
        if tag == "a" and self.href is None:
            self.href = dict(attrs).get("href")


def archive_reference(value):
    if not isinstance(value, str):
        return None
    parser = ArchiveReferenceParser()
    parser.feed(value)
    if not parser.href:
        return None
    try:
        url = urljoin(ARCHIVE_HOME, parser.href)
        target = urlsplit(url)
    except ValueError:
        return None
    return url if target.scheme == "https" and target.hostname else None


# --- TAB 6: PLANET PROFILE -----------------------------------------------
if st.session_state["active_view"] == "Profile":
    st.subheader("Individual planet profile")
    st.markdown(
        "Pick any world in the current selection to see its measurements, how it "
        "compares with Earth, and where it sits relative to its own star's "
        "habitable zone."
    )

    ranked = fdf.sort_values(ESI, ascending=False, na_position="last")
    planet_options = ranked[NAME].tolist()
    # Hold the chosen planet across reruns, but fall back to the most
    # Earth-like one whenever a filter change drops it out of the selection.
    if st.session_state.get("profile_pick") not in planet_options:
        st.session_state["profile_pick"] = planet_options[0]
    NO_RIVAL = "\u2014 none \u2014"
    sel_left, sel_right = st.columns(2)
    with sel_left:
        pick = st.selectbox(
            "Planet",
            options=planet_options,
            key="profile_pick",
            help="Sorted by Earth Similarity Index; type to search the whole selection.",
        )
    planet = fdf[fdf[NAME] == pick].iloc[0]

    rival_options = [NO_RIVAL] + [p for p in planet_options if p != pick]
    if st.session_state.get("profile_rival") not in rival_options:
        st.session_state["profile_rival"] = NO_RIVAL
    with sel_right:
        rival_pick = st.selectbox(
            "Compare with",
            options=rival_options,
            key="profile_rival",
            help="Put a second world side by side with this one and with Earth.",
        )
    rival = (
        fdf[fdf[NAME] == rival_pick].iloc[0] if rival_pick != NO_RIVAL else None
    )

    status = planet[HZ_STATUS]
    chip = HZ_COLORS.get(status, HZ_COLORS["Unknown"])
    st.markdown(
        f"""<div class="profile-head">
              <span class="profile-name">{html.escape(str(planet[NAME]))}</span>
              <span class="profile-chip chip-{hz_slug(status)}">
                {zone_label(status)}
              </span>
            </div>
            <div class="profile-sub">
              Orbits <b>{html.escape(str(planet[HOST]))}</b> &middot; found in
              <b>{int(planet[YEAR])}</b> by <b>{html.escape(str(planet[METHOD]))}</b>
            </div>""",
        unsafe_allow_html=True,
    )

    def toggle_saved():
        saved = list(st.session_state["favorites"])
        if pick in saved:
            saved.remove(pick)
            st.session_state["collections"] = {
                name: [world for world in planets if world != pick]
                for name, planets in st.session_state["collections"].items()
            }
        else:
            saved.append(pick)
        st.session_state["favorites"] = saved

    st.button("Remove saved planet" if pick in st.session_state["favorites"] else "Save planet",
              on_click=toggle_saved, key="save_planet")
    st.caption("Saved planets and collections are stored in this browser.")
    selected = pick in tray
    st.button("Remove from comparison" if selected else "Add to comparison",
              on_click=enqueue_comparison, args=(pick,), key="profile_compare",
              disabled=not selected and len(tray) >= COMPARISON_LIMIT)
    known = sum(pd.notna(planet[c]) for c in [RADIUS, MASS, PERIOD, TEMP, DISTANCE, SEMI_MAJOR])
    st.caption(f"Data coverage: {known}/6 core measurements published. A dash means unknown, not zero.")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Radius", measurement(planet[RADIUS] / radius_divisor, ".2f", f" R{size_suffix}"),
              help=f"Radius in {st.session_state['size_unit']} radii.")
    p2.metric("Mass", measurement(planet[MASS] / mass_divisor, ".2f", f" M{size_suffix}"),
              help=f"Mass in {st.session_state['size_unit']} masses.")
    p3.metric("Orbital period", measurement(planet[PERIOD], ",.2f", " d"))
    p4.metric(
        "Distance",
        measurement(planet[DISTANCE] * distance_factor, ",.1f", f" {distance_suffix}"),
        help=(f"{planet[DISTANCE] * 3.26156:,.1f} light-years away."
              if pd.notna(planet[DISTANCE]) else "No published distance."),
    )

    p5, p6, p7, p8 = st.columns(4)
    p5.metric(
        "Equilibrium temp",
        measurement(planet[TEMP], ",.0f", " K"),
        help=(f"{planet[TEMP] - 273.15:,.0f} degrees C"
              if pd.notna(planet[TEMP]) else "No published temperature."),
    )
    p6.metric("Insolation", measurement(planet[INSOLATION], ",.2f", " S" + EARTH_SYM),
              help="Starlight received relative to Earth. Earth = 1.")
    p7.metric("Semi-major axis", measurement(planet[SEMI_MAJOR], ",.4f", " au"))
    p8.metric("Earth Similarity", measurement(planet[ESI], ".3f"),
              help="Two-parameter ESI from radius and equilibrium temperature. 1.0 = Earth.")

    st.markdown("##### Where it orbits, against its star's habitable zone")
    inner_o, inner_c = planet[HZ_INNER_OPT], planet[HZ_INNER]
    outer_c, outer_o = planet[HZ_OUTER], planet[HZ_OUTER_OPT]
    a_planet = planet[SEMI_MAJOR]
    bounds_known = not any(
        pd.isna(v) for v in (inner_o, inner_c, outer_c, outer_o, a_planet)
    ) and min(inner_o, inner_c, outer_c, outer_o, a_planet) > 0 and status != "Unknown"

    if not bounds_known:
        st.info(
            "This system is missing the stellar luminosity, temperature or orbital "
            "distance needed to place a habitable zone."
        )
    else:
        siblings = df[(df[HOST] == planet[HOST]) & df[SEMI_MAJOR].gt(0)]
        hz = go.Figure()
        # Optimistic band first, so the conservative band paints on top of it.
        hz.add_shape(type="rect", x0=inner_o, x1=outer_o, y0=0.28, y1=0.72,
                     fillcolor=HZ_COLORS["Optimistic"], opacity=0.22,
                     line_width=0, layer="below")
        hz.add_shape(type="rect", x0=inner_c, x1=outer_c, y0=0.28, y1=0.72,
                     fillcolor=HZ_COLORS["Conservative"], opacity=0.38,
                     line_width=0, layer="below")

        others = siblings[siblings[NAME] != pick]
        if not others.empty:
            hz.add_trace(go.Scatter(
                x=others[SEMI_MAJOR], y=[0.5] * len(others), mode="markers",
                marker=dict(size=11, color=MUTED_TEXT, symbol="circle-open",
                            line=dict(width=2)),
                customdata=others[[NAME]].to_numpy(),
                hovertemplate="<b>%{customdata[0]}</b><br>%{x:.4f} au<extra></extra>",
                showlegend=False,
            ))
        hz.add_trace(go.Scatter(
            x=[a_planet], y=[0.5], mode="markers+text",
            marker=dict(size=19, color=chip, symbol="circle",
                        line=dict(width=2, color="rgba(128,128,128,0.65)")),
            text=[pick], textposition="top center",
            textfont=dict(size=13, color=CHART_FONT),
            hovertemplate="<b>" + pick + "</b><br>%{x:.4f} au<extra></extra>",
            showlegend=False,
        ))
        hz.add_trace(go.Scatter(
            x=[min(inner_o, a_planet) * 0.5], y=[0.5], mode="markers+text",
            marker=dict(size=17, color="#f5c065", symbol="star"),
            text=[planet[HOST]], textposition="bottom center",
            textfont=dict(size=11, color=MUTED_TEXT),
            hoverinfo="skip", showlegend=False,
        ))

        transparent(
            hz, height=250, margin=dict(t=44, b=10, l=10, r=10),
            xaxis_title="Distance from star (au, log scale)", showlegend=False,
        )
        hz.update_xaxes(type="log")
        hz.update_yaxes(visible=False, range=[0, 1])
        hz.add_annotation(
            x=np.log10(np.sqrt(inner_c * outer_c)), y=0.86,
            text="conservative habitable zone", showarrow=False,
            font=dict(size=12, color=HZ_COLORS["Conservative"]),
        )
        st.plotly_chart(hz, width="stretch", config=CHART_CONFIG)
        st.caption(
            f"Conservative zone {inner_c:.3f}-{outer_c:.3f} au, "
            f"optimistic {inner_o:.3f}-{outer_o:.3f} au. "
            f"This planet orbits at {a_planet:.4f} au."
        )

    system_panel = st.expander("System diagram", on_change="rerun")
    if system_panel.open:
        with system_panel:
            system_orbits = df[(df[HOST] == planet[HOST]) & df[SEMI_MAJOR].gt(0)].sort_values(SEMI_MAJOR)
            if system_orbits.empty:
                st.info("The archive has no orbital distances for this system.")
            else:
                diagram = go.Figure()
                angles = np.linspace(0, 2 * np.pi, 100)
                for _, world in system_orbits.iterrows():
                    radius = np.log10(1 + world[SEMI_MAJOR] * 10)
                    diagram.add_scatter(x=radius * np.cos(angles), y=radius * np.sin(angles),
                                        mode="lines", name=world[NAME],
                                        hovertemplate=f"{html.escape(world[NAME])} · {world[SEMI_MAJOR]:.3g} au<extra></extra>")
                diagram.add_scatter(x=[0], y=[0], mode="markers", name=str(planet[HOST]),
                                    marker=dict(color="#f5c065", size=12, symbol="star"))
                transparent(diagram, height=360, xaxis_visible=False, yaxis_visible=False,
                            yaxis_scaleanchor="x", yaxis_scaleratio=1, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(diagram, config=CHART_CONFIG)
                st.caption("Schematic with compressed distances. Circular paths illustrate the relative ordering of published semi-major axes.")
                st.dataframe(system_orbits[[NAME, SEMI_MAJOR]], hide_index=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("##### Size against Earth")
        r = planet[RADIUS]
        if pd.isna(r) or r <= 0:
            st.info("No published radius for this planet.")
        else:
            names, values, colors = ["Earth", pick], [1.0, float(r)], ["#4C9BE8", chip]
            if rival is not None and pd.notna(rival[RADIUS]) and rival[RADIUS] > 0:
                names.append(rival_pick)
                values.append(float(rival[RADIUS]))
                colors.append(HZ_COLORS.get(rival[HZ_STATUS], HZ_COLORS["Unknown"]))
            size = go.Figure(go.Bar(
                y=names, x=values, orientation="h", marker_color=colors,
                text=[f"{v:.2f}" for v in values], textposition="outside",
                textfont=dict(color=CHART_FONT),
                hovertemplate="%{y}: %{x:.2f} Earth radii<extra></extra>",
            ))
            transparent(size, height=190 + 30 * (len(names) - 2),
                        margin=dict(t=10, b=10, l=10, r=54),
                        showlegend=False, xaxis_title="Radius (Earth radii)")
            size.update_xaxes(range=[0, max(values) * 1.3])
            st.plotly_chart(size, width="stretch", config=CHART_CONFIG)

    with c_right:
        st.markdown("##### The rest of this system")
        system = df[df[HOST] == planet[HOST]]
        if len(system) == 1:
            st.info(
                f"NASA lists one confirmed planet for {planet[HOST]}."
            )
        else:
            st.dataframe(
                system.sort_values(SEMI_MAJOR)[
                    [NAME, RADIUS, SEMI_MAJOR, TEMP, HZ_STATUS]
                ],
                hide_index=True, width="stretch", height=190,
                column_config={
                    NAME: "Planet",
                    RADIUS: st.column_config.NumberColumn("R" + EARTH_SYM, format="%.2f"),
                    SEMI_MAJOR: st.column_config.NumberColumn("au", format="%.3f"),
                    TEMP: st.column_config.NumberColumn("Eq. T (K)", format="%.0f"),
                    HZ_STATUS: "Zone",
                },
            )

    comparison_names = list(dict.fromkeys([
        pick, *([rival_pick] if rival is not None else []), *tray,
    ]))[:COMPARISON_LIMIT]
    if len(comparison_names) > 1 or selected:
        st.markdown("### Compare with Earth")
        comparison_rows = df.set_index(NAME).loc[comparison_names]
        measure_specs = [
            (RADIUS, f"Radius (R{size_suffix})", 1 / radius_divisor, 1.0, ".3g"),
            (MASS, f"Mass (M{size_suffix})", 1 / mass_divisor, 1.0, ".3g"),
            (PERIOD, "Orbital period (days)", 1.0, 365.25, ",.2f"),
            (SEMI_MAJOR, "Semi-major axis (au)", 1.0, 1.0, ".4f"),
            (TEMP, "Equilibrium temperature (K)", 1.0, EARTH_EQ_TEMP, ".0f"),
            (INSOLATION, "Insolation (Earth = 1)", 1.0, 1.0, ".2f"),
            (ESI, "Earth similarity", 1.0, 1.0, ".3f"),
            (DISTANCE, f"Distance ({distance_suffix})", distance_factor, 0.0, ",.1f"),
        ]
        values = {
            "Measure": [label for _, label, *_ in measure_specs] + ["Habitable zone"],
            "Earth": [earth * factor for _, _, factor, earth, _ in measure_specs] + ["Conservative zone"],
        }
        for name, row in comparison_rows.iterrows():
            values[name] = [row[column] * factor for column, _, factor, _, _ in measure_specs] + [
                zone_label(row[HZ_STATUS])
            ]
        formats = [spec for _, _, _, _, spec in measure_specs]
        comparison = {
            name: [measurement(value, spec) for value, spec in zip(column, formats)] + [column[-1]]
            for name, column in values.items() if name != "Measure"
        }
        comparison = {"Measure": values["Measure"], **comparison}
        layout = st.radio("Comparison layout", ["One at a time", "All planets"], horizontal=True)
        if layout == "One at a time":
            against = st.selectbox("Show planet beside Earth", comparison_names)
            table = {key: comparison[key] for key in ["Measure", "Earth", against]}
        else:
            table = comparison
        st.dataframe(pd.DataFrame(table), hide_index=True, column_config={"Measure": st.column_config.TextColumn(pinned=True)})
        st.caption("Earth reference temperature: 255 K equilibrium temperature.")
        st.download_button("Download full comparison (CSV)", pd.DataFrame(values).to_csv(index=False),
                           file_name="exoplanet_comparison.csv", mime="text/csv", on_click="ignore")
        st.caption("Share this page’s URL to open the same planet selection.")
        export_panel = st.expander("Export comparison as an image", on_change="rerun")
        if export_panel.open:
            summary = go.Figure(go.Table(
                header=dict(values=list(table), fill_color=SIDEBAR_BG, font=dict(color=TEXT_COLOR)),
                cells=dict(values=list(table.values()), fill_color=CARD_BG, font=dict(color=TEXT_COLOR)),
            ))
            summary.update_layout(title="Exoplanet comparison · NASA Exoplanet Archive", height=350,
                                  paper_bgcolor=PAGE_BG, font_color=TEXT_COLOR, margin=dict(l=0, r=0, t=50, b=0))
            export_panel.caption("Choose Download plot as a PNG in the chart toolbar.")
            export_panel.plotly_chart(summary, config=CHART_CONFIG)

    with st.expander("Measurements and sources"):
        st.caption("NASA publishes uncertainty ranges and source links for some values. "
                   "Mass can be measured, a minimum mass, or estimated.")
        for column, label, raw in zip(CORE_COLUMNS, CORE_LABELS, [
            "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt", "sy_dist", "pl_orbsmax",
        ]):
            lower, upper = planet.get(raw + "err2"), planet.get(raw + "err1")
            measured = measurement(planet[column], ".5g")
            deviations = "" if pd.isna(lower) or pd.isna(upper) else f" ({lower:+.3g}, {upper:+.3g})"
            source = archive_reference(planet.get(raw + "_reflink"))
            st.write(f"**{label}** · {measured}{deviations} · {column}")
            if source:
                st.link_button(f"Source for {label.lower()}", source)

    similarity_panel = st.expander("Find similar planets", on_change="rerun")
    if similarity_panel.open:
        with similarity_panel:
            features = st.multiselect("Compare measurements", [RADIUS, TEMP, PERIOD], default=[RADIUS, TEMP])
            if features and planet[features].notna().all() and planet[features].gt(0).all():
                eligible = df[df[features].gt(0).all(axis=1) & df[NAME].ne(pick)].copy()
                ratios = eligible[features].divide(planet[features].astype(float), axis="columns")
                similar = eligible.loc[np.log(ratios).pow(2).mean(axis=1).nsmallest(5).index]
                st.caption("Ranked by relative differences in the selected measurements.")
                for other in similar[NAME]:
                    st.button(f"Open similar planet {other}", on_click=open_profile, args=(other, True))
            else:
                st.info("Choose measurements that this planet has published.")

    summary_panel = st.expander("Planet summary image", on_change="rerun")
    if summary_panel.open:
        with summary_panel:
            summary_items = [
                f"{planet[HOST]} system · Discovered {int(planet[YEAR])}",
                f"Radius: {measurement(planet[RADIUS] / radius_divisor, '.3g', f' R{size_suffix}')}",
                f"Mass: {measurement(planet[MASS] / mass_divisor, '.3g', f' M{size_suffix}')}",
                f"Distance: {measurement(planet[DISTANCE] * distance_factor, ',.1f', f' {distance_suffix}')}",
                zone_label(status),
            ]
            summary = go.Figure(go.Table(
                header=dict(values=[html.escape(pick)], fill_color=SIDEBAR_BG, font=dict(color=TEXT_COLOR)),
                cells=dict(values=[[html.escape(item) for item in summary_items]],
                           fill_color=CARD_BG, font=dict(color=TEXT_COLOR)),
            ))
            summary.update_layout(title="NASA Exoplanet Archive", height=250, paper_bgcolor=PAGE_BG,
                                  font_color=TEXT_COLOR, margin=dict(l=0, r=0, t=50, b=0))
            st.caption("Choose Download plot as a PNG in the chart toolbar.")
            st.plotly_chart(summary, config=CHART_CONFIG)

    st.link_button(
        "Open this planet in the NASA Exoplanet Archive",
        "https://exoplanetarchive.ipac.caltech.edu/overview/"
        + quote(str(pick), safe=""),
    )

if st.session_state["active_view"] == "Learn":
    st.subheader("A field guide to other worlds")
    with st.expander("Guided discoveries"):
        render_guided_discoveries("learn")

    st.markdown("You don’t need an astronomy degree to explore. Start with these essentials.")
    for title, explanation in [
        ("How do we find planets?", "**Transits** are tiny dips in starlight when a planet crosses its star. "
         "**Radial velocity** measures a star’s wobble from a planet’s gravity. "
         "**Imaging** captures light from a planet directly; **microlensing** uses gravitational magnification. "
         "Different methods favour different worlds, so the catalogue is not an unbiased census."),
        ("What does Earth similarity mean?", "The Earth Similarity Index here combines radius and equilibrium "
         "temperature. One means a match to those two Earth values. It does not measure atmosphere, water or life."),
        ("What is a habitable zone?", "An estimated range of orbital distances where an Earth-like atmosphere "
         "could allow liquid surface water. Conservative and optimistic bands use different model limits. "
         "Being in the zone does not establish habitability."),
        ("How do I read the units?", "**R⊕ / M⊕:** Earth radii / Earth masses. **au:** Earth’s average distance "
         "from the Sun. **pc:** parsecs; multiply by 3.26 for light-years. **K:** kelvin; subtract 273.15 for Celsius. "
         "Equilibrium temperature is a radiation-balance estimate, not a measured surface temperature."),
        ("Why are some values missing?", "Many discoveries constrain only some measurements. Unknown values "
         "are shown as dashes; charts exclude planets missing their required axes. An ESI threshold excludes "
         "unscored planets. K-Means groups are exploratory statistical clusters, not official planet classes."),
        ("What can I do with the charts?", "Hover or tap a point for details. Use the chart toolbar to zoom, "
         "reset or download an image. Use the filters for a focused selection and download its CSV from the "
         "sidebar. The 3D view samples large selections for performance."),
    ]:
        with st.expander(title):
            st.markdown(explanation)
    st.link_button("Explore the NASA source catalogue", ARCHIVE_HOME)

st.divider()
st.caption(
    "Data provided by the NASA Exoplanet Archive, operated by the California Institute "
    "of Technology under contract with the National Aeronautics and Space Administration."
)
