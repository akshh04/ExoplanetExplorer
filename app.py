"""Exoplanet Explorer — an interactive dashboard over the NASA Exoplanet Archive."""

import hashlib
import inspect
import threading
import time
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from streamlit.runtime.scriptrunner import add_script_run_ctx

st.set_page_config(
    page_title="Exoplanet Explorer",
    layout="wide",
    page_icon="🪐",
    # "auto" keeps the sidebar open on desktop but collapsed on phones,
    # where 300px of filters would otherwise cover most of the screen.
    initial_sidebar_state="auto",
    menu_items={
        "About": (
            "Interactive exploration of the NASA Exoplanet Archive "
            "(Planetary Systems Composite Parameters table)."
        )
    },
)

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
    GRADIENT = "linear-gradient(95deg, #38bdf8 0%, #a78bfa 55%, #f472b6 100%)"
    GRADIENT_FALLBACK = "#a78bfa"
    CHART_FONT, CHART_GRID = "#e6edf3", "rgba(230,237,243,0.15)"
    ACCENT_SOFT, SHADOW = "rgba(167,139,250,0.55)", "rgba(0,0,0,0.45)"
    HOVER_TINT, FOCUS_RING = "rgba(230,237,243,0.07)", "#a78bfa"
    TAB_BAR_BG, LINK_COLOR = "rgba(14,17,23,0.85)", "#7dd3fc"
    TAB_ACTIVE_BG = "rgba(167,139,250,0.18)"
    CHIP_BG = "rgba(230,237,243,0.07)"
    CHIP_ACTIVE_BG, CHIP_ACTIVE_FG = "#a78bfa", "#0e1117"
else:
    PAGE_BG, SIDEBAR_BG, CARD_BG = "#ffffff", "#f0f2f6", "rgba(128,128,128,0.08)"
    TEXT_COLOR, MUTED_TEXT, BORDER = "#31333f", "#5c6470", "rgba(49,51,63,0.16)"
    INPUT_BG = "#ffffff"
    GRADIENT = "linear-gradient(95deg, #0284c7 0%, #7c3aed 55%, #db2777 100%)"
    GRADIENT_FALLBACK = "#7c3aed"
    CHART_FONT, CHART_GRID = "#31333f", "rgba(49,51,63,0.12)"
    ACCENT_SOFT, SHADOW = "rgba(124,58,237,0.45)", "rgba(15,23,42,0.10)"
    HOVER_TINT, FOCUS_RING = "rgba(49,51,63,0.05)", "#7c3aed"
    TAB_BAR_BG, LINK_COLOR = "rgba(255,255,255,0.85)", "#0284c7"
    TAB_ACTIVE_BG = "rgba(124,58,237,0.12)"
    CHIP_BG = "rgba(49,51,63,0.06)"
    CHIP_ACTIVE_BG, CHIP_ACTIVE_FG = "#7c3aed", "#ffffff"

_dark_overrides = f"""
    html, body, .stApp, [data-testid="stHeader"], [data-testid="stAppViewContainer"] {{
        background-color: {PAGE_BG} !important;
        color: {TEXT_COLOR} !important;
    }}
    [data-testid="stSidebar"] {{
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
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"],
    textarea, input[type="text"] {{
        background-color: {INPUT_BG} !important;
        color: {TEXT_COLOR} !important;
        border-color: {BORDER} !important;
    }}
    [data-baseweb="tag"] {{ background-color: #334155 !important; }}
    [role="listbox"], [data-baseweb="popover"] [role="listbox"] {{ background-color: {INPUT_BG} !important; }}
    [data-testid="stTabs"] [role="tab"] {{ color: {MUTED_TEXT} !important; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{ color: {TEXT_COLOR} !important; }}
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
        font-size: 3rem !important;
        font-weight: 800;
        line-height: 1.15;
        margin: 0;
        padding-bottom: 0.2rem;
        background: {GRADIENT};
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
    }}
    /* Without background-clip:text the rules above would render the title
       invisible, so fall back to a solid colour. */
    @supports not ((-webkit-background-clip: text) or (background-clip: text)) {{
        h1.hero-title {{
            background: none;
            color: {GRADIENT_FALLBACK};
            -webkit-text-fill-color: {GRADIENT_FALLBACK};
        }}
    }}
    .hero-sub {{
        font-size: 1.05rem;
        opacity: 0.8;
        margin-top: -0.2rem;
    }}

    /* Tabs a little roomier than the default. NOTE: Streamlit renders each
       tab as <div data-testid="stTab" role="tab">, not a <button> -- a
       button[role="tab"] selector matches nothing. */
    [data-testid="stTabs"] [role="tab"] {{
        padding: 0.55rem 1.05rem;
        font-size: 0.98rem;
    }}

    /* Top-right theme toggle, level with the title. `vertical_alignment`
       centres the column boxes, but Streamlit collapses the title's column
       to a few pixels, so the glyphs ended up 26px below the switch on
       desktop. Centring the column *content* is what actually lines them up. */
    [data-testid="stHorizontalBlock"] div[data-testid="stToggle"] {{
        display: flex;
        justify-content: flex-end;
    }}
    .st-key-header-row [data-testid="stHorizontalBlock"] {{
        align-items: center;
    }}
    .st-key-header-row [data-testid="stColumn"],
    .st-key-header-row [data-testid="stColumn"] > div,
    .st-key-header-row [data-testid="stColumn"] [data-testid="stElementContainer"] {{
        display: flex;
        align-items: center;
    }}
    .st-key-header-row [data-testid="stColumn"] [data-testid="stMarkdownContainer"] {{
        width: 100%;
    }}
    .st-key-header-row h1.hero-title {{
        padding-top: 0;
        padding-bottom: 0;
    }}
    /* Streamlit's toggle label pins its switch to the top with
       align-items:flex-start, leaving the switch ~10px above the centre of
       its own 42px box and out of line with the title. */
    .st-key-header-row [data-testid="stCheckbox"] label {{
        align-items: center;
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

    /* Section sub-headings inside tabs. */
    [data-testid="stMarkdownContainer"] h5 {{
        margin-top: 1.4rem;
        margin-bottom: 0.35rem;
        font-size: 1.02rem;
        font-weight: 650;
        opacity: 0.9;
    }}

    /* Metric cards lift slightly on hover so they read as inspectable
       objects rather than static text. */
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

    /* Keep the tab bar reachable on the long tabs. */
    [data-testid="stTabs"] div[role="tablist"] {{
        position: sticky;
        top: 2.6rem;
        z-index: 5;
        backdrop-filter: blur(8px);
        background: {TAB_BAR_BG};
        border-bottom: 1px solid {BORDER};
    }}
    [data-testid="stTabs"] [role="tab"] {{
        transition: color 120ms ease, background-color 120ms ease;
        border-radius: 8px 8px 0 0;
    }}
    [data-testid="stTabs"] [role="tab"]:hover {{
        background-color: {HOVER_TINT};
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

        /* Sized to keep "Exoplanet Explorer" on one line in the ~250px left
           beside the toggle: at 1.95rem it wrapped to two lines, and even
           1.45rem ran 5px into the switch. */
        /* The title measures 232px against 246px of column yet still broke
           to two lines, dropping its centre below the toggle -- nowrap pins
           it to one. Streamlit's h1 top padding goes for the same reason. */
        h1.hero-title {{
            font-size: 1.3rem !important;
            line-height: 1.2;
            padding-top: 0;
            padding-bottom: 0;
            white-space: nowrap;
        }}
        .hero-sub {{ font-size: 0.92rem; }}

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
        [data-testid="stMetricDelta"] {{ font-size: 0.68rem; }}
        /* Stack value over delta: side by side is too tight at this width. */
        [data-testid="stMetric"] > div {{
            grid-template-columns: 1fr;
        }}
        [data-testid="stMetric"] > div > div:has([data-testid="stMetricDelta"]) {{
            grid-row: 3;
            grid-column: 1;
        }}

        /* Nav: six tabs ran to 704px inside a 343px strip, so the last two --
           one of them the planet profile -- sat off-screen with no affordance
           that they existed. Rebuilt as a 3x2 grid of equal pills: everything
           is visible at once, each is a proper tap target, and the selected
           one is filled rather than marked by a 2px underline. */
        [data-testid="stTabs"] div[role="tablist"] {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.45rem;
            position: static;      /* two rows of chips are too tall to pin */
            backdrop-filter: none;
            background: none;
            border-bottom: none;
            padding: 0.15rem 0 0.35rem 0;
            /* Streamlit's horizontal-scroll styling clips the second row. */
            overflow: visible !important;
        }}
        /* Streamlit paints its underline rail as a 2px ::after bar, which no
           border property can remove -- it left a second line running under
           the pill grid. */
        [data-testid="stTabs"] div[role="tablist"]::after {{
            display: none !important;
        }}
        /* !important throughout: Streamlit's own emotion rules for tabs
           out-specify a plain [data-testid] selector, so without it the chips
           come out transparent and auto-height.
           No border: two bordered rows put a line 6px above another line
           straight across the nav, which read as a double border. The fill
           alone separates the chips. */
        [data-testid="stTabs"] [role="tab"] {{
            width: 100% !important;
            min-height: 44px !important;
            margin: 0 !important;
            padding: 0.4rem 0.3rem !important;
            font-size: 0.79rem !important;
            line-height: 1.15 !important;
            text-align: center;
            justify-content: center;
            border: 0 !important;
            border-radius: 10px !important;
            background: {CHIP_BG} !important;
        }}
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
            background: {CHIP_ACTIVE_BG} !important;
            color: {CHIP_ACTIVE_FG} !important;
            font-weight: 650 !important;
        }}
        /* Hover tooltips cannot be hovered on a touch screen: the "?" icons
           only steal room beside the labels, and their help text is already
           repeated in the captions. */
        [data-testid="stTooltipIcon"],
        [data-testid="stTooltipHoverTarget"] {{
            display: none !important;
        }}

        /* Keep the theme toggle on the title's row. Left to itself Streamlit
           stacks the two columns and the switch eats a full 348px band
           between the heading and the standfirst. */
        .st-key-header-row [data-testid="stHorizontalBlock"] {{
            flex-wrap: nowrap !important;
            gap: 0.4rem;
            min-height: 44px;
        }}
        .st-key-header-row [data-testid="stColumn"] {{
            min-width: 0 !important;
            width: auto !important;
        }}
        .st-key-header-row [data-testid="stColumn"]:first-child {{
            flex: 1 1 auto !important;
        }}
        .st-key-header-row [data-testid="stColumn"]:last-child {{
            flex: 0 0 auto !important;
        }}
        .st-key-header-row [data-testid="stToggle"] {{
            padding-top: 0;
        }}
        .st-key-header-row [data-testid="stToggle"] label {{
            font-size: 0.74rem;
            white-space: nowrap;
        }}

        /* Sub-headings and section spacing scale down with everything else. */
        [data-testid="stMarkdownContainer"] h5 {{ margin-top: 1rem; }}
        .profile-name {{ font-size: 1.35rem; }}
        .profile-sub {{ font-size: 0.88rem; margin-bottom: 0.8rem; }}
        .profile-head {{ gap: 0.45rem; }}

        /* Tables: scroll inside their own box rather than stretching the page. */
        [data-testid="stDataFrame"] {{ max-width: 100%; overflow-x: auto; }}

        /* Full-width tap targets for the download/link buttons. */
        [data-testid="stDownloadButton"] button,
        [data-testid="stLinkButton"] a {{ width: 100%; }}
    }}

    /* Only the very narrowest phones need another step down; from ~360px up
       the title clears the toggle at 1.3rem with room to spare. */
    @media (max-width: 340px) {{
        h1.hero-title {{ font-size: 1rem !important; }}
        .st-key-header-row [data-testid="stToggle"] label {{ font-size: 0.68rem; }}
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


# The archive answers with Transfer-Encoding: chunked and no Content-Length,
# so there is no true percentage to report. The bar is paced against a typical
# payload size while the caption reports the real number of bytes received.
ESTIMATED_PAYLOAD_BYTES = 850_000


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_exoplanet_table(_sink=None) -> pd.DataFrame:
    """Fetch the Planetary Systems Composite Parameters table from NASA.

    `_sink` is an optional plain dict that receives progress as it goes. It
    must never be a Streamlit element: Streamlit replays whatever a cached
    function draws, and replaying a widget bound to a container from an
    earlier run raises CacheReplayClosureError. The leading underscore also
    keeps the dict out of the cache key.
    """

    def report(fraction: float, message: str) -> None:
        if _sink is not None:
            _sink["fraction"] = fraction
            _sink["message"] = message

    query = """
    select pl_name, hostname, discoverymethod, disc_year, pl_rade, pl_bmasse,
           pl_orbper, pl_orbsmax, pl_eqt, st_teff, st_lum, sy_dist
    from pscomppars
    """
    report(0.04, "Opening a connection to the archive…")
    response = requests.get(
        ARCHIVE_URL,
        params={"query": query, "format": "csv"},
        timeout=(10, 120),  # (connect, read) — without this the app can hang forever
        stream=True,
    )
    response.raise_for_status()

    # The server runs the query before it sends anything, so the first bytes
    # arrive several seconds in; that wait is covered by the step above.
    report(0.18, "Query accepted — receiving the table…")
    parts, received, last_tick = [], 0, 0.0
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
            share = min(received / ESTIMATED_PAYLOAD_BYTES, 1.0)
            report(0.18 + 0.62 * share, f"Receiving… {received / 1024:,.0f} KB")

    report(0.82, f"Received {received / 1024:,.0f} KB — parsing…")
    text = b"".join(parts).decode(response.encoding or "utf-8", errors="replace")
    if not text.lstrip().lower().startswith("pl_name"):
        # TAP reports query errors with a 200 and a plain-text body.
        raise RuntimeError(f"Archive returned an unexpected response: {text[:300]}")

    df = pd.read_csv(StringIO(text))
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

    df[YEAR] = pd.to_numeric(df[YEAR], errors="coerce").astype("Int64")
    df = df.dropna(subset=[YEAR])
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
        _sink.update(fraction=0.92, message="Computing habitable zones…")
    df = add_habitability_columns(df)
    if _sink is not None:
        _sink.update(fraction=1.0, message=f"Ready — {len(df):,} planets")
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


CHART_CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}

# Plotly bakes a pixel height into the figure and CSS cannot resize the SVG,
# so these are single values chosen to stay comfortable on a phone while still
# looking generous on a desktop.
CHART_H_TALL = 460
CHART_H_SCATTER = 510
CHART_H_3D = 520


# --- HEADER --------------------------------------------------------------
header_row = st.container(key="header-row")
header_col, toggle_col = header_row.columns([6, 1], vertical_alignment="center")
with toggle_col:
    st.toggle("🌙 Dark", key="dark_mode", help="Switch between light and dark mode")
with header_col:
    st.markdown('<h1 class="hero-title">🪐 Exoplanet Explorer</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="hero-sub">Live data on every confirmed world beyond our solar system, '
    f'straight from the <a href="{ARCHIVE_HOME}">NASA Exoplanet Archive</a>.</p>',
    unsafe_allow_html=True,
)

def load_with_progress():
    """Fetch on a worker thread so this thread can keep painting the bar.

    Every Streamlit call stays here on the script thread; the worker only
    writes numbers into `shared`.
    """
    shared = {"fraction": 0.0, "message": "Contacting the archive…"}
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

    bar = None
    while thread.is_alive():
        if bar is None:
            box = st.empty().container()
            status = box.status(
                "Fetching live data from the NASA Exoplanet Archive", expanded=True
            )
            status.caption(
                "The archive runs the query before it starts sending, so the "
                "first few seconds are the server working rather than the "
                "download."
            )
            holder = status.empty()
            bar = holder.progress(0.0, text=shared["message"])
        bar.progress(
            min(max(float(shared["fraction"]), 0.0), 1.0), text=str(shared["message"])
        )
        thread.join(timeout=0.1)

    if "error" in result:
        raise result["error"]
    return result["df"]


loader = st.empty()
with loader.container():
    try:
        df = load_with_progress()
    except requests.Timeout:
        loader.empty()
        st.error(
            "**The NASA Exoplanet Archive did not respond in time.** "
            "The service is occasionally slow under load."
        )
        if st.button("Try again", type="primary"):
            fetch_exoplanet_table.clear()
            load_exoplanet_data.clear()
            st.rerun()
        st.stop()
    except Exception as exc:
        loader.empty()
        st.error("**Could not load data from the NASA Exoplanet Archive.**")
        st.caption(f"{type(exc).__name__}: {exc}")
        if st.button("Try again", type="primary"):
            fetch_exoplanet_table.clear()
            load_exoplanet_data.clear()
            st.rerun()
        st.stop()
# Drops the whole loading block; a no-op when nothing was drawn.
loader.empty()

COLOR_MAP = method_colors(df[METHOD_SHORT].unique())
YEAR_MIN, YEAR_MAX = int(df[YEAR].min()), int(df[YEAR].max())
ALL_METHODS = sorted(df[METHOD].unique())
MAX_DIST = float(np.ceil(df[DISTANCE].max() / 100) * 100)

FILTER_DEFAULTS = {
    "year_range": (YEAR_MIN, YEAR_MAX),
    "methods": ALL_METHODS,
    "dist_limit": MAX_DIST,
    "keep_unknown_dist": True,
    "hz_filter": [],
    "esi_floor": 0.0,
}


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
for _key, _default in FILTER_DEFAULTS.items():
    st.session_state.setdefault(_key, _default)

# A data refresh can move the bounds under stored state (a new discovery year,
# a newly catalogued far-away planet), so re-fit what is already there.
_lo, _hi = st.session_state["year_range"]
st.session_state["year_range"] = (max(YEAR_MIN, _lo), min(YEAR_MAX, _hi))
st.session_state["dist_limit"] = min(max(0.0, st.session_state["dist_limit"]), MAX_DIST)
st.session_state["methods"] = [m for m in st.session_state["methods"] if m in ALL_METHODS]
st.session_state["hz_filter"] = [h for h in st.session_state["hz_filter"] if h in HZ_ORDER]

# --- SIDEBAR FILTERS -----------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">Filters</div>'
        '<div class="sidebar-brand-sub">Applied to every chart and table.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("🔭  Discovery", expanded=True):
        year_range = st.slider(
            "Discovery year",
            min_value=YEAR_MIN,
            max_value=YEAR_MAX,
            key="year_range",
        )
        all_methods = ALL_METHODS
        selected_methods = st.multiselect(
            "Discovery method",
            options=ALL_METHODS,
            placeholder="All methods",
            key="methods",
        )

    with st.expander("📍  Distance", expanded=True):
        max_dist = MAX_DIST
        dist_limit = st.slider(
            "Max distance (parsecs)",
            min_value=0.0,
            max_value=max_dist,
            step=50.0,
            key="dist_limit",
        )
        keep_unknown_dist = st.checkbox(
            "Keep planets with unknown distance",
            key="keep_unknown_dist",
            help=f"{df[DISTANCE].isna().sum():,} planets have no published distance.",
        )

    with st.expander("🌍  Habitability", expanded=False):
        hz_filter = st.multiselect(
            "Habitable-zone class",
            options=HZ_ORDER,
            placeholder="Any class",
            key="hz_filter",
            help=HZ_HELP,
        )
        esi_floor = st.slider(
            "Minimum Earth Similarity Index",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="esi_floor",
            help=(
                "0 keeps everything. Planets with no published radius or "
                "temperature have no ESI and drop out above 0."
            ),
        )

    st.button("Reset filters", width="stretch", on_click=reset_filters)

    st.divider()
    if st.button("🔄 Refresh data from NASA", width="stretch"):
        fetch_exoplanet_table.clear()
        load_exoplanet_data.clear()
        st.rerun()
    st.caption("Data is cached for 24 hours.")

# Apply the global filters.
methods_filter = selected_methods or all_methods
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
    active_filters.append(f"within {dist_limit:,.0f} pc")
if not keep_unknown_dist:
    active_filters.append("known distance only")
if hz_filter:
    active_filters.append(" / ".join(hz_filter) + " HZ")
if esi_floor > 0:
    active_filters.append(f"ESI ≥ {esi_floor:.2f}")

if fdf.empty:
    detail = (" · ".join(active_filters)) if active_filters else ""
    st.warning("**No planets match the current filters.**")
    if detail:
        st.caption(f"Active filters: {detail}")
    st.info("Widen the filters in the sidebar, or press **Reset filters**, to continue.")
    st.stop()

# --- HEADLINE METRICS ----------------------------------------------------
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

if active_filters:
    st.caption(
        f"**{len(fdf):,}** of {len(df):,} planets · filtered by "
        + " · ".join(active_filters)
    )

st.divider()

(
    tab_trends,
    tab_sizes,
    tab_ml,
    tab_3d,
    tab_hz,
    tab_profile,
) = st.tabs(
    [
        "📈 Trends",
        "🌌 Size & orbit",
        "🤖 Classes",
        "🧭 3D view",
        "💧 Habitable",
        "🔭 Profile",
    ]
)

# --- TAB 1: DISCOVERY TRENDS --------------------------------------------
with tab_trends:
    st.subheader("The pace of discovery")
    st.markdown(
        "Since the first confirmed planets in the 1990s, the rate of discovery has "
        "skyrocketed — driven largely by space telescopes using the transit method."
    )

    view = st.radio(
        "View",
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

    if view == "Cumulative":
        pivot = yearly.pivot(index=YEAR, columns=METHOD_SHORT, values="Count").fillna(0)
        pivot = pivot.reindex(range(year_range[0], year_range[1] + 1), fill_value=0).cumsum()
        plot_df = pivot.reset_index().melt(
            id_vars=YEAR, var_name=METHOD_SHORT, value_name="Count"
        )
        fig = px.area(plot_df, x=YEAR, y="Count", color=METHOD_SHORT, color_discrete_map=COLOR_MAP)
        y_title = "Planets known (cumulative)"
    else:
        plot_df = yearly.copy()
        y_title = "Planets discovered"
        if view == "Share of year (%)":
            totals = plot_df.groupby(YEAR)["Count"].transform("sum")
            plot_df["Count"] = plot_df["Count"] / totals * 100
            y_title = "Share of discoveries (%)"
        fig = px.bar(plot_df, x=YEAR, y="Count", color=METHOD_SHORT, color_discrete_map=COLOR_MAP)

    transparent(
        fig,
        xaxis_title="Year",
        yaxis_title=y_title,
        hovermode="x unified",
        margin=dict(t=10, b=0, l=0, r=0),
        height=CHART_H_TALL,
    )
    fig.update_xaxes(dtick=5, tickformat="d")
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
            "Share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
        },
    )

# --- TAB 2: SIZE & ORBIT -------------------------------------------------
with tab_sizes:
    st.subheader("Orbital period vs. planet size")
    st.markdown(
        "Each dot is a world; our own planets are overlaid as stars for scale. "
        "Look for the **hot Jupiters** at the top left and the dense band of "
        "**super-Earths** through the middle."
    )

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        radius_range = st.slider(
            "Planet radius (Earth radii)", 0.0, 30.0, (0.0, 30.0), 0.5
        )
    with ctrl2:
        period_limit = st.select_slider(
            "Max orbital period (days)",
            options=[10, 100, 1_000, 5_000, 10_000, 100_000],
            value=10_000,
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
            hover_data={TEMP: ":.0f", DISTANCE: ":.1f"},
            log_x=True,
            log_y=True,
            opacity=0.55,
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))

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

        transparent(fig, margin=dict(t=10, b=0, l=0, r=0), height=CHART_H_SCATTER)
        fig.update_xaxes(title_text="Orbital period (days, log scale)")
        fig.update_yaxes(title_text="Radius (Earth radii, log scale)")
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

        cap = f"{len(scatter_df):,} planets plotted."
        if hidden:
            cap += f" Outside the current window: {', '.join(hidden)}."
        st.caption(cap)

# --- TAB 3: K-MEANS CLASSIFICATION ---------------------------------------
with tab_ml:
    st.subheader("Unsupervised classification (K-Means)")
    st.markdown(
        "Rather than hand-defining what a *hot Jupiter* or *super-Earth* is, K-Means "
        "finds natural groupings in log mass–radius space. Groups are numbered by "
        "median mass, so the labels stay consistent as the archive grows."
    )

    ml_df = fdf.dropna(subset=[MASS, RADIUS]).copy()
    ml_df = ml_df[(ml_df[MASS] > 0) & (ml_df[RADIUS] > 0)]

    n_clusters = st.slider("Number of groups (k)", 2, 6, 3)

    if len(ml_df) <= n_clusters:
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

# --- TAB 4: 3D EXPLORER --------------------------------------------------
with tab_3d:
    st.subheader("Three-dimensional view")
    st.markdown(
        "Rotate and zoom to see how radius, orbital period and equilibrium "
        "temperature relate. Hot, close-in planets sit at the front-left."
    )

    td_df = fdf.dropna(subset=[RADIUS, PERIOD, TEMP]).copy()
    td_df = td_df[(td_df[RADIUS] > 0) & (td_df[PERIOD] > 0)]

    if td_df.empty:
        st.info("No planets in the current selection have all three parameters.")
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
        st.plotly_chart(fig, width="stretch")

        if len(sampled) < len(td_df):
            st.caption(f"Showing a random sample of {len(sampled):,} of {len(td_df):,} planets to keep the 3D view responsive.")
        else:
            st.caption(f"{len(sampled):,} planets plotted.")

# --- TAB 5: HABITABLE WORLDS ---------------------------------------------
with tab_hz:
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

# --- TAB 6: PLANET PROFILE -----------------------------------------------
with tab_profile:
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

    def fmt(value, spec: str, suffix: str = "") -> str:
        """Format a possibly-missing measurement for display."""
        if pd.isna(value):
            return "-"
        return f"{value:{spec}}{suffix}"

    status = planet[HZ_STATUS]
    chip = HZ_COLORS.get(status, HZ_COLORS["Unknown"])
    st.markdown(
        f"""<div class="profile-head">
              <span class="profile-name">{planet[NAME]}</span>
              <span class="profile-chip chip-{hz_slug(status)}">
                {status} zone
              </span>
            </div>
            <div class="profile-sub">
              Orbits <b>{planet[HOST]}</b> &middot; found in
              <b>{int(planet[YEAR])}</b> by <b>{planet[METHOD]}</b>
            </div>""",
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Radius", fmt(planet[RADIUS], ".2f", " R" + EARTH_SYM),
              help="Earth radii. Earth = 1.")
    p2.metric("Mass", fmt(planet[MASS], ".2f", " M" + EARTH_SYM),
              help="Earth masses. Earth = 1.")
    p3.metric("Orbital period", fmt(planet[PERIOD], ",.2f", " d"))
    p4.metric(
        "Distance",
        fmt(planet[DISTANCE], ",.1f", " pc"),
        help=(f"{planet[DISTANCE] * 3.26156:,.1f} light-years away."
              if pd.notna(planet[DISTANCE]) else "No published distance."),
    )

    p5, p6, p7, p8 = st.columns(4)
    p5.metric(
        "Equilibrium temp",
        fmt(planet[TEMP], ",.0f", " K"),
        help=(f"{planet[TEMP] - 273.15:,.0f} degrees C"
              if pd.notna(planet[TEMP]) else "No published temperature."),
    )
    p6.metric("Insolation", fmt(planet[INSOLATION], ",.2f", " S" + EARTH_SYM),
              help="Starlight received relative to Earth. Earth = 1.")
    p7.metric("Semi-major axis", fmt(planet[SEMI_MAJOR], ",.4f", " au"))
    p8.metric("Earth Similarity", fmt(planet[ESI], ".3f"),
              help="Two-parameter ESI from radius and equilibrium temperature. 1.0 = Earth.")

    st.markdown("##### Where it orbits, against its star's habitable zone")
    inner_o, inner_c = planet[HZ_INNER_OPT], planet[HZ_INNER]
    outer_c, outer_o = planet[HZ_OUTER], planet[HZ_OUTER_OPT]
    a_planet = planet[SEMI_MAJOR]
    bounds_known = not any(
        pd.isna(v) for v in (inner_o, inner_c, outer_c, outer_o, a_planet)
    ) and a_planet > 0 and inner_o > 0

    if not bounds_known:
        st.info(
            "This system is missing the stellar luminosity, temperature or orbital "
            "distance needed to place a habitable zone."
        )
    else:
        siblings = fdf[(fdf[HOST] == planet[HOST]) & fdf[SEMI_MAJOR].notna()]
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
        system = fdf[fdf[HOST] == planet[HOST]]
        if len(system) == 1:
            st.info(
                f"{planet[HOST]} has just this one known planet "
                "in the current selection."
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

    if rival is not None:
        st.markdown("##### Head to head")

        def cell(row, column, spec: str) -> str:
            """One formatted measurement, or a dash when unpublished."""
            if row is None:
                return "\u2014"
            value = row[column]
            return "\u2014" if pd.isna(value) else f"{value:{spec}}"

        # Earth's own reference row: equilibrium temperature (not the 288 K
        # surface value) so it is comparable with the archive's figures.
        earth = {
            "Radius (R" + EARTH_SYM + ")": "1.00",
            "Mass (M" + EARTH_SYM + ")": "1.00",
            "Orbital period (days)": "365.25",
            "Semi-major axis (au)": "1.0000",
            "Equilibrium temp (K)": f"{EARTH_EQ_TEMP:.0f}",
            "Insolation (S" + EARTH_SYM + ")": "1.00",
            "Earth Similarity": "1.000",
            "Habitable zone": "Conservative",
            "Distance (pc)": "0.0",
            "Discovered": "\u2014",
        }

        def column_for(row) -> list:
            return [
                cell(row, RADIUS, ".2f"),
                cell(row, MASS, ".2f"),
                cell(row, PERIOD, ",.2f"),
                cell(row, SEMI_MAJOR, ",.4f"),
                cell(row, TEMP, ",.0f"),
                cell(row, INSOLATION, ",.2f"),
                cell(row, ESI, ".3f"),
                str(row[HZ_STATUS]),
                cell(row, DISTANCE, ",.1f"),
                f"{int(row[YEAR])} \u00b7 {row[METHOD]}",
            ]

        head_to_head = pd.DataFrame(
            {
                "Measure": list(earth.keys()),
                "Earth": list(earth.values()),
                pick: column_for(planet),
                rival_pick: column_for(rival),
            }
        )
        st.dataframe(head_to_head, hide_index=True, width="stretch",
                     column_config={"Measure": st.column_config.TextColumn(width="medium")})
        st.caption(
            "Earth's row uses its equilibrium temperature (255 K), not its 288 K "
            "surface temperature, so the comparison is like for like."
        )

    st.link_button(
        "Open this planet in the NASA Exoplanet Archive",
        "https://exoplanetarchive.ipac.caltech.edu/overview/"
        + pick.replace(" ", "%20"),
    )

st.divider()
st.caption(
    "Data provided by the NASA Exoplanet Archive, operated by the California Institute "
    "of Technology under contract with the National Aeronautics and Space Administration."
)

# Switching tabs is a client-side change, so a chart built during the last
# script run while its tab was hidden measured a zero-width container and fell
# back to Plotly's 700px default -- badly clipped inside a 348px phone column.
# Re-lay-out any visible chart whose SVG no longer matches its box, once per
# tab click. Guarded by an actual width mismatch so nothing happens on the
# common path.
st.html(
    """
<script>
(function () {
  if (document.__plotlyTabFix) return;   // survive Streamlit reruns
  document.__plotlyTabFix = true;

  const fix = () => {
    const Plotly = window.Plotly;
    if (!Plotly) return;
    document.querySelectorAll('.js-plotly-plot').forEach((el) => {
      if (el.offsetParent === null) return;               // still hidden
      const box = el.getBoundingClientRect().width;
      const svg = el.querySelector('.svg-container');
      if (!svg || box <= 0) return;
      if (Math.abs(svg.getBoundingClientRect().width - box) > 4) {
        // Relayout against Streamlit's wrapper, which already carries the
        // figure's declared height. A window resize event leaves WebGL
        // scenes stuck, and Plotly.Plots.resize() reads the plot's own
        // collapsed box and locks that in -- it left the 3D view 130px
        // shorter than the space reserved for it.
        const wrap = el.parentElement;
        const h = wrap ? wrap.clientHeight : 0;
        if (h > 0) {
          Plotly.relayout(el, { width: wrap.clientWidth, height: h });
        } else {
          Plotly.Plots.resize(el);
        }
      }
    });
  };

  document.addEventListener(
    'click',
    (e) => { if (e.target.closest('[role="tab"]')) setTimeout(fix, 90); },
    true
  );
  setTimeout(fix, 400);
})();
</script>
""",
    unsafe_allow_javascript=True,
)
