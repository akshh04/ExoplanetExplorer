# Exoplanet Explorer 🪐

An interactive dashboard over the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
It pulls the live `pscomppars` (Planetary Systems Composite Parameters) table and visualises how
exoplanets are discovered, how diverse they are, and which of them could plausibly host liquid water.

## 🚀 Features

- **Live data ingestion** — queries the archive's TAP (Table Access Protocol) service and caches the
  result for 24 hours, with a manual refresh button. The first load streams the response and shows a
  progress bar with the real number of kilobytes received.
- **Global filters** — discovery year, method, distance, habitable-zone class and a minimum Earth
  Similarity Index, all applied to every chart and table at once, with one-click reset.
- **Discovery trends** — per-year, cumulative, or share-of-year views broken down by discovery method.
- **Size vs. orbit** — a log–log scatter of every world, with Mercury, Earth, Neptune and Jupiter
  overlaid for scale.
- **Unsupervised classification** — K-Means over log mass–radius space, with 2–6 groups. Clusters are
  renumbered by median mass and named from their own statistics, so the labels stay correct as the
  archive grows.
- **3D explorer** — period, radius and equilibrium temperature in one rotatable view, coloured by
  discovery method or temperature.
- **Habitable worlds** — shortlist candidates either by the computed habitable zone or by a plain
  equilibrium-temperature band, with an Earth-Similarity leaderboard and CSV export.
- **Planet profile** — pick any world for its measurements, a size comparison against Earth, the
  other planets in its system, and a diagram placing its orbit against its own star's habitable zone.
- **Head-to-head comparison** — set *Compare with* on the profile tab to put a second world beside
  the first, with Earth as a reference column across radius, mass, orbit, temperature, insolation,
  ESI and habitable-zone class.

Charts share a single colour scheme, so a given discovery method is the same colour everywhere. The
tab bar sticks to the top while scrolling, and the chosen planet survives filter changes that do not
remove it.

## 📱 Responsive layout

The layout is driven entirely by CSS media queries rather than user-agent sniffing, so it adapts live
as a window is resized and treats a narrow desktop window the same as a phone.

| Width | Headline metrics | Navigation | Sidebar | Tooltips |
| --- | --- | --- | --- | --- |
| ≤ 640px | 2 per row, compact | 3 × 2 grid of pills | collapsed | hidden |
| 641–950px | 3 + 2 | one row | collapsed by Streamlit | shown |
| > 950px | 5 across | one row, sticky | open | shown |

On a 375px screen this cut the distance from the top of the page to the navigation from roughly
1070px to 674px, and the planet profile's eight measurements from about 990px of scrolling to 370px.
The sidebar uses `initial_sidebar_state="auto"` so 300px of filters no longer cover a phone screen
on load.

**Navigation.** Six tabs ran to 704px inside a 343px strip, so the last two — one of them the planet
profile — sat off-screen with nothing to signal they existed. On phones the tab strip becomes a
3 × 2 grid of equal 44px chips, and the labels are shortened (*Trends*, *Classes*, *3D view*,
*Habitable*, *Profile*) so each fits.

The chips are filled rather than outlined, and deliberately so: with a border on each chip, row one's
bottom edges sat 6px above row two's top edges, drawing a pair of parallel lines straight across the
nav that read as a double border. Fill alone separates them, with the active chip in solid accent
(6.9:1 contrast in dark, 5.7:1 in light). Streamlit also paints its own underline rail as a 2px
`::after` on the tab list, which no border property can remove — that is hidden on mobile too, and the
list's `overflow-x: auto` is reset to `visible` so the second row is not clipped.

Because the responsive rules have to beat the dark-mode overrides, the whole `@media` section is
emitted *after* them in the stylesheet; otherwise the dark rules repaint the chips and the inactive
labels drop to 1.5:1 against the page.

**Theme toggle.** Left to itself Streamlit stacked the header columns and the switch took a full
348px band between the heading and the standfirst; it now shares the title's row, right-aligned.

**Tooltips.** The `help` "?" icons are hover-only affordances, so they are hidden below 640px rather
than sitting there un-hoverable. Plotly's own hover panels stay, since tapping a point is the only
way to identify it.

**Chart legends** sit horizontally *above* the plot rather than in a right-hand column, which at
348px was taking roughly a third of the chart's width. Above rather than below because underneath,
Plotly's automargin packs the key hard against the x-axis title with a fixed ~4px gap that no margin
or standoff setting will widen; above it, automargin reserves the room and the clearance lands at
15–22px. A proportional `entrywidth` lays the keys out in even columns instead of letting them flow
ragged — 11 methods previously wrapped into seven rows of one to three items. Long method names carry
a legend-only alias (*Transit Timing*, *Orbital Brightness*) so entries cannot overrun their column;
filters, tables and hover text keep the archive's full wording. The 3D view needs its own handling:
a gl3d scene ignores automargin, so its legend is pinned to the container edge and the scene's domain
is shortened to make room.

**Alignment.** Streamlit stacks columns on mobile by forcing `min-width:100%`, and its metric cards
size to their content — so a card carrying a delta came out 107px beside an 80px neighbour. Metric
columns are pinned to half-width and stretched to their row, giving an even grid at every width. The
theme toggle needed a separate fix: `vertical_alignment="center"` centres the *column boxes*, but
Streamlit collapses the title's column to a few pixels and its toggle label pins the switch with
`align-items:flex-start`, which together left the switch 26px above the title's letters on desktop.
Centring the column content and the toggle label brings both to within 3px.

## 🔬 The derived science

Two columns are computed rather than read from the archive.

**Habitable zone** uses the [Kopparapu et al. (2014)](https://arxiv.org/abs/1404.5292) limits. The
effective stellar flux at each boundary is a quartic in `Teff - 5780 K`; combined with the star's
luminosity it gives the boundary distances, which are compared against the planet's semi-major axis.

- *Conservative* — between the runaway-greenhouse and maximum-greenhouse limits.
- *Optimistic* — the wider recent-Venus to early-Mars band.

The published fit covers 2600–7200 K. The app extends the floor to 2500 K so late M-dwarfs such as
TRAPPIST-1 (2566 K) are classified rather than dropped — a mild extrapolation. Spot-checked against
the literature: TRAPPIST-1 e, f and g, Proxima Cen b, TOI-700 d, Kepler-186 f, Kepler-442 b and
Kepler-62 f all land in the conservative zone, while Kepler-452 b and K2-18 b come out optimistic.

**Earth Similarity Index** is the two-parameter form from
[Schulze-Makuch et al. (2011)](https://doi.org/10.1089/ast.2010.0592), combining radius and
equilibrium temperature. Note that exoplanet catalogues publish *equilibrium* temperature, so the
reference point is Earth's 255 K equilibrium temperature rather than its 288 K surface temperature.

Both are approximations: equilibrium temperature ignores atmosphere, albedo and greenhouse effects,
and the habitable-zone limits assume an Earth-like atmosphere around a single star. Treat the output
as a shortlist for follow-up, not a list of habitable planets.

## ⏳ How the loading indicator works

The archive answers with `Transfer-Encoding: chunked` and no `Content-Length`, and it spends several
seconds running the query before the first byte arrives — so there is no true percentage available.
The bar is paced against a typical payload size while the caption reports the real byte count, and
the opening stage covers the server-side wait so it never looks frozen.

`HAB_VERSION` is passed to the cached loader **positionally**, not left as a default: Streamlit builds
its cache key only from the arguments actually supplied, so a default value never reaches the key and
the versioning silently does nothing.

The fetch runs on a worker thread and the script thread paints the bar. That split is required, not
stylistic: Streamlit replays whatever elements a `@st.cache_data` function draws, so a progress
callback that painted into a container from an earlier run raises `CacheReplayClosureError` on the
second rerun. The cached function therefore writes only plain numbers into a shared dict. On a cache
hit the worker finishes before the first poll, so no loading UI is created and the page does not
flicker on ordinary reruns.

## 🛠️ Installation & usage

1. **Install the dependencies** (Python 3.9+):

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard**:

   ```bash
   streamlit run app.py
   ```

3. Open <http://localhost:8501>.

Use the **🌙 Dark** toggle at the top right to switch themes; `.streamlit/config.toml` sets the
default.

## 📚 Technologies used

- **Python 3** · **Streamlit** (web framework) · **Pandas** / **NumPy** (data)
- **Plotly Express** (interactive charting) · **scikit-learn** (K-Means clustering)
- **NASA Exoplanet Archive TAP API** (data source)

## 📝 License

Open-source, for educational and portfolio purposes. Data is provided by the NASA Exoplanet Archive,
operated by the California Institute of Technology under contract with the National Aeronautics and
Space Administration.
