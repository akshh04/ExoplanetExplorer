# Exoplanet Explorer 🪐

Explore confirmed planets beyond our solar system through interactive charts, planet profiles,
and comparisons with Earth. Exoplanet Explorer is a Python/Streamlit dashboard powered by live
data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).

**[Open the live dashboard → https://exoexplorer.streamlit.app/](https://exoexplorer.streamlit.app/)**

The app reads NASA's `pscomppars` (Planetary Systems Composite Parameters) table and caches it
for 24 hours. No account or API key is required. Habitable-zone estimates and the Earth Similarity
Index help identify candidates for further study; they do not establish that a planet supports life.

## Quick tour

1. Search the full catalogue by planet name, or open **Explore** and pick a collection.
2. Open **Filters** beside the result count. The drawer applies changes together. On a laptop,
   the sidebar also offers live filters, named filter sets, unit preferences and CSV export.
3. Browse planets as cards or as a sortable table. Select a planet to preview it, open its
   profile or add it to the comparison tray.
4. Open **Charts** for discovery trends, size and orbit, statistical clusters, the 3D and 2D
   explorers, habitable-zone candidates, and a sky map. **Learn** explains the science and units.
5. Open **Saved** to organize bookmarked planets into collections. The browser remembers your
   theme, units, catalogue layout and saved items. URLs record the current view, filters and the
   primary profile comparison.

Use **/** to focus search and **F** to open filters when focus is outside a form. On a phone,
catalogue cards stack and comparisons start with one selected planet alongside Earth.

## Features

- **Catalogue:** curated collections, pagination, a pinned planet-name column in table mode,
  configurable columns, and a side-by-side preview on wide screens.
- **Profiles:** measurement coverage, uncertainty ranges and sources when NASA supplies them,
  a host system diagram and planets with similar published measurements.
- **Comparison:** compare up to four selected planets with Earth, switch to one planet at a time,
  and export a summary image with NASA attribution.
- **Discovery:** mission and facility timelines, first archive release dates, an accessible
  data table alongside charts, and a sky map of reported coordinates.
- **Units:** select light-years or parsecs and Earth or Jupiter units for catalogue and profile values.
  Research charts keep their labeled source units.
- **Data availability:** the dashboard shows the latest successful fetch time. If a refresh
  fails, a last-known-good local snapshot remains browsable and is clearly marked.

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

## Data loading and offline use

The initial request reports connection, received bytes and data preparation. NASA does not
provide a total download size. Successful data is cached in memory for up to 24 hours and
stored as a compressed, validated snapshot at `.cache/archive.json.gz`. Set
`EXOPLANET_ARCHIVE_CACHE` to use another local path. The snapshot contains public
NASA data and is excluded from Git.

Only the selected page renders. Large 3D and 2D views sample up to 2,500 eligible planets.
The catalogue initially shows 12 cards, and chart data tables load when expanded.

## 🛠️ Installation & usage

1. **Clone the repository and install dependencies** (Python 3.12):

   ```bash
   git clone https://github.com/akshh04/ExoplanetExplorer.git
   cd ExoplanetExplorer
   python -m venv .venv
   source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. **Run the dashboard**:

   ```bash
   python -m streamlit run app.py
   ```

3. Open <http://localhost:8501>.

Use the **🌙 Dark** toggle at the top right to switch themes; `.streamlit/config.toml` sets the
default.

An internet connection to `exoplanetarchive.ipac.caltech.edu` is required. The first load may take
longer while NASA runs the query. If data loading fails, check connectivity and use **Try again**.

## Local regression checks

After installing dependencies, run:

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic NASA data for repeatable checks, including fallback after a failed
refresh. To run the responsive browser tests locally:

```bash
python -m pip install playwright==1.65.2
python -m playwright install chromium
python tests/browser_dashboard.py
```

GitHub Actions runs both suites on pull requests and pushes to `main`.

See [the UI roadmap](docs/UI_ROADMAP.md) for delivered changes and further mobile/laptop ideas.

## Hosting and manual visits

The live app is hosted on **Streamlit Community Cloud** at <https://exoexplorer.streamlit.app/>.
Community Cloud can put inactive apps to sleep. If that happens, open the site and use its wake-up
button.

All automatic dashboard visits have been removed. To make a one-off cloud browser visit, use
**Actions → Manual dashboard visit → Run workflow**. The workflow checks the dashboard's planet
count and attempts to wake the app if needed; it has no scheduled triggers.

For the repository's GitHub **About** settings, use this description and website:

> Interactive exoplanet dashboard with live NASA data, discovery trends, 3D charts, planet comparisons, and habitability estimates.

Website: <https://exoexplorer.streamlit.app/>

## 📚 Technologies used

- **Python 3** · **Streamlit** (web framework) · **Pandas** / **NumPy** (data)
- **Plotly Express** (interactive charting) · **scikit-learn** (K-Means clustering)
- **NASA Exoplanet Archive TAP API** (data source)

## 📝 License

Open-source, for educational and portfolio purposes. Data is provided by the NASA Exoplanet Archive,
operated by the California Institute of Technology under contract with the National Aeronautics and
Space Administration.
