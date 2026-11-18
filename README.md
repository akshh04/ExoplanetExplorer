# Exoplanet Explorer

Exoplanet Explorer is a Streamlit dashboard for browsing confirmed planets, exploring their
measurements, and comparing them with Earth. It uses the Planetary Systems Composite
Parameters table from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).

[Open the live dashboard](https://exoexplorer.streamlit.app/)

## Using the dashboard

1. In **Explore**, the **Selection overview** shows planets, planetary systems, habitable-zone
   candidates, the best Earth Similarity Index, and discoveries in the latest year in the selection.
   It updates with the global filters. The catalogue search narrows the displayed results.
   On wider screens the five stats share a single strip, and search and catalogue controls
   sit beside the catalogue heading. On tablets and phones the controls sit below the heading.
2. Search by planet name or host star. Select a planet name to open its profile, or use the eye
   button for a quick preview and comparison controls. Use **Filters** to narrow the selection.
   **Options** holds sorting, collections, table view, and search across the entire archive.
   **Discover** has presets and guided searches, including the short-orbit example.
3. Open **Charts** for discovery trends, size and orbit, statistical clusters, 2D and 3D plots,
   habitable-zone candidates, and a sky map. **Learn** explains the measurements and methods.
4. Save planets to browser-local collections, or add up to four planets to compare with Earth.
   The comparison URL preserves the selection. Export the measurements as CSV or the displayed
   comparison as an image.

Press **/** to focus search or **F** to open filters when focus is outside a form. Use the
**Dark** switch next to the title. Expand the sidebar for saved filters, CSV export, data
refresh, and unit preferences. The browser remembers preferences and saved planets between visits.

## Science and interpretation

The dashboard derives habitable-zone classifications and Earth Similarity Index scores from
published measurements.

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
The catalogue shows 12 planets per page with each planet’s host, radius, distance, and
habitable-zone classification. On wider screens Explore shows a three-column grid with
labeled measurements. The grid uses fewer columns as space narrows. Table view lets you
choose columns and open a planet from a selected row.

## Run locally

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

The first load requires access to `exoplanetarchive.ipac.caltech.edu`. If NASA is temporarily
unavailable, select **Try again** or continue with the labeled snapshot when one is available.
GitHub Codespaces can use the included Python 3.12 dev container.

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

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit views, filters, charts, and archive queries |
| `archive_cache.py`, `data_contract.py` | Snapshot storage and required NASA columns |
| `explorer_browser.py`, `browser_state.js` | Browser preferences, keyboard shortcuts, and navigation history |
| `tests/` | Synthetic archive fixture, regression tests, and Chromium checks |
| `scripts/visit_site.py` | Browser check for the hosted catalogue |

See the [interface overview](docs/UI_ROADMAP.md) for the features available across screen sizes.

## Hosting and manual visits

The live app is hosted on **Streamlit Community Cloud** at <https://exoexplorer.streamlit.app/>.
Community Cloud can put inactive apps to sleep. If that happens, open the site and use its wake-up
button.

Use **Actions → Manual dashboard visit → Run workflow** for a one-off browser check of the hosted
catalogue. It checks the visible planet count and can wake the app if needed.

## Data attribution

The NASA Exoplanet Archive is operated by the California Institute of Technology under contract
with the National Aeronautics and Space Administration.
