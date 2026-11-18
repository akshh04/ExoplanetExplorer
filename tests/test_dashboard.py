"""Dashboard interactions against explicitly synthetic archive data; no network needed."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        cache = TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        cache_env = patch.dict("os.environ", {"EXOPLANET_ARCHIVE_CACHE": str(Path(cache.name) / "archive.json.gz")})
        cache_env.start()
        self.addCleanup(cache_env.stop)
        response = requests.Response()
        response.status_code = 200
        response.encoding = 'utf-8'
        response._content = (ROOT / 'tests/fixtures/archive.csv').read_bytes()
        response._content_consumed = True
        self.network = patch('requests.get', return_value=response)
        self.get = self.network.start()
        self.addCleanup(self.network.stop)
        self.app = AppTest.from_file(str(ROOT / 'app.py'))

    def run_app(self):
        self.app.run(timeout=30)
        self.assertEqual([e.message for e in self.app.exception], [])
        self.assertEqual(len(self.app.error), 0)

    def click(self, label):
        next(b for b in self.app.button if b.label == label).click()
        self.run_app()

    def planet_count(self):
        return next(item.proto.body.split(">")[1].split("<")[0]
                    for item in self.app.get("html") if 'data-testid="catalogue-count"' in item.proto.body)

    def test_network_failure_has_retry_and_recovers(self):
        self.get.side_effect = requests.ConnectionError("Test archive unavailable")
        self.app.run(timeout=30)
        self.assertFalse(self.app.exception)
        self.assertTrue(any('Could not load data' in e.value for e in self.app.error))
        self.assertTrue(any(b.label == 'Try again' for b in self.app.button))
        self.get.side_effect = None
        self.click('Try again')
        self.assertEqual(self.planet_count(), "24 planets")

    def test_snapshot_after_failed_refresh(self):
        self.run_app()
        self.get.side_effect = requests.exceptions.ProxyError("Test archive unavailable")
        self.click("Refresh data from NASA")
        self.assertTrue(any("Showing a saved NASA snapshot" in item.value for item in self.app.warning))
        self.assertEqual(self.planet_count(), "24 planets")
        requests_made = self.get.call_count
        self.app.radio(key="nav_group").set_value("Learn")
        self.run_app()
        self.assertEqual(self.get.call_count, requests_made)

    def test_saved_comparison_and_mapped_measurements(self):
        self.run_app()
        choice = next(b.label for b in self.app.button if b.label.startswith("Test World"))
        self.click(choice)
        self.click("Add to comparison")
        self.assertEqual(len(self.app.session_state["comparison_tray"]), 1)
        self.app.checkbox(key="complete_only").check()
        self.run_app()
        self.assertEqual(self.app.session_state["route_params"]["complete"], "1")
        counts = [item.value.replace("**", "") for item in self.app.markdown
                  if "of 24 planets" in item.value]
        self.assertTrue(counts)
        self.assertLess(int(counts[0].split()[0]), 24)

    def test_all_sections_and_lazy_charts(self):
        self.run_app()
        self.assertEqual(self.planet_count(), "24 planets")
        self.assertEqual(len(self.app.get('plotly_chart')), 0)
        for view in ['Trends', 'Orbits', 'Classes', '3D', 'Habitable', 'Profile', 'Learn']:
            with self.subTest(view=view):
                self.app.session_state['active_view'] = view
                self.run_app()
                self.assertEqual(self.app.session_state['route_params']['view'], view)
                if view not in ['Learn']:
                    self.assertGreater(len(self.app.get('plotly_chart')), 0)
        self.app.session_state['active_view'] = 'Profile'
        self.run_app()
        self.app.selectbox(key='profile_rival').select('Test World 8 b')
        self.run_app()
        self.assertIn('Test World 8 b', self.app.session_state['route_params']['compare'])
        self.app.toggle(key='dark_mode').set_value(True)
        self.run_app()

    def test_search_bookmark_comparison_and_navigation_retention(self):
        self.run_app()
        self.click('Nearby worlds')
        self.assertEqual(self.app.session_state['dist_limit'], 50.0)
        self.app.selectbox(key='global_search').select('Test World 20 b')
        self.run_app()
        self.assertEqual(self.app.session_state['active_view'], 'Profile')
        self.assertEqual(self.app.selectbox(key='profile_pick').value, 'Test World 20 b')
        self.assertIn("24 of 24 planets", " ".join(item.value.replace("**", "") for item in self.app.markdown))
        self.click('Save planet')
        self.app.selectbox(key='profile_rival').select('Test World 8 b')
        self.run_app()
        self.app.session_state['active_view'] = 'Explore'
        self.run_app()
        self.app.radio(key='nav_group').set_value('Saved')
        self.run_app()
        self.assertTrue(any(b.label == "Test World 20 b" for b in self.app.button))
        self.assertEqual(len([b for b in self.app.button if b.label.startswith("Test World")]), 1)
        self.app.session_state['active_view'] = 'Profile'
        self.run_app()
        self.assertEqual(self.app.selectbox(key='profile_pick').value, 'Test World 20 b')
        self.assertEqual(self.app.selectbox(key='profile_rival').value, 'Test World 8 b')
        self.click('Remove saved planet')
        self.assertEqual(self.app.session_state['favorites'], [])

    def test_malformed_url_and_deep_link(self):
        self.app.query_params.update({'years': '9999-1000', 'esi': 'nan', 'distance': 'inf',
                                      'view': 'broken', 'methods': 'fake'})
        self.run_app()
        self.assertEqual(self.app.session_state['active_view'], 'Explore')
        self.assertEqual(self.app.slider(key='year_range').value, (2000, 2023))
        self.assertEqual(self.app.slider(key='esi_floor').value, 0.0)
        linked = AppTest.from_file(str(ROOT/'app.py'))
        linked.query_params.update({'view': 'Profile', 'planet': 'Test World 20 b', 'compare': 'Test World 8 b'})
        linked.run(timeout=30)
        self.assertFalse(linked.exception)
        self.assertEqual(linked.selectbox(key='profile_pick').value, 'Test World 20 b')
        self.assertEqual(linked.selectbox(key='profile_rival').value, 'Test World 8 b')

    def test_empty_filter_recovery_and_missing_data(self):
        self.run_app()
        self.app.multiselect(key='hz_filter').set_value(['Unknown'])
        self.run_app()
        self.assertTrue(self.app.warning)
        self.click('Relax filters')
        self.assertEqual(self.planet_count(), "24 planets")
        self.app.selectbox(key='global_search').select('Test World 23 b')
        self.run_app()
        self.assertTrue(any('No published radius' in e.value for e in self.app.info))
        self.assertEqual(self.app.selectbox(key='profile_pick').value, 'Test World 23 b')

    def test_catalogue_host_search_keeps_context_after_opening_a_profile(self):
        self.run_app()
        self.click("Next")
        self.app.text_input(key="catalogue_search").set_value("test star 2")
        self.run_app()
        cards = {button.label for button in self.app.button if button.label.startswith("Test World")}
        self.assertEqual(cards, {"Test World 6 b", "Test World 7 b", "Test World 8 b"})
        self.click("Test World 8 b")
        self.assertEqual(self.app.selectbox(key="profile_pick").value, "Test World 8 b")
        self.click("Back to catalogue")
        self.assertEqual(self.app.text_input(key="catalogue_search").value, "test star 2")
        self.assertTrue(any(button.label == "Test World 8 b" for button in self.app.button))
        self.app.text_input(key="catalogue_search").set_value("No matching system")
        self.run_app()
        self.assertTrue(any("No planets match this search" in item.value for item in self.app.info))
        self.click("Clear search")
        self.assertEqual(self.app.text_input(key="catalogue_search").value, "")

    def test_home_overview_tracks_filters_without_using_catalogue_search(self):
        self.run_app()
        metrics = {metric.label: metric.value for metric in self.app.metric}
        self.assertEqual(
            metrics,
            {
                "Planets shown": "24",
                "Planetary systems": "8",
                "In the habitable zone": "6",
                "Best Earth Similarity": "0.973",
                "Discovered in 2023": "1",
            },
        )
        self.app.text_input(key="catalogue_search").set_value("Test Star 0")
        self.run_app()
        self.assertEqual(self.planet_count(), "3 planets")
        self.assertEqual(self.app.metric[0].value, "24")
        self.click("Nearby worlds")
        self.assertEqual(self.app.metric[0].value, "6")
        self.assertEqual(self.app.metric[1].value, "2")
        self.click("Clear all filters")
        self.assertEqual(self.app.metric[0].value, "24")

    def test_comparison_permalink_keeps_order_and_ignores_invalid_entries(self):
        planets = [f"Test World {index} b" for index in range(6, 10)]
        self.app.query_params.update({
            "view": "Profile", "planet": planets[0], "compare": planets[1],
            "tray": json.dumps([planets[0], planets[0], {}, "Unknown planet", *planets[1:], "Test World 10 b"]),
        })
        self.run_app()
        self.assertEqual(json.loads(self.app.session_state["route_params"]["tray"]), planets)
        next(radio for radio in self.app.radio if radio.label == "Comparison layout").set_value("All planets")
        self.run_app()
        comparison = next(table.value for table in self.app.dataframe if "Measure" in table.value.columns)
        self.assertEqual(comparison.columns.tolist(), ["Measure", "Earth", *planets])
        self.assertIn("Habitable zone", comparison["Measure"].tolist())
        self.click(f"Remove {planets[3]} from comparison")
        self.assertEqual(json.loads(self.app.session_state["route_params"]["tray"]), planets[:3])


if __name__ == '__main__':
    unittest.main()
