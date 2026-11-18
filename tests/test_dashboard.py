"""Dashboard interactions against explicitly synthetic archive data; no network needed."""
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

    def test_network_failure_has_retry_and_recovers(self):
        self.get.side_effect = requests.ConnectionError("Test archive unavailable")
        self.app.run(timeout=30)
        self.assertFalse(self.app.exception)
        self.assertTrue(any('Could not load data' in e.value for e in self.app.error))
        self.assertTrue(any(b.label == 'Try again' for b in self.app.button))
        self.get.side_effect = None
        self.click('Try again')
        self.assertEqual(self.app.metric[0].value, '24')

    def test_snapshot_after_failed_refresh(self):
        self.run_app()
        self.get.side_effect = requests.exceptions.ProxyError("Test archive unavailable")
        self.click("Refresh data from NASA")
        self.assertTrue(any("Showing a saved NASA snapshot" in item.value for item in self.app.warning))
        self.assertEqual(self.app.metric[0].value, "24")
        requests_made = self.get.call_count
        self.app.radio(key="nav_group").set_value("Learn")
        self.run_app()
        self.assertEqual(self.get.call_count, requests_made)

    def test_saved_comparison_and_mapped_measurements(self):
        self.run_app()
        choice = next(b.label for b in self.app.button if b.label.startswith("Preview Test"))
        self.click(choice)
        self.click("Add to comparison")
        self.assertEqual(len(self.app.session_state["comparison_tray"]), 1)
        self.app.checkbox(key="complete_only").check()
        self.run_app()
        self.assertEqual(self.app.session_state["route_params"]["complete"], "1")
        self.assertLess(int(self.app.metric[0].value), 24)

    def test_all_sections_and_lazy_charts(self):
        self.run_app()
        self.assertEqual(self.app.metric[0].value, '24')
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
        self.assertEqual(self.app.metric[0].value, '24')
        self.click('Save planet')
        self.app.selectbox(key='profile_rival').select('Test World 8 b')
        self.run_app()
        self.app.session_state['active_view'] = 'Explore'
        self.run_app()
        self.app.radio(key='nav_group').set_value('Saved')
        self.run_app()
        self.assertTrue(any(b.label == 'Preview Test World 20 b' for b in self.app.button))
        self.assertEqual(len([b for b in self.app.button if b.label.startswith('Preview Test')]), 1)
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
        self.assertEqual(self.app.metric[0].value, '24')
        self.app.selectbox(key='global_search').select('Test World 23 b')
        self.run_app()
        self.assertTrue(any('No published radius' in e.value for e in self.app.info))
        self.assertEqual(self.app.selectbox(key='profile_pick').value, 'Test World 23 b')


if __name__ == '__main__':
    unittest.main()
