import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eda_tool.profiler import Profiler, ProfileConfig
from eda_tool.visualizer import Visualizer
from eda_tool.html_report import HtmlReport
from eda_tool.cli import main
from eda_tool.artifacts import load_run


class CategoryTests(unittest.TestCase):
    def test_full_stream_threshold_and_exact_counts_with_small_sample(self):
        frame = pd.DataFrame({'binary': [0, 1]*30, 'label': ['a', 'b', 'c']*20,
                              'late_numeric': [0]*40 + list(range(20)),
                              'late_text': ['a']*40 + [str(i) for i in range(20)]})
        for size in (1, 7, 60):
            batches = (frame.iloc[i:i+size] for i in range(0, len(frame), size))
            r = Profiler(batches, ProfileConfig(sample_size=1, max_unique=1)).run()
            cols = r['columns'].data
            self.assertEqual(cols['binary']['type'], 'category')
            self.assertEqual(cols['binary']['counts'], {0: 30, 1: 30})
            self.assertEqual(cols['label']['counts'], {'a': 20, 'b': 20, 'c': 20})
            self.assertEqual(cols['binary']['methods']['frequencies']['method'], 'exact')
            self.assertEqual(cols['binary']['length_stats']['count'], 60)
            self.assertEqual(cols['late_numeric']['type'], 'numeric')
            self.assertEqual(cols['late_text']['type'], 'text')

    def test_boundary_overrides_and_disable(self):
        frame = pd.DataFrame({'ten': list(range(10))*11, 'eleven': list(range(11))*10,
                              'words': ['yes', 'no']*55})
        cols = Profiler(frame).run()['columns'].data
        self.assertEqual(cols['ten']['type'], 'category')
        self.assertEqual(cols['eleven']['type'], 'numeric')
        config = ProfileConfig(roles={'ten': 'numeric', 'words': 'text', 'eleven': 'category'})
        cols = Profiler(frame, config).run()['columns'].data
        self.assertEqual([cols[c]['type'] for c in frame], ['numeric', 'category', 'text'])
        for threshold in (0, 5):
            cols = Profiler(frame, ProfileConfig(categorical_threshold=threshold)).run()['columns'].data
            self.assertEqual(cols['ten']['type'], 'numeric')
        with self.assertRaises(ValueError):
            ProfileConfig(categorical_threshold=-1)

    def test_missing_constant_and_dates(self):
        frame = pd.DataFrame({'missing': [np.nan]*4, 'constant': [7]*4,
                              'flag': [0, None, 1, None], 'date': pd.date_range('2025-01-01', periods=4)})
        cols = Profiler(frame).run()['columns'].data
        self.assertEqual(cols['missing']['type'], 'numeric')
        self.assertEqual(cols['constant']['type'], 'category')
        self.assertEqual(cols['flag']['num_missing'], 2)
        self.assertEqual(cols['flag']['num_unique'], 2)
        self.assertEqual(cols['date']['type'], 'datetime')

    def test_associations_follow_final_roles_and_pair_limit(self):
        frame = pd.DataFrame({'label': ['a', 'b']*20, 'binary': [0, 1]*20,
                              'x': np.arange(40.), 'y': np.arange(40.)*3})
        r = Profiler(frame, ProfileConfig(batch_size=3)).run()
        pairs = {tuple(p['columns']): p for p in r['correlations'].metadata['pairs']}
        self.assertEqual(pairs['label', 'binary']['method'], 'cramers_v')
        self.assertAlmostEqual(pairs['label', 'binary']['value'], 1)
        self.assertEqual(pairs['binary', 'x']['method'], 'correlation_ratio_eta')
        self.assertEqual(pairs['x', 'y']['method'], 'pearson')
        self.assertAlmostEqual(pairs['x', 'y']['value'], 1)
        limited = Profiler(frame, ProfileConfig(max_pairs=1)).run()['correlations'].metadata
        self.assertEqual(limited['pairs'][0]['columns'], ('label', 'binary'))
        self.assertEqual(limited['omitted_pairs'], 5)

    def test_summary_pies_and_report_branding(self):
        frame = pd.DataFrame({'flag': [0, 1]*20, 'label': ['yes', 'no']*20, 'x': range(40)})
        profile = Profiler(frame).run()
        for source in (profile, frame):
            charts = Visualizer(source).summary()
            try:
                selected = {c.request.get('x'): c.kind for c in charts if c.request.get('x')}
                self.assertEqual(selected['flag'], 'pie')
                self.assertEqual(selected['label'], 'pie')
                self.assertEqual(selected['x'], 'histogram')
                html = HtmlReport(profile, charts).render()
                self.assertIn('ETML WORKBENCH EDA', html)
                self.assertNotIn('Orchid', html)
                self.assertNotIn('figcaption', html)
                self.assertEqual(html.count('<img '), len(charts))
                self.assertTrue(all(c.figure.texts for c in charts))
            finally:
                for c in charts:
                    c.close()

    def test_cli_threshold_and_role_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'data.csv'
            pd.DataFrame({'flag': [0, 1]*10, 'rating': list(range(5))*4}).to_csv(source, index=False)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = main(['eda', 'analyze', str(source), '--output', str(root/'run'),
                               '--categorical-threshold', '5', '--role', 'flag=numeric'])
            self.assertEqual(status, 0)
            saved = load_run(root/'run')
            self.assertEqual(saved.profile['columns'].data['flag']['type'], 'numeric')
            self.assertEqual(saved.profile['columns'].data['rating']['type'], 'category')
            self.assertEqual(saved.manifest['profile_config']['categorical_threshold'], 5)

    def test_pie_uses_population_counts_without_sample(self):
        frame = pd.DataFrame({'flag': [0]*90 + [1]*10})
        profile = Profiler(frame, ProfileConfig(sample_size=1, interactions=False)).run()
        chart = Visualizer(profile).plot('pie', x='flag')
        try:
            self.assertEqual(chart.metadata['status'], 'ok')
            self.assertEqual(chart.metadata['method'], 'exact')
            self.assertEqual(chart.metadata['rows_used'], 100)
            percentages = [t.get_text() for t in chart.axes[0].texts]
            self.assertIn('90%', percentages)
            self.assertIn('10%', percentages)
        finally:
            chart.close()


if __name__ == '__main__':
    unittest.main()
