import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from eda_tool.profiler import Profiler, ProfileConfig
from eda_tool.profiler.aggregations import Moments
from eda_tool.profiler.correlations import cramers_v, correlation_ratio


def batches(frame, size):
    for start in range(0, len(frame), size):
        yield frame.iloc[start:start+size]


class NumericTests(unittest.TestCase):
    def test_moments_match_pandas_across_batches(self):
        values = np.random.default_rng(12).gamma(2, 3, 537)
        frame = pd.DataFrame({'x': values})
        expected = frame.x
        for size in [1, 7, 100, 1000]:
            result = Profiler(batches(frame, size)).run()['columns'].data['x']
            for key, value in {'mean': expected.mean(), 'variance': expected.var(),
                               'std': expected.std(), 'skew': expected.skew(),
                               'kurtosis': expected.kurt(), 'sum': expected.sum()}.items():
                self.assertAlmostEqual(result[key], value, places=9, msg=f'{size}: {key}')
            self.assertAlmostEqual(result['median'], expected.median())
            self.assertAlmostEqual(result['MAD'], (expected-expected.median()).abs().median())
            self.assertEqual(sum(result['histogram']['counts']), len(frame))

    def test_moment_merge_and_large_offset(self):
        x = 1e9 + np.arange(1000)/10
        a, b = Moments(), Moments()
        a.update(x[:430]); b.update(x[430:]); a.merge(b)
        self.assertAlmostEqual(a.result()['variance'], np.var(x, ddof=1), places=5)

    def test_nonfinite_and_degenerate(self):
        frame = pd.DataFrame({'x':[1., np.nan, np.inf, -np.inf, -2., 0.], 'constant':[4.]*6,
                              'empty':[np.nan]*6})
        result = Profiler(batches(frame, 2), ProfileConfig(roles={'x':'numeric', 'constant':'numeric'})).run()
        x = result['columns'].data['x']
        self.assertEqual((x['count'], x['num_missing'], x['num_infinity']), (3, 1, 2))
        self.assertAlmostEqual(x['mean'], -1/3)
        self.assertEqual((x['num_zeros'], x['num_neg']), (1, 1))
        self.assertIsNone(result['columns'].data['constant']['skew'])
        self.assertEqual(result['columns'].data['constant']['variance'], 0)
        self.assertIsNone(result['columns'].data['empty']['mean'])
        self.assertTrue(result['columns'].data['empty']['all_missing'])

    def test_pearson_pairwise_complete(self):
        frame = pd.DataFrame({'x':[1., 2., np.nan, 4., 8., np.inf],
                              'y':[3., np.nan, 5., 8., 7., 9.]})
        expected = frame.replace([np.inf, -np.inf], np.nan).corr().loc['x', 'y']
        result = Profiler(batches(frame, 2), ProfileConfig(roles={'x':'numeric', 'y':'numeric'})).run()['correlations']
        self.assertAlmostEqual(result.data.loc['x', 'y'], expected)
        self.assertEqual(result.metadata['pairs'][0]['rows_examined'], 3)

    def test_perfect_binary_numeric(self):
        frame = pd.DataFrame({'x':[0., 0., 1., 1.], 'y':[0., 0., 1., 1.]})
        self.assertAlmostEqual(Profiler(batches(frame, 1)).run()['correlations'].data.loc['x', 'y'], 1.)


class SamplingTests(unittest.TestCase):
    def test_sample_repeatability_and_bounded_state(self):
        frame = pd.DataFrame({'x': range(1000), 'y': np.arange(1000)*2})
        config = ProfileConfig(sample_size=31, max_unique=10)
        a = Profiler(batches(frame, 17), config).run()
        b = Profiler(batches(frame, 64), config).run()
        assert_frame_equal(a['interactions'].data['sample'], b['interactions'].data['sample'])
        self.assertEqual(len(a['interactions'].data['sample']), 31)
        self.assertEqual(a['columns'].data['x']['count'], 1000)
        self.assertIsNone(a['columns'].data['x']['num_unique'])
        self.assertEqual(a['columns'].data['x']['unique_lower_bound'], 11)
        self.assertEqual(a['columns'].data['x']['methods']['quantiles']['method'], 'sampled')
        self.assertIsNone(a['summary'].data['num_duplicates'])

    def test_pairs_capped_and_no_diagonal(self):
        frame = pd.DataFrame(np.arange(200).reshape(10, 20), columns=[f'x{i}' for i in range(20)])
        result = Profiler(frame, ProfileConfig(max_pairs=3, categorical_threshold=0)).run()
        pairs = result['interactions'].data['pairs']
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(a != b for a, b in pairs))
        self.assertEqual(result['correlations'].metadata['omitted_pairs'], 187)
        self.assertLessEqual(result['correlations'].data.shape[0], 6)

    def test_disabled_sections(self):
        r = Profiler(pd.DataFrame({'x':[1,2], 'y':[2,3]}),
                     ProfileConfig(correlations=False, interactions=False)).run()
        self.assertTrue(r['correlations'].data.empty)
        self.assertIsNone(r['interactions'].data['sample'])


class MixedAndEdgeTests(unittest.TestCase):
    def test_categorical_associations_known_results(self):
        x = ['a']*30 + ['b']*30
        y = ['c']*10 + ['d']*20 + ['c']*20 + ['d']*10
        self.assertAlmostEqual(cramers_v(x, y), 1/3)
        self.assertAlmostEqual(correlation_ratio(['a','a','b','b'], [0.,0.,1.,1.]), 1.)
        self.assertIsNone(cramers_v(['a','a'], ['b','c']))
        frame = pd.DataFrame({'x':x, 'y':y})
        r = Profiler(batches(frame, 7), ProfileConfig(roles={'x':'category', 'y':'category'})).run()
        self.assertAlmostEqual(r['correlations'].data.loc['x', 'y'], 1/3)

    def test_text_datetime_nullable(self):
        frame = pd.DataFrame({'text':['hello world', None, 'hello'],
                              'date':pd.date_range('2024-01-01', periods=3, tz='UTC'),
                              'n':pd.Series([1, None, 3], dtype='Int64'),
                              'b':pd.Series([True, None, False], dtype='boolean')})
        before = frame.copy(deep=True)
        r = Profiler(batches(frame, 1), ProfileConfig(roles={'text':'text', 'n':'numeric'})).run()['columns'].data
        self.assertEqual(r['text']['top words'], {'hello':2, 'world':1})
        self.assertEqual(r['text']['length_stats']['mean'], 8)
        self.assertEqual(r['date']['min'], frame.date.min())
        self.assertEqual(r['date']['type'], 'datetime')
        self.assertEqual(r['n']['mean'], 2)
        self.assertEqual(r['b']['type'], 'category')
        assert_frame_equal(before, frame)

    def test_empty(self):
        for frame in [pd.DataFrame(), pd.DataFrame({'x':pd.Series(dtype=float)})]:
            r = Profiler(frame).run()
            self.assertEqual(r['summary'].data['rows'], 0)
            self.assertEqual(r['summary'].data['num_duplicates'], 0)

    def test_duplicates_cross_batch_and_cap(self):
        frame = pd.DataFrame({'x':[1., np.nan, 1., np.nan, 2., 2.]})
        r = Profiler(batches(frame, 2), ProfileConfig(sample_size=2, duplicate_limit=10)).run()
        self.assertEqual(r['summary'].data['num_duplicates'], 3)
        r = Profiler(batches(frame, 2), ProfileConfig(sample_size=2, duplicate_limit=1)).run()
        self.assertIsNone(r['summary'].data['num_duplicates'])

    def test_bad_schema_and_types(self):
        with self.assertRaises(ValueError):
            Profiler(iter([pd.DataFrame({'a':[1]}), pd.DataFrame({'b':[2]})])).run()
        with self.assertRaises(TypeError):
            Profiler(iter([pd.DataFrame({'a':[1]}), pd.DataFrame({'a':['bad']})])).run()
        with self.assertRaises(TypeError):
            Profiler(pd.DataFrame({'a':[1+2j]})).run()
        with self.assertRaises(ValueError):
            ProfileConfig(sample_size=0)

    def test_repeated_runs_have_fresh_state(self):
        profiler = Profiler(pd.DataFrame({'x':[1,2,3]}))
        self.assertEqual(profiler.run()['summary'].data['rows'], 3)
        self.assertEqual(profiler.run()['summary'].data['rows'], 3)


if __name__ == '__main__':
    unittest.main()
