"""Run with: python -m unittest discover -s outputs -p test_loader.py -v

Requires pandas, numpy, and pyarrow. Place beside loader.py when testing.
"""
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal
from eda_tool.loader import open_dataset, load_dataset


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame = pd.DataFrame({'id': range(107), 'value': [float(x) for x in range(107)],
                                   'label': ['a', 'b'] * 53 + ['a']})
        self.csv = self.root / 'data.csv'
        self.parquet = self.root / 'data.parquet'
        self.frame.to_csv(self.csv, index=False)
        self.frame.to_parquet(self.parquet, index=False, row_group_size=19)

    def test_open_is_lazy(self):
        with patch('pandas.read_csv', side_effect=AssertionError('eager read')):
            self.assertTrue(open_dataset(self.csv).supports_streaming)

    def test_csv_and_parquet_batches(self):
        for path in [self.csv, self.parquet]:
            with self.subTest(path=path):
                source = open_dataset(path)
                batches = list(source.iter_batches(batch_size=13, columns=['label', 'id']))
                self.assertTrue(all(len(x) <= 13 for x in batches))
                assert_frame_equal(pd.concat(batches, ignore_index=True), self.frame[['label', 'id']])
                assert_frame_equal(load_dataset(source), self.frame)

    def test_csv_parser_always_bounded(self):
        real_read = pd.read_csv
        def bounded(*args, **kwargs):
            self.assertTrue('nrows' in kwargs or 'chunksize' in kwargs)
            return real_read(*args, **kwargs)
        with patch('pandas.read_csv', side_effect=bounded):
            self.assertEqual(sum(len(b) for b in open_dataset(self.csv).iter_batches(batch_size=9)), 107)

    def test_sample_repeatable_across_batch_sizes(self):
        for path in [self.csv, self.parquet]:
            source = open_dataset(path)
            a = source.sample(n=17, seed=9, batch_size=7)
            b = source.sample(n=17, seed=9, batch_size=23)
            assert_frame_equal(a, b)
            self.assertEqual(a.id.nunique(), 17)
            self.assertTrue(a.id.max() > 17)
            self.assertEqual(len(source.sample(n=200)), 107)

    def test_folder_schema_and_order(self):
        folder = self.root / 'parts'
        folder.mkdir()
        self.frame.iloc[:40].to_csv(folder / 'a.csv', index=False)
        self.frame.iloc[40:][['label', 'id', 'value']].to_csv(folder / 'b.csv', index=False)
        (folder / 'README.md').write_text('Ignored')
        assert_frame_equal(load_dataset(folder), self.frame)
        assert_frame_equal(load_dataset(str(folder / '*.csv')), self.frame)
        pd.DataFrame({'other': [1]}).to_csv(folder / 'c.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'matching'):
            list(open_dataset(folder).iter_batches())

    def test_upload_replay_and_early_close(self):
        for path in [self.csv, self.parquet]:
            stream = io.BytesIO(path.read_bytes())
            stream.seek(3)
            source = open_dataset([SimpleNamespace(filename=path.name, file=stream)])
            self.assertEqual(list(source.schema()), list(self.frame.columns))
            self.assertEqual(stream.tell(), 3)
            iterator = source.iter_batches(batch_size=5)
            self.assertEqual(len(next(iterator)), 5)
            iterator.close()
            self.assertFalse(stream.closed)
            self.assertEqual(stream.tell(), 3)
            assert_frame_equal(load_dataset(source), self.frame)
            self.assertEqual(stream.tell(), 3)

    def test_dtype_override_and_drift(self):
        path = self.root / 'mixed.csv'
        path.write_text('code\n001\n002\nhello\n', encoding='utf-8')
        batches = list(open_dataset(path, dtype={'code':'string'}).iter_batches(batch_size=2))
        self.assertEqual(batches[0].code.iloc[0], '001')
        self.assertTrue(all(str(b.code.dtype).startswith('string') for b in batches))

    def test_empty_and_validation(self):
        empty = self.root / 'empty.csv'
        empty.write_text('id,value,label\n')
        self.assertEqual(list(load_dataset(empty).columns), list(self.frame.columns))
        self.assertEqual(len(open_dataset(empty).sample()), 0)
        self.assertEqual(len(load_dataset([])), 0)
        for kwargs in [{'batch_size':0}, {'columns':['absent']}, {'columns':[]}, {'columns':'id'}]:
            with self.assertRaises((ValueError, TypeError)):
                list(open_dataset(self.csv).iter_batches(**kwargs))

    def test_frames_records_and_fallback(self):
        assert_frame_equal(load_dataset([self.frame.iloc[:30], self.frame.iloc[30:]]), self.frame)
        assert_frame_equal(load_dataset(self.frame.to_dict('records')), self.frame)
        path = self.root / 'records.jsonl'
        self.frame.to_json(path, orient='records', lines=True)
        source = open_dataset(path)
        self.assertFalse(source.supports_streaming)
        self.assertEqual(sum(len(b) for b in source.iter_batches(batch_size=9)), 107)
        path = self.root / 'data.pkl'
        self.frame.to_pickle(path)
        with self.assertRaisesRegex(ValueError, 'Pickle'):
            open_dataset(path)
        assert_frame_equal(load_dataset(path, allow_pickle=True), self.frame)


if __name__ == '__main__':
    unittest.main()

