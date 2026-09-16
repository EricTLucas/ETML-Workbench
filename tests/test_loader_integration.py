import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from eda_tool.loader import open_dataset
from eda_tool.profiler import Profiler, ProfileConfig


class LoaderIntegrationTests(unittest.TestCase):
    def test_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.csv'
            frame = pd.DataFrame({'x':np.arange(101, dtype=float), 'y':np.arange(101)*3})
            frame.to_csv(path, index=False)
            result = Profiler(open_dataset(path), ProfileConfig(batch_size=7, sample_size=10)).run()
            self.assertEqual(result['summary'].data['rows'], 101)
            self.assertAlmostEqual(result['columns'].data['x']['variance'], frame.x.var())
            self.assertAlmostEqual(result['correlations'].data.loc['x','y'], 1.)

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'PyArrow not installed')
    def test_parquet(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.parquet'
            pd.DataFrame({'x':range(127)}).to_parquet(path, index=False, row_group_size=13)
            result = Profiler(open_dataset(path), ProfileConfig(batch_size=5)).run()
            self.assertEqual(result['columns'].data['x']['count'], 127)
            self.assertEqual(result['columns'].data['x']['mean'], 63.)


if __name__ == '__main__':
    unittest.main()
