import unittest
from unittest.mock import patch
from training.curves import learning_curves
from etml.web.csv_exports import ranges

class AnalysisExportTests(unittest.TestCase):
    def test_ranges_without_expansion(self):
        self.assertEqual(ranges('1, 3-1000000000'),[(1,1),(3,1000000000)])
        for value in ['0','3-1','x','-1']:
            with self.assertRaises(ValueError):ranges(value)

    def test_loss_history_shapes(self):
        for history in [
            {'history':[{'epoch':1,'train_loss':2},{'epoch':2,'train_loss':1}]},
            [{'epoch':1,'train_loss':2,'validation_loss':3}],
            {'history':{'validation_0':{'logloss':[2,1]},'validation_1':{'logloss':[3,2]}}}]:
            curves=learning_curves(history)
            self.assertEqual(len(curves),1)
            self.assertTrue(curves[0]['image'].startswith('data:image/png;base64,iVBOR'))
        self.assertEqual(learning_curves({'history':[]}),[])
        self.assertEqual(learning_curves([{'epoch':1,'train_accuracy':.8}]),[])
