import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from data import DatasetWorkspace
from models import ModelConfig
from tasks import TaskConfig
from preprocessing import Recipe, Step
from workflows import prepare_task
from workflows.prediction import predict_file
from training import train_models, evaluate_test
from training.features import fit_encoder, encode, normalize_features
from prediction import Predictor
from exporting import export_bundle, export_native, export_onnx
from etml.cli import main


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = DatasetWorkspace(self.root/'datasets')

    def prepare(self, regression=False, recipe=None):
        self.frame = pd.DataFrame({'x': np.arange(100, dtype=float), 'group': ['a','b']*50,
                                   'target': np.arange(100)*2. if regression else ['no']*50+['yes']*50})
        source = self.root/'input.csv'
        self.frame.to_csv(source,index=False)
        self.ws.import_files(source,dataset_id='demo')
        self.preparation = prepare_task(self.ws,'demo',TaskConfig('t','target',
            'regression' if regression else 'classification'),recipe=recipe)
        return self.preparation

    def train(self, regression=False, recipe=None, configs=None):
        prep = self.prepare(regression,recipe)
        return train_models(self.ws,'demo','t',prep.run_id,
                            configs=configs or [ModelConfig(algorithm='linear')])

    def test_dummy_is_only_trained_when_requested(self):
        result = self.train(configs=[ModelConfig(algorithm='dummy')])
        self.assertEqual(len(result.leaderboard),1)
        self.assertTrue(result.leaderboard[0]['is_baseline'])

    def test_classification_roundtrip_export_and_fresh_process(self):
        result = self.train()
        self.assertEqual(len(result.leaderboard),1)
        predictor = Predictor.load(result.bundle)
        self.assertNotIn('target',predictor.schema['raw_columns'])
        expected = predictor.predict(self.frame)
        self.assertEqual(set(expected.prediction),{'no','yes'})
        exported = export_bundle(result.bundle,self.root/'export')
        pd.testing.assert_frame_equal(expected,Predictor.load(exported).predict(self.frame))
        script = 'from prediction import Predictor; import pandas as pd,sys; print(len(Predictor.load(sys.argv[1]).predict(pd.read_csv(sys.argv[2]))))'
        completed = subprocess.run([sys.executable,'-c',script,str(exported),str(self.root/'input.csv')],
                                   capture_output=True,text=True,check=True)
        self.assertEqual(completed.stdout.strip(),'100')
        native = export_native(result.bundle,self.root/'native')
        self.assertFalse(json.loads((native/'manifest.json').read_text())['metadata']['preprocessing_included'])
        with self.assertRaises(FileExistsError):
            export_bundle(result.bundle,exported)

    def test_regression_and_explicit_test_evaluation(self):
        result = self.train(regression=True)
        selection = json.loads((result.directory/'selection.json').read_text())
        self.assertFalse(selection['test_evaluated'])
        report = evaluate_test(self.ws,'demo','t',result.run_id)
        self.assertEqual(report['candidate'],result.winner)
        self.assertLess(report['metrics']['rmse'],2.)
        Predictor.load(result.bundle)

    def test_training_never_loads_test_for_selection(self):
        prep = self.prepare()
        from training.runner import _load_frame
        paths = []
        def tracked(path,*args):
            paths.append(str(path).replace('\\','/'))
            return _load_frame(path,*args)
        with patch('training.runner._load_frame',side_effect=tracked):
            train_models(self.ws,'demo','t',prep.run_id,configs=[ModelConfig(algorithm='dummy')])
        self.assertEqual(len(paths),2)
        self.assertTrue(all('/test/' not in p for p in paths))

    def test_training_encoder_statistics_and_unknown_categories(self):
        frame = pd.DataFrame({'x':[1.,3.,np.nan], 'cat':[0,1,0]})
        schema = {'x':'numeric','cat':'categorical'}
        encoder = fit_encoder(frame,schema)
        self.assertEqual(encoder.named_transformers_['numeric']['imputer'].statistics_[0],2.)
        test = pd.DataFrame({'x':[np.nan,999999.],'cat':[9.,1.]})
        transformed = encode(encoder,test,schema).toarray()
        self.assertEqual(transformed[0,0],2.)
        self.assertEqual(transformed[0,1:].sum(),0.)
        self.assertEqual(transformed[1,1:].sum(),1.)
        self.assertEqual(encoder.named_transformers_['numeric']['imputer'].statistics_[0],2.)
        with self.assertRaises(ValueError):
            fit_encoder(frame,schema,max_features=1)

    def test_batched_prediction_and_row_exclusion(self):
        recipe = Recipe((Step('drop_missing',('x',)).approve(),))
        result = self.train(recipe=recipe)
        frame = self.frame.copy()
        frame.loc[[2,5,99],'x'] = np.nan
        frame.to_csv(self.root/'new.csv',index=False)
        report = predict_file(result.bundle,self.root/'new.csv',self.root/'predictions.csv',batch_size=7)
        self.assertEqual(report['excluded_rows'],3)
        saved = pd.read_csv(self.root/'predictions.csv')
        expected = Predictor.load(result.bundle).predict(frame)
        self.assertEqual(saved.status.tolist(),expected.status.tolist())
        self.assertEqual(saved.input_row.tolist(),list(range(100)))
        self.assertEqual(saved.prediction.dropna().tolist(),expected.prediction.dropna().tolist())
        with self.assertRaises(FileExistsError):
            predict_file(result.bundle,self.root/'new.csv',self.root/'predictions.csv')
        with self.assertRaises(ValueError):
            Predictor.load(result.bundle).predict(frame.drop(columns='x'))
        with self.assertRaises(ValueError):
            Predictor.load(result.bundle).predict(frame,strict=True)

    def test_tampering_rejected(self):
        result = self.train()
        (result.bundle/'schema.json').write_text('{}')
        with self.assertRaises(ValueError):
            Predictor.load(result.bundle)

    def test_limits_and_failure_publish_nothing(self):
        prep = self.prepare()
        with self.assertRaises(ValueError):
            train_models(self.ws,'demo','t',prep.run_id,max_rows=1)
        with self.assertRaises((ValueError,TypeError)):
            train_models(self.ws,'demo','t',prep.run_id,
                         configs=[ModelConfig(algorithm='linear',params={'invalid_parameter':1})])
        folder = self.ws.get('demo').directory/'tasks/t/training'
        self.assertFalse(folder.exists() and list(folder.iterdir()))

    def test_cli_train_predict_export_test(self):
        prep = self.prepare()
        def call(args):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(['models',*args,'--json','--quiet'])
            self.assertEqual(status,0)
            return json.loads(output.getvalue())
        result = call(['train','demo','t',prep.run_id,'--workspace',str(self.ws.root),
                       '--model','sklearn:linear'])
        call(['predict',str(self.root/'input.csv'),'--bundle',result['bundle'],
              '--output',str(self.root/'cli.csv'),'--batch-size','13'])
        call(['export','--bundle',result['bundle'],'--output',str(self.root/'cli-export')])
        call(['test','demo','t',result['training_run'],'--workspace',str(self.ws.root)])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['models','export','--bundle',result['bundle'],
                '--output',str(self.root/'bad'),'--format','onnx']),2)

    @unittest.skipUnless(importlib.util.find_spec('xgboost'),'optional xgboost extra')
    def test_xgboost_bundle(self):
        result = self.train(configs=[ModelConfig('xgboost','boosted_trees',{'n_estimators':5})])
        bundle = result.directory/'candidates/candidate-000'
        self.assertTrue((bundle/'model.ubj').exists())
        self.assertEqual(len(Predictor.load(bundle).predict(self.frame)),100)

    @unittest.skipUnless(importlib.util.find_spec('onnxruntime') and importlib.util.find_spec('skl2onnx'),
                         'optional onnx extra')
    def test_onnx_parity(self):
        result = self.train()
        bundle = result.directory/'candidates/candidate-000'
        exported = export_onnx(bundle,self.root/'onnx',sample_data=self.frame)
        self.assertTrue(json.loads((exported/'parity.json').read_text())['verified'])
        self.assertFalse(json.loads((exported/'manifest.json').read_text())['metadata']['preprocessing_included'])

    @unittest.skipUnless(importlib.util.find_spec('onnxruntime') and importlib.util.find_spec('skl2onnx'),
                         'optional onnx extra')
    def test_forest_regression_onnx(self):
        result = self.train(regression=True,configs=[ModelConfig(algorithm='random_forest',params={'n_estimators':5})])
        bundle = result.directory/'candidates/candidate-000'
        exported = export_onnx(bundle,self.root/'onnx',sample_data=self.frame)
        self.assertTrue((exported/'model.onnx').exists())


if __name__ == '__main__':
    unittest.main()
