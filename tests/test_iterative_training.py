from contextlib import redirect_stdout,redirect_stderr
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from data import DatasetWorkspace
from tasks import TaskConfig
from workflows import prepare_task
from models import ModelConfig,create_model
from training import train_models,search_models,search_configs,resume_training,compare_runs
from training.experiments import training_runs,resolve_run
from training.checkpoints import load_checkpoint
from prediction import Predictor
from exporting import export_bundle,export_native
from etml.cli import main


class IterativeTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.ws=DatasetWorkspace(self.root/'datasets with spaces')
        self.frame=pd.DataFrame({'x':np.linspace(-1,1,60),'group':['a','b']*30,'y':['no']*30+['yes']*30})
        self.frame.to_csv(self.root/'raw.csv',index=False)
        self.ws.import_files(self.root/'raw.csv',dataset_id='demo')
        self.prep=prepare_task(self.ws,'demo',TaskConfig('t','y','classification'))

    def cli(self,*args,json_output=True):
        out,err=io.StringIO(),io.StringIO()
        with redirect_stdout(out),redirect_stderr(err):
            code=main([*args,'--workspace',str(self.ws.root),*(['--json'] if json_output else [])])
        return code,json.loads(out.getvalue()) if json_output and out.getvalue() else out.getvalue(),err.getvalue()

    def test_search_reproducible_without_replacement_and_budget(self):
        base=ModelConfig(algorithm='random_forest')
        space={'max_depth':[2,4,6],'n_estimators':[3,5]}
        a=search_configs(base,space,trials=4,seed=8)
        b=search_configs(base,space,trials=4,seed=8)
        self.assertEqual(a,b)
        self.assertEqual(len({json.dumps(c.params,sort_keys=True) for c in a}),4)
        with self.assertRaises(ValueError): search_configs(base,space,method='grid',max_trials=3)
        result=search_models(self.ws,'demo','t',self.prep.run_id,base=base,space=space,trials=2,label='first')
        self.assertEqual(len(result.leaderboard),2)
        metadata=json.loads((result.directory/'selection.json').read_text())
        self.assertEqual(metadata['experiment']['trials'],2)
        self.assertFalse(metadata['test_evaluated'])

    def test_comparison_groups_preparations_and_retains_runs(self):
        configs=[ModelConfig(algorithm='dummy')]
        a=train_models(self.ws,'demo','t',self.prep.run_id,configs=configs,label='one')
        b=train_models(self.ws,'demo','t',self.prep.run_id,configs=configs,label='two')
        compared=compare_runs(self.ws,'demo','t',preparation_run=self.prep.run_id)
        self.assertEqual({r['run_id'] for r in compared['leaderboard']},{a.run_id,b.run_id})
        second=prepare_task(self.ws,'demo',TaskConfig('t','y','classification'))
        train_models(self.ws,'demo','t',second.run_id,configs=configs)
        compared=compare_runs(self.ws,'demo','t',preparation_run=self.prep.run_id)
        self.assertEqual(compared['excluded_runs_from_other_preparations'],1)

    def test_cli_latest_hints_json_and_selected_prediction(self):
        code,result,error=self.cli('models','train','demo','t','--model','sklearn:linear')
        self.assertEqual(code,0,error)
        self.assertIn(result['training_run'],result['next_steps'][1]['command'])
        self.assertIn(str(self.ws.root),result['next_steps'][0]['command'])
        for operation in ('compare','runs','history'):
            code,_,error=self.cli('models',operation,'demo','t')
            self.assertEqual(code,0,error)
        code,_,error=self.cli('models','predict',str(self.root/'raw.csv'),'--dataset','demo','--task','t',
                              '--output',str(self.root/'predictions.csv'))
        self.assertEqual(code,0,error)
        code,_,error=self.cli('models','test','demo','t','--candidate','candidate-000')
        self.assertEqual(code,0,error)
        code,text,error=self.cli('models','train','demo','t','--model','sklearn:dummy',json_output=False)
        self.assertEqual(code,0,error)
        self.assertIn('Suggested next steps:',text)
        self.assertIn('Validation balanced_accuracy',text)
        code,_,error=self.cli('models','train','demo','t','--model','sklearn:linear','--epochs','3')
        self.assertEqual(code,2)

    def test_preparation_hint_contains_created_id(self):
        code,payload,error=self.cli('tasks','prepare','demo','t')
        self.assertEqual(code,0,error)
        self.assertIn(payload['run_id'],payload['next_steps'][0]['command'])

    def test_cli_small_search(self):
        space=self.root/'space.json'
        space.write_text(json.dumps({'n_estimators':[3,5],'max_depth':[2]}))
        code,result,error=self.cli('models','search','demo','t','--space',str(space),'--method','grid')
        self.assertEqual(code,0,error)
        self.assertEqual(len(result['leaderboard']),2)

    @unittest.skipUnless(importlib.util.find_spec('xgboost'),'xgboost extra')
    def test_xgboost_validation_stopping_and_history(self):
        config=ModelConfig('xgboost','boosted_trees',{'n_estimators':50,'early_stopping_rounds':3,'max_depth':2})
        model=create_model(config,'classification')
        x=np.arange(80,dtype=np.float32).reshape(-1,1)
        y=(x[:,0]>40).astype(int)
        model.fit_validation(x,y,validation_data=(x,1-y))
        self.assertLess(model.training_summary['boosting_rounds'],50)
        self.assertIsNotNone(model.training_summary['best_iteration'])
        result=train_models(self.ws,'demo','t',self.prep.run_id,configs=[config])
        from training.history import read_history,plot_history
        payload=read_history(resolve_run(self.ws,'demo','t',result.run_id),'candidate-000')
        self.assertTrue(payload['rows'])
        self.assertIn('validation_logloss',payload['rows'][0])
        self.assertTrue(plot_history(payload,self.root/'curve.png').exists())
        with self.assertRaises(FileExistsError): plot_history(payload,self.root/'curve.png')

    def neural_resume(self,backend):
        config=ModelConfig(backend,'mlp',{'epochs':2,'hidden_sizes':[4],'batch_size':16,'patience':None})
        result=train_models(self.ws,'demo','t',self.prep.run_id,configs=[config])
        checkpoint=result.leaderboard[0]['checkpoint']
        self.assertEqual(load_checkpoint(checkpoint)['config']['backend'],backend)
        with patch('training.runner.fit_encoder',side_effect=AssertionError('must reuse checkpoint encoder')):
            resumed=resume_training(self.ws,checkpoint,epochs=4)
        full=train_models(self.ws,'demo','t',self.prep.run_id,configs=[replace(config,params={**config.params,'epochs':4})])
        actual=Predictor.load(resumed.bundle).predict(self.frame)
        expected=Predictor.load(full.directory/'candidates/candidate-000').predict(self.frame)
        pd.testing.assert_frame_equal(actual,expected,check_exact=False,rtol=1e-5,atol=1e-6)
        self.assertEqual(resumed.leaderboard[0]['training']['epoch'],4)
        history=json.loads((resumed.bundle/'history.json').read_text())['history']
        self.assertEqual([r['epoch'] for r in history],[1,2,3,4])
        full_history=json.loads((full.directory/'candidates/candidate-000/history.json').read_text())['history']
        np.testing.assert_allclose([r['train_loss'] for r in history],[r['train_loss'] for r in full_history],rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose([r['validation_loss'] for r in history],[r['validation_loss'] for r in full_history],rtol=1e-6,atol=1e-7)
        exported=export_bundle(resumed.bundle,self.root/'portable')
        pd.testing.assert_frame_equal(actual,Predictor.load(exported).predict(self.frame))
        self.assertTrue((export_native(resumed.bundle,self.root/'native')/'architecture.json').exists())
        with self.assertRaises(ValueError): resume_training(self.ws,checkpoint,epochs=2)
        code,payload,error=self.cli('models','checkpoints','demo','t')
        self.assertEqual(code,0,error)
        self.assertGreaterEqual(len(payload['checkpoints']),6)
        (Path(checkpoint)/'state.json').write_text('{}')
        with self.assertRaises(ValueError): resume_training(self.ws,checkpoint,epochs=5)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'pytorch extra')
    def test_pytorch_epoch_resume_parity(self): self.neural_resume('pytorch')

    @unittest.skipUnless(importlib.util.find_spec('tensorflow'),'tensorflow extra')
    def test_tensorflow_epoch_resume_parity(self): self.neural_resume('tensorflow')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'pytorch extra')
    def test_interrupted_training_checkpoint_survives(self):
        checkpoint=[]
        def progress(event):
            if event['stage']=='epoch':
                checkpoint.append(event['checkpoint'])
                raise KeyboardInterrupt()
        config=ModelConfig('pytorch','mlp',{'epochs':3,'hidden_sizes':[4],'batch_size':16})
        with self.assertRaises(KeyboardInterrupt):
            train_models(self.ws,'demo','t',self.prep.run_id,configs=[config],progress=progress)
        self.assertTrue(Path(checkpoint[0]).exists())
        self.assertEqual(training_runs(self.ws,'demo','t'),[])
        result=resume_training(self.ws,checkpoint[0],epochs=3)
        self.assertEqual(result.leaderboard[0]['training']['epoch'],3)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'pytorch extra')
    def test_patience_best_weights_and_explicit_reset(self):
        config=ModelConfig('pytorch','mlp',{'epochs':8,'hidden_sizes':[4],'batch_size':16,'patience':1,'min_delta':1e9})
        result=train_models(self.ws,'demo','t',self.prep.run_id,configs=[config])
        row=result.leaderboard[0]
        self.assertEqual(row['training']['epoch'],2)
        self.assertEqual(row['training']['best_epoch'],1)
        self.assertTrue(row['training']['stopped_early'])
        with self.assertRaises(ValueError): resume_training(self.ws,row['checkpoint'],epochs=4)
        resumed=resume_training(self.ws,row['checkpoint'],epochs=4,reset_patience=True)
        self.assertEqual(resumed.leaderboard[0]['training']['epoch'],3)
        pd.testing.assert_frame_equal(Predictor.load(result.directory/'candidates/candidate-000').predict(self.frame),
                                      Predictor.load(resumed.bundle).predict(self.frame))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'pytorch extra')
    def test_neural_regression_and_data_mismatch(self):
        self.ws.create_task('demo',TaskConfig('reg','x','regression'))
        prep=prepare_task(self.ws,'demo',TaskConfig('reg','x','regression'))
        config=ModelConfig('pytorch','mlp',{'epochs':2,'hidden_sizes':[4]})
        result=train_models(self.ws,'demo','reg',prep.run_id,configs=[config])
        self.assertTrue(np.isfinite(Predictor.load(result.directory/'candidates/candidate-000').predict(self.frame).prediction.astype(float)).all())
        with self.assertRaises(ValueError):
            train_models(self.ws,'demo','t',self.prep.run_id,configs=[config],resume=result.leaderboard[0]['checkpoint'])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'pytorch extra')
    def test_neural_cli_resume_and_plot(self):
        code,result,error=self.cli('models','train','demo','t','--model','pytorch:mlp',
            '--epochs','2','--batch-size','16','--hidden-sizes','4','--learning-rate','0.002')
        self.assertEqual(code,0,error)
        row=result['leaderboard'][0]
        code,resumed,error=self.cli('models','resume',row['checkpoint'],'--epochs','3')
        self.assertEqual(code,0,error)
        self.assertEqual(resumed['leaderboard'][0]['training']['epoch'],3)
        code,payload,error=self.cli('models','history','demo','t',resumed['training_run'],
                                    '--plot',str(self.root/'learning.png'))
        self.assertEqual(code,0,error)
        self.assertEqual(len(payload['rows']),3)
        self.assertTrue(Path(payload['plot']).exists())

    @unittest.skipUnless(importlib.util.find_spec('tensorflow'),'tensorflow extra')
    def test_tensorflow_regression(self):
        prep=prepare_task(self.ws,'demo',TaskConfig('reg','x','regression'))
        result=train_models(self.ws,'demo','reg',prep.run_id,
            configs=[ModelConfig('tensorflow','mlp',{'epochs':2,'hidden_sizes':[4]})])
        values=Predictor.load(result.directory/'candidates/candidate-000').predict(self.frame).prediction.astype(float)
        self.assertTrue(np.isfinite(values).all())


if __name__=='__main__': unittest.main()
