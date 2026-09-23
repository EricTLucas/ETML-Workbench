import contextlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from data.projects import ProjectStore
from preprocessing import Recipe
from splitting import SplitConfig
from workflows.project_preparation import ProjectPreparation
from workflows.project_models import ProjectModels,train_named_tabular
from workflows.model_lifecycle import train_saved,saved_record
from etml.prediction_wizard import run


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
        self.project=ProjectStore(self.root/'projects').create('demo')
        source=self.root/'data.csv'; pd.DataFrame({'x':np.arange(40),'target':np.arange(40)*2.}).to_csv(source,index=False)
        self.project.workspace.import_files(source,dataset_id='data')
        self.service=ProjectPreparation(self.project); self.service.configure('target','regression'); self.service.save_recipe(Recipe())
        fitted,_,_=self.service.fit_preview(); self.service.save_processed(fitted)
        self.service.split(SplitConfig('random',.8,0,.2))
        self.args=SimpleNamespace(max_rows=200000,max_memory_mb=512,projects_dir=str(self.root/'projects'),dataset_id=None)

    def fail_model(self):
        with patch('training.train_models',side_effect=ImportError('install package')):
            with self.assertRaises(ImportError): train_named_tabular(self.project,self.service,'sklearn:ridge',{},name='model1')
        return saved_record(self.project,'model1')

    def test_failed_retry_uses_original_split_and_name(self):
        failed=self.fail_model(); original=failed['input']['run_id']
        self.service.split(SplitConfig('random',.7,.2,.1,seed=9))
        result=train_saved(self.project,'model1')
        self.assertEqual(result['status'],'complete'); self.assertEqual(result['input']['run_id'],original)
        self.assertEqual(len(ProjectModels(self.project).list()),1); self.assertIsNone(result['error'])
        self.assertEqual(result['history'][-1]['status'],'failed')

    def test_failed_model_is_selectable_and_retryable_in_menu(self):
        self.fail_model(); actions=iter([0,0,6]); output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(run(self.project,self.args,choose=lambda *a:next(actions),ask=lambda *a:''),0)
        self.assertIn('saved configuration',output.getvalue()); self.assertEqual(saved_record(self.project,'model1')['status'],'complete')

    def test_retry_failure_stays_available(self):
        self.fail_model(); actions=iter([0,0,4]); output=io.StringIO()
        with patch('training.train_models',side_effect=ImportError('still missing')),contextlib.redirect_stdout(output):
            self.assertEqual(run(self.project,self.args,choose=lambda *a:next(actions),ask=lambda *a:''),0)
        self.assertEqual(saved_record(self.project,'model1')['status'],'failed'); self.assertIn('still missing',output.getvalue())

    def test_new_model_action_when_inspecting_and_when_empty(self):
        with patch('etml.model_wizard.run',return_value=0) as new:
            self.assertEqual(run(self.project,self.args,choose=lambda title,choices:choices.index('Create a new model')),0)
            self.assertTrue(new.call_args.kwargs['start_new'])
        train_named_tabular(self.project,self.service,'sklearn:ridge',{},name='model1')
        with patch('etml.model_wizard.run',return_value=0) as new,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run(self.project,self.args,name='model1',choose=lambda title,choices:choices.index('Create a new model')),0)
            self.assertTrue(new.call_args.kwargs['start_new'])

    def test_split_order_and_custom_selection(self):
        from etml.preparation_wizard import _split_config
        prompts=[]
        def choose(title,choices):
            prompts.append((title,choices)); return 3 if 'Train /' in title else 0
        config,_=_split_config(self.service,choose,lambda *a:'')
        labels=prompts[0][1]
        self.assertEqual(labels[3],'80% / 0% / 20%'); self.assertEqual(labels[4],'Custom percentages')
        self.assertEqual(config.validation,0)

    def test_neural_continuation_preserves_previous_model_and_clears_test_scores(self):
        for backend,module in [('pytorch','torch'),('tensorflow','tensorflow')]:
            if not importlib.util.find_spec(module): continue
            with self.subTest(backend=backend):
                name=backend
                train_named_tabular(self.project,self.service,backend+':mlp',{'epochs':2,'hidden_sizes':[4],'batch_size':16},name=name)
                old=saved_record(self.project,name); ProjectModels(self.project).update(name,test_evaluation={'metrics':{'rmse':123}})
                result=train_saved(self.project,name,additional_epochs=2)
                self.assertEqual(result['metrics']['training']['epoch'],4)
                self.assertNotEqual(old['bundle'],result['bundle']); self.assertTrue((self.project.directory/old['bundle']).exists())
                self.assertIsNone(result['test_evaluation']); self.assertEqual(result['params']['epochs'],4)
                with patch('training.checkpoints.resume_training',side_effect=RuntimeError('training error')):
                    with self.assertRaises(RuntimeError): train_saved(self.project,name,additional_epochs=1)
                current=saved_record(self.project,name)
                self.assertEqual(current['status'],'complete'); self.assertEqual(current['bundle'],result['bundle'])

    def test_specialized_failed_retry(self):
        if not importlib.util.find_spec('statsmodels'): self.skipTest('optional statsmodels')
        from data.model_inputs import prepare_model_input
        source=self.root/'series.csv'; pd.DataFrame({'time':pd.date_range('2020-01-01',periods=40),'y':np.arange(40)+np.sin(np.arange(40))}).to_csv(source,index=False)
        inputs=prepare_model_input(self.project,{'modality':'time_series','task':'forecasting','train':str(source),'columns':{'time':'time','value':'y'}})
        store=ProjectModels(self.project); store.create('statsmodels:arima',name='forecast',params={'order':[1,0,0]},input_reference=str(inputs))
        store.update('forecast',status='failed',error='missing statsmodels')
        result=train_saved(self.project,'forecast')
        self.assertEqual(result['status'],'complete'); self.assertIn('rmse',result['metrics']['validation'])
