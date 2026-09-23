import contextlib
import io
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from data.projects import ProjectStore
from workflows.project_preparation import ProjectPreparation
from workflows.project_models import ProjectModels,train_named_tabular
from workflows.model_inspection import ModelInspection,row_at
from preprocessing import Recipe
from splitting import SplitConfig
from models import ModelConfig
from training import train_models
from etml.cli import main


class PredictionFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)
        self.project=ProjectStore(self.root/'projects').create('demo')
        self.source=self.root/'source.csv'
        pd.DataFrame({'x':np.arange(90)/90,'category':['a','b','c']*30,'target':['yes','no']*45}).to_csv(self.source,index=False)
        self.project.workspace.import_files(self.source,dataset_id='data')
        self.service=ProjectPreparation(self.project); self.service.configure('target','classification')
        self.service.save_recipe(Recipe()); fitted,_,_=self.service.fit_preview(); self.service.save_processed(fitted)
        self.args=SimpleNamespace(projects_dir=str(self.root/'projects'),dataset_id=None,max_memory_mb=512,max_rows=200000,batch_size=1000)

    def prepared(self,validation=.2):
        return self.service.split(SplitConfig('stratified',train=.7,validation=validation,test=.3-validation))

    def named(self,validation=.2):
        self.prepared(validation)
        train_named_tabular(self.project,self.service,'sklearn:logistic_regression',{},name='model1')
        return ModelInspection(self.project,'model1')

    def test_no_validation_selected_model_training_and_no_test_reads(self):
        prep=self.prepared(0); paths=[]
        from training.runner import _load_frame
        def read(path,*args): paths.append(str(path).replace('\\','/')); return _load_frame(path,*args)
        with patch('training.runner._load_frame',side_effect=read):
            result=train_models(self.project.workspace,'data',prep.task_id,prep.run_id,configs=[ModelConfig('sklearn','logistic_regression')])
        selected=next(r for r in result.leaderboard if r['candidate']==result.winner)
        self.assertFalse(selected['is_baseline']); self.assertEqual(selected['validation'],{})
        self.assertIsNone(selected['improvement_over_baseline']); self.assertIn('accuracy',selected['training_metrics'])
        self.assertTrue(all('/test/' not in p for p in paths))
        selection=json.loads((result.directory/'selection.json').read_text()); self.assertIsNone(selection['selection_split'])
        from training.experiments import compare_runs
        with self.assertRaisesRegex(ValueError,'not ranked'): compare_runs(self.project.workspace,'data',prep.task_id)
        with self.assertRaisesRegex(ValueError,'exactly one'): train_models(self.project.workspace,'data',prep.task_id,prep.run_id)

    def test_optional_boosters_without_validation(self):
        prep=self.prepared(0)
        for backend in ('xgboost','lightgbm','catboost'):
            if not importlib.util.find_spec(backend): continue
            with self.subTest(backend=backend):
                config=ModelConfig(backend,'boosted_trees',params={'iterations' if backend=='catboost' else 'n_estimators':3})
                result=train_models(self.project.workspace,'data',prep.task_id,prep.run_id,configs=[config])
                self.assertEqual(result.leaderboard[-1]['validation'],{})
                from prediction import Predictor
                self.assertEqual(len(Predictor.load(result.bundle).predict(pd.read_csv(self.source).head(2))),2)
        with self.assertRaisesRegex(ValueError,'early_stopping_rounds'):
            train_models(self.project.workspace,'data',prep.task_id,prep.run_id,configs=[ModelConfig('xgboost','boosted_trees',params={'early_stopping_rounds':2})])

    def test_neural_no_validation_runs_all_epochs_and_can_resume(self):
        prep=self.prepared(0)
        for backend,module in [('pytorch','torch'),('tensorflow','tensorflow')]:
            if not importlib.util.find_spec(module): continue
            with self.subTest(backend=backend):
                events=[]
                config=ModelConfig(backend,'mlp',params={'epochs':2,'patience':1,'hidden_sizes':[4],'batch_size':16})
                result=train_models(self.project.workspace,'data',prep.task_id,prep.run_id,configs=[config],progress=events.append)
                epoch=[e for e in events if e['stage']=='epoch']
                self.assertEqual(len(epoch),2); self.assertTrue(all('validation_loss' not in e for e in epoch))
                from training.checkpoints import resume_training
                resumed=resume_training(self.project.workspace,epoch[-1]['checkpoint'],epochs=3)
                self.assertEqual(resumed.leaderboard[0]['training']['epoch'],3)
                self.assertEqual(resumed.leaderboard[0]['validation'],{})

    def test_presplit_can_keep_all_training_rows(self):
        other=ProjectStore(self.root/'projects').create('presplit')
        frame=pd.read_csv(self.source); train=self.root/'train.csv'; test=self.root/'test.csv'
        frame.iloc[:60].to_csv(train,index=False); frame.iloc[60:].to_csv(test,index=False)
        other.workspace.import_presplit(train,test=test,dataset_id='data')
        service=ProjectPreparation(other); service.configure('target','classification'); service.save_recipe(Recipe())
        fitted,_,_=service.fit_preview(); service.save_processed(fitted)
        result=service.split(SplitConfig('stratified'),validation_fraction=0)
        self.assertEqual(result.prepared_counts,{'train':60,'validation':0,'test':30})

    def test_saved_split_row_true_value_and_details_after_new_split(self):
        inspection=self.named(); original=inspection.data_root()
        expected=row_at(original/'splits/test/part-00000.parquet',1)
        self.service.split(SplitConfig('stratified',.8,.1,.1,seed=7))
        self.assertEqual(inspection.data_root(),original)
        result=inspection.predict_row(); self.assertEqual(result['split'],'test')
        self.assertEqual(result['actual'],expected['target']); self.assertEqual(result['values'],expected)
        self.assertIn('correct',result); self.assertEqual(result['row'],1)
        self.assertIn('effective_parameters',inspection.details()['configuration'])
        with self.assertRaises(ValueError): inspection.predict_row(0)
        with self.assertRaises(ValueError): inspection.predict_row(9999)

    def test_exports_model_and_exact_split_data(self):
        inspection=self.named()
        output=inspection.export(self.root/'export',include_data=True)
        self.assertTrue((output/'data/splits/test/part-00000.parquet').is_file())
        self.assertTrue((output/'model-details.json').is_file())
        from prediction import Predictor
        values=pd.read_csv(self.source).head(1)
        pd.testing.assert_frame_equal(Predictor.load(inspection.bundle).predict(values),Predictor.load(output/'bundle').predict(values))
        with self.assertRaises(FileExistsError): inspection.export(output)
        with self.assertRaisesRegex(ValueError,'outside'): inspection.export(inspection.bundle/'bad')
        with self.assertRaisesRegex(ValueError,'budget'): inspection.export(self.root/'tiny',include_data=True,max_bytes=1)

    def test_explicit_test_metrics_are_saved_for_the_named_candidate(self):
        inspection=self.named(0)
        report=inspection.evaluate_test()
        self.assertIn('accuracy',report['metrics'])
        self.assertEqual(report['candidate'],inspection.manifest['metadata']['candidate'])
        from etml.prediction_wizard import show_metrics
        with contextlib.redirect_stdout(io.StringIO()) as output: show_metrics(ModelInspection(self.project,'model1'))
        self.assertIn('Saved test metrics (held-out)',output.getvalue())

    def test_unlabeled_row_does_not_invent_a_true_value(self):
        other=ProjectStore(self.root/'projects').create('unlabeled')
        frame=pd.read_csv(self.source); test=self.root/'unlabeled.csv'
        frame.drop(columns='target').head(4).to_csv(test,index=False)
        other.workspace.import_presplit(self.source,test=test,dataset_id='data')
        service=ProjectPreparation(other); service.configure('target','classification'); service.save_recipe(Recipe())
        fitted,_,_=service.fit_preview(); service.save_processed(fitted); service.split(SplitConfig('stratified'))
        train_named_tabular(other,service,'sklearn:logistic_regression',{},name='model1')
        result=ModelInspection(other,'model1').predict_row()
        self.assertEqual(result['status'],'predicted'); self.assertNotIn('actual',result)

    def test_exploration_row_returns_stored_assignment_and_no_ground_truth(self):
        from training.exploration import train_exploration
        store=ProjectModels(self.project); record=store.create('sklearn:dbscan',name='clusters')
        destination=store.root/'clusters/bundle'
        train_exploration(self.source,'sklearn:dbscan',destination,features=['x'])
        store.update('clusters',status='complete',bundle=destination.relative_to(self.project.directory).as_posix())
        result=ModelInspection(self.project,'clusters').predict_row()
        self.assertEqual(result['status'],'stored_assignment'); self.assertNotIn('actual',result)

    def test_specialized_series_row_matches_saved_true_value(self):
        if not importlib.util.find_spec('statsmodels'): self.skipTest('statsmodels optional')
        from data.model_inputs import prepare_model_input
        from training.specialized import train_specialized
        source=self.root/'series.csv'
        pd.DataFrame({'date':pd.date_range('2020-01-01',periods=50),'y':np.sin(np.arange(50)/5)}).to_csv(source,index=False)
        inputs=prepare_model_input(self.project,{'modality':'time_series','task':'forecasting','train':str(source),
            'columns':{'time':'date','value':'y'}})
        store=ProjectModels(self.project); store.create('statsmodels:arima',name='forecast',input_reference=str(inputs))
        dest=store.root/'forecast/bundle'; train_specialized(inputs,'statsmodels:arima',dest,params={'order':[1,0,0]})
        store.update('forecast',status='complete',bundle=dest.relative_to(self.project.directory).as_posix())
        inspection=ModelInspection(self.project,'forecast'); result=inspection.predict_row()
        self.assertAlmostEqual(result['actual'],row_at(inputs/'splits/test.parquet',1)['y'])
        self.assertIn('prediction',result); self.assertIn('Fixed-origin',result['protocol'])

    def test_no_validation_replay_uses_training_for_parity(self):
        inspection=self.named(0)
        from training.reproducibility import export_reproduction,replay
        path=export_reproduction(inspection.bundle,self.root/'replay',workspace=self.project.workspace)
        result=replay(path)
        self.assertTrue(result['matches']); self.assertEqual(result['parity_split'],'train')

    def test_prediction_wizard_default_row_metrics_and_export(self):
        inspection=self.named()
        from etml.prediction_wizard import run
        actions=iter([0,0,2,3,0,6]); answers=iter(['',str(self.root/'wizard-export')]); output=io.StringIO()
        with contextlib.redirect_stdout(output):
            code=run(self.project,self.args,name='model1',choose=lambda *a:next(actions),ask=lambda *a:next(answers))
        self.assertEqual(code,0)
        self.assertIn('True value:',output.getvalue()); self.assertIn('f1_macro',output.getvalue())
        self.assertIn('Test row 1',output.getvalue())
        self.assertTrue((self.root/'wizard-export/bundle/manifest.json').exists())

    def test_duplicate_artifact_summary_removed(self):
        self.prepared(); output=io.StringIO()
        with patch('builtins.input',side_effect=['1']),contextlib.redirect_stdout(output):
            code=main(['demo','--continue','--projects-dir',str(self.root/'projects')])
        self.assertEqual(code,0); self.assertEqual(output.getvalue().count('Saved project data'),1)

    def test_split_automatically_opens_model_wizard(self):
        from etml.preparation_wizard import run
        with patch('etml.project_cli._choose',side_effect=[0,3,0]),patch('etml.project_cli._ask',return_value=''),patch('etml.model_wizard.run',return_value=0) as models,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run(self.project,self.args),0)
        models.assert_called_once(); self.assertEqual(self.service.state()['split']['prepared_counts']['validation'],0)

    def test_training_automatically_opens_predictions_for_selected_name(self):
        self.prepared()
        from etml.model_wizard import run
        from models.catalog import CATALOG
        with patch('etml.model_wizard._select',return_value=CATALOG['sklearn:logistic_regression']),patch('etml.model_wizard._settings',return_value={}),patch('etml.prediction_wizard.run',return_value=0) as prediction,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run(self.project,self.args,choose=lambda *a:0,ask=lambda *a:''),0)
        self.assertEqual(prediction.call_args.kwargs['name'],'model1')

    def test_main_project_menu_contains_prediction_option(self):
        self.named(); output=io.StringIO()
        with patch('etml.project_cli.sys.stdin.isatty',return_value=True),patch('builtins.input',side_effect=['2','1','3','1','7']),contextlib.redirect_stdout(output):
            self.assertEqual(main(['--projects-dir',str(self.root/'projects')]),0,output.getvalue())
        self.assertIn('Project actions',output.getvalue()); self.assertIn('Prediction and model tools',output.getvalue())
        self.assertEqual(output.getvalue().count('Saved project data'),1)

    def test_advanced_training_cli_without_validation(self):
        prep=self.prepared(0); output=io.StringIO()
        with contextlib.redirect_stdout(output),contextlib.redirect_stderr(io.StringIO()):
            code=main(['models','train','data',prep.task_id,prep.run_id,'--model','sklearn:logistic_regression',
                       '--workspace',str(self.project.workspace.root)])
        self.assertEqual(code,0,output.getvalue()); self.assertIn('Training (in-sample)',output.getvalue())
