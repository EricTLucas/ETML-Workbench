from contextlib import redirect_stdout,redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from data import DatasetWorkspace
from data.sources import import_sklearn,import_url,import_openml,import_huggingface
from tasks import TaskConfig,ROW_ID
from preprocessing import Recipe,Step
from workflows import prepare_task
from training import train_models,evaluate_test
from models import ModelConfig
from prediction.example import predict_example,input_template
from training.reproducibility import export_reproduction,replay
from etml.cli import main


class DataExperienceTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.ws=DatasetWorkspace(self.root/'datasets')

    def csv(self,name,frame):
        path=self.root/name; frame.to_csv(path,index=False); return path

    def imported(self,unlabeled=False,validation=False):
        train=pd.DataFrame({'x':[1.,np.nan]*30,'y':['a','b']*30})
        test=pd.DataFrame({'x':[999.,888.]*5,'y':['a','b']*5})
        if unlabeled: test=test.drop(columns='y')
        tr=self.csv('train.csv',train); te=self.csv('test.csv',test)
        va=self.csv('validation.csv',pd.DataFrame({'x':[1000.]*10,'y':['a','b']*5})) if validation else None
        return self.ws.import_presplit(tr,test=te,validation=va,dataset_id='demo')

    def trained(self):
        import_sklearn(self.ws,'iris',dataset_id='demo')
        prep=prepare_task(self.ws,'demo',TaskConfig('t','target','classification'))
        result=train_models(self.ws,'demo','t',prep.run_id,configs=[ModelConfig(algorithm='linear')])
        return prep,result

    def test_upload_roles_bytes_and_atomic_failure(self):
        a=io.BytesIO(b'x,y\n1,0\n2,1\n'); b=io.BytesIO(b'x,y\n9,1\n')
        dataset=self.ws.import_split_uploads({'train':('same.csv',a),'test':('same.csv',b)},dataset_id='demo')
        loaded=self.ws.get('demo',verify=True)
        self.assertEqual(set(loaded.manifest.split_files),{'train','test'})
        self.assertEqual((dataset.directory/loaded.manifest.split_files['train']).read_bytes(),a.getvalue())
        self.assertFalse(a.closed)
        with self.assertRaises(ValueError):
            self.ws.import_split_uploads({'train':('x.csv',io.BytesIO(b'12345'))},dataset_id='bad',max_bytes=2)
        self.assertFalse((self.ws.root/'bad').exists())

    def test_presplit_test_preserved_and_fit_training_only(self):
        dataset=self.imported(validation=True)
        recipe=Recipe((Step('fill_missing',('x',),{'strategy':'mean'}).approve(),))
        prep=prepare_task(self.ws,'demo',TaskConfig('t','y','classification'),recipe=recipe,batch_size=7)
        self.assertEqual(prep.split_counts['train'],60)
        self.assertEqual(prep.split_counts['validation'],10)
        self.assertEqual(prep.split_counts['test'],10)
        self.assertEqual(prep.fitted.states[0]['fills']['x'],1.)
        frame=pd.read_parquet(prep.directory/'splits/test/part-00000.parquet')
        self.assertEqual(frame.x.tolist(),[999.,888.]*5)
        assignment=pd.read_parquet(prep.directory/'assignments/part-00000.parquet')
        self.assertEqual(len(assignment),80)
        self.assertTrue(assignment[ROW_ID].is_unique)
        from workflows.preprocessing import PreprocessingWorkflow
        _,source,_=PreprocessingWorkflow(self.ws)._source('demo','raw')
        self.assertEqual(source,[dataset.directory/dataset.manifest.split_files['train']])

    def test_missing_validation_carved_only_from_train_and_unlabeled_test(self):
        self.imported(unlabeled=True)
        prep=prepare_task(self.ws,'demo',TaskConfig('t','y','classification'),validation_fraction=.25,batch_size=8)
        # Stratification rounds each class separately (30*.75 -> 23 each).
        self.assertEqual(prep.split_counts['train'],46)
        self.assertEqual(prep.split_counts['validation'],14)
        for name in ('train','validation'):
            frame=pd.read_parquet(prep.directory/f'splits/{name}/part-00000.parquet')
            self.assertTrue(frame.x.dropna().eq(1.).all())
        result=train_models(self.ws,'demo','t',prep.run_id,configs=[ModelConfig(algorithm='dummy')])
        with self.assertRaisesRegex(ValueError,'no target labels'):
            evaluate_test(self.ws,'demo','t',result.run_id)
        self.assertEqual(predict_example(result.bundle,{'x':5.})['status'],'predicted')

    def test_presplit_schema_mismatch_rejected(self):
        self.ws.import_presplit(self.csv('tr.csv',pd.DataFrame({'x':range(20),'y':[0,1]*10})),
            test=self.csv('te.csv',pd.DataFrame({'wrong':[1]})),dataset_id='demo')
        with self.assertRaisesRegex(ValueError,'columns must match'):
            prepare_task(self.ws,'demo',TaskConfig('t','y','classification'))

    def test_sklearn_and_openml_provenance(self):
        dataset=import_sklearn(self.ws,'iris',dataset_id='iris')
        self.assertEqual(len(pd.read_parquet(dataset.raw_files[0])),150)
        self.assertEqual(dataset.manifest.provenance['class_names'],['setosa','versicolor','virginica'])
        bunch=SimpleNamespace(frame=pd.DataFrame({'x':[1,2],'label':['a','b']}),details={'id':'61'},target_names=['label'])
        with patch('sklearn.datasets.fetch_openml',return_value=bunch) as fetch:
            imported=import_openml(self.ws,61,dataset_id='openml')
            fetch.assert_called_once_with(data_id=61,as_frame=True)
        self.assertEqual(imported.manifest.provenance['data_id'],61)
        with self.assertRaises(ValueError): import_sklearn(self.ws,'iris',max_rows=10)

    def test_url_preserves_bytes_redacts_query_and_enforces_limit(self):
        content=b'x,y\n1,2\n'
        with patch('data.sources.urlopen',return_value=io.BytesIO(content)):
            data=import_url(self.ws,'https://example.com/data.csv?secret=abc',dataset_id='demo')
        self.assertEqual(data.raw_files[0].read_bytes(),content)
        self.assertNotIn('secret',json.dumps(data.manifest.provenance))
        with patch('data.sources.urlopen',return_value=io.BytesIO(content)):
            with self.assertRaises(ValueError): import_url(self.ws,'https://example.com/data.csv',max_bytes=2)

    @unittest.skipUnless(importlib.util.find_spec('datasets'),'huggingface extra')
    def test_huggingface_stream_snapshot_pins_revision_and_roles(self):
        import datasets
        path=self.csv('hub.csv',pd.DataFrame({'x':[1,2,3,4],'y':['a','b','a','b']}))
        stream=datasets.load_dataset('csv',data_files=str(path),split='train',streaming=True)
        with patch('huggingface_hub.HfApi.dataset_info',return_value=SimpleNamespace(sha='abc123')), \
             patch('datasets.load_dataset',return_value=stream) as load:
            dataset=import_huggingface(self.ws,'owner/data',splits={'train':'train','test':'test'},dataset_id='demo')
        self.assertEqual(dataset.manifest.provenance['revision'],'abc123')
        self.assertEqual(set(dataset.manifest.split_files),{'train','test'})
        self.assertEqual(load.call_args.kwargs['revision'],'abc123')
        self.assertTrue(load.call_args.kwargs['streaming'])

    def test_results_example_and_setup_metadata(self):
        _,result=self.trained()
        baseline=result.leaderboard[0]
        self.assertTrue(baseline['is_baseline'])
        self.assertEqual(baseline['improvement_over_baseline'],0.)
        for record in result.leaderboard:
            self.assertGreaterEqual(record['fit_seconds'],0.)
            self.assertIn('f1_weighted',record['validation'])
        metrics=json.loads((result.bundle/'metrics.json').read_text())
        self.assertEqual(len(metrics['validation_details']['confusion_matrix']),3)
        self.assertTrue((result.bundle/'SETUP.md').exists())
        self.assertIn('scikit-learn==',(result.bundle/'requirements.txt').read_text())
        frame=pd.read_parquet(self.ws.get('demo').raw_files[0])
        one=predict_example(result.bundle,json.loads(frame.iloc[[0]].to_json(orient='records'))[0])
        self.assertTrue(one['correct'])
        self.assertAlmostEqual(sum(v['probability'] for v in one['probabilities']),1.,places=6)
        self.assertNotIn('target',input_template(result.bundle)['values'])

    def test_reproduction_package_replays_in_new_process(self):
        _,result=self.trained()
        package=export_reproduction(result.bundle,self.root/'replay',workspace=self.ws)
        self.assertTrue((package/'source/training/runner.py').exists())
        self.assertFalse((package/'data/test.parquet').exists())
        process=subprocess.run([sys.executable,str(package/'reproduce.py')],capture_output=True,text=True)
        self.assertEqual(process.returncode,0,process.stderr)
        output=json.loads(process.stdout)
        self.assertTrue(output['matches'])
        with self.assertRaises(FileExistsError): replay(package)
        with (package/'data/train.parquet').open('ab') as file: file.write(b'changed')
        with self.assertRaises(ValueError): replay(package,output=self.root/'bad')

    def test_cli_sources_and_single_example(self):
        def call(args):
            out=io.StringIO(); err=io.StringIO()
            with redirect_stdout(out),redirect_stderr(err):
                status=main([*args,'--workspace',str(self.ws.root),'--json'])
            self.assertEqual(status,0,err.getvalue())
            return json.loads(out.getvalue())
        call(['datasets','from-sklearn','iris','--id','demo'])
        prep=prepare_task(self.ws,'demo',TaskConfig('t','target','classification'))
        train_models(self.ws,'demo','t',prep.run_id,configs=[ModelConfig(algorithm='linear')])
        result=call(['models','predict-one','--dataset','demo','--task','t','--input',str(self.ws.get('demo').raw_files[0]),'--row','0'])
        self.assertTrue(result['correct'])
        call(['models','input-template','--dataset','demo','--task','t'])
        call(['models','setup','--dataset','demo','--task','t','--output',str(self.root/'setup')])

    @unittest.skipUnless(importlib.util.find_spec('streamlit'),'ui extra')
    def test_ui_upload_slots_and_single_prediction(self):
        from streamlit.testing.v1 import AppTest
        import etml.app
        at=AppTest.from_file(etml.app.__file__,default_timeout=30).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.file_uploader),3)
        _,result=self.trained()
        at.sidebar.text_input(key='workspace').set_value(str(self.ws.root)).run()
        at.sidebar.radio(key='page').set_value('Predict one').run()
        self.assertFalse(at.exception)
        frame=pd.read_parquet(self.ws.get('demo').raw_files[0])
        for widget in at.text_input:
            if widget.label in frame.columns: widget.set_value(str(frame[widget.label].iloc[0]))
        next(button for button in at.button if button.label=='Predict example').click().run()
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertEqual(at.metric[0].label,'Prediction')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'pytorch extra')
    def test_resumed_neural_reproduction(self):
        from training import resume_training
        import_sklearn(self.ws,'iris',dataset_id='demo')
        prep=prepare_task(self.ws,'demo',TaskConfig('t','target','classification'))
        first=train_models(self.ws,'demo','t',prep.run_id,
            configs=[ModelConfig('pytorch','mlp',{'epochs':1,'hidden_sizes':[4],'batch_size':32,'patience':None})])
        second=resume_training(self.ws,first.leaderboard[1]['checkpoint'],epochs=2)
        package=export_reproduction(second.bundle,self.root/'replay-neural',workspace=self.ws)
        self.assertTrue((package/'checkpoint/checkpoint.pt').exists())
        self.assertTrue(replay(package)['matches'])


if __name__=='__main__': unittest.main()
