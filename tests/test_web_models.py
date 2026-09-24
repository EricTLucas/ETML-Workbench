import importlib.util
import io
import json
import zipfile
import unittest
from unittest.mock import patch
import test_web_ui as _web_tests
HAS_UI = _web_tests.HAS_UI


@unittest.skipUnless(HAS_UI, 'Install .[ui,test-ui]')
class ModelWebTests(unittest.TestCase):
    setUp= _web_tests.WebUITests.setUp
    tearDown= _web_tests.WebUITests.tearDown
    get= _web_tests.WebUITests.get
    post= _web_tests.WebUITests.post
    wait= _web_tests.WebUITests.wait
    imported= _web_tests.WebUITests.imported
    prepare= _web_tests.WebUITests.prepare

    def split(self,validation=.2):
        did=self.prepare()
        self.wait(self.post('projects/demo/split',{'dataset_id':did,'config':{
            'strategy':'stratified','train':.8-validation,'validation':validation,'test':.2}}))
        return did

    def train(self,did,key='sklearn:logistic_regression',params=None,name='model1'):
        queued=self.post('projects/demo/models/train',{'dataset_id':did,'name':name,'key':key,
            'task':'classification','params':params or {}})
        self.wait(queued)
        return self.get('projects/demo/models/'+name),self.get('jobs/'+queued['job_id'])

    def test_csv_predictions_download_and_compare(self):
        import pandas as pd
        did=self.split()
        model,_=self.train(did)
        self.train(did,name='model2')
        group=self.get('projects/demo/models/compare')['groups'][0]
        self.assertEqual(len(group['models']),2)
        row=self.wait(self.post('projects/demo/models/model1/predict',{'mode':'row','split':'test','row':1}))
        frame=pd.DataFrame([row['values']]*10005)
        target=model['custom_target']['name']
        frame[target]='original label'
        response=self.client.post('/api/projects/demo/models/model1/predict-csv',headers=self.headers,
            files={'file':('examples.csv',frame.to_csv(index=False).encode(),'text/csv')})
        self.assertEqual(response.status_code,200,response.text)
        result=self.wait(response.json())
        self.assertEqual(result['rows'],10005)
        output=pd.read_csv(io.BytesIO(self.client.get(result['url']).content))
        self.assertEqual(len(output),10005)
        self.assertTrue((output[target]==row['prediction']).all())
        self.assertTrue((output[target+'_actual']=='original label').all())
        self.assertNotIn('prediction_status',output.columns)
        self.assertEqual(self.client.post('/api/projects/demo/models/model1/predict-csv',headers=self.headers,
            files={'file':('bad.txt',b'x', 'text/plain')}).status_code,400)

    def test_filtered_csv_and_analysis_report(self):
        import pandas as pd
        did=self.split();model,_=self.train(did)
        row=self.wait(self.post('projects/demo/models/model1/predict',{'mode':'row','split':'test','row':1}))
        frame=pd.DataFrame([row['values']]*6);frame['identifier']=['001','002','003','004','005','006']
        queued=self.client.post('/api/projects/demo/models/model1/predict-csv',headers=self.headers,
            files={'file':('rows.csv',frame.to_csv(index=False).encode(),'text/csv')}).json()
        result=self.wait(queued)
        filtered=self.wait(self.post('projects/demo/prediction-exports/'+result['id']+'/filter',{
            'columns':['identifier','target'],'include_rows':'2-5','exclude_rows':'3'}))
        output=pd.read_csv(io.BytesIO(self.client.get(filtered['url']).content),dtype=str)
        self.assertEqual(output.columns.tolist(),['identifier','target'])
        self.assertEqual(output.identifier.tolist(),['002','004','005'])
        self.wait(self.post('projects/demo/prediction-exports/'+result['id']+'/filter',{
            'columns':['target'],'include_rows':'999'}),False)
        reference=self.get('projects/demo/models/compare')['groups'][0]['reference']
        report=self.wait(self.post('projects/demo/analysis-summary',{'reference':reference}))
        html=self.client.get(report['url']).text
        self.assertEqual(report['best_model'],'model1')
        self.assertIn('Column overview',html)
        self.assertIn('Model leaderboard',html)
        self.assertIn('Best model: model1',html)
        self.assertIn('Test results',html)
        self.assertIn('data:image/png;base64,',html)
        self.assertNotIn('<script',html)
        self.assertLess(html.index('Column overview'),html.index('Model leaderboard'))
        self.assertLess(html.index('Model leaderboard'),html.index('Best model:'))

    def test_csv_invalid_schema_fails_cleanly(self):
        did=self.split();self.train(did)
        response=self.client.post('/api/projects/demo/models/model1/predict-csv',headers=self.headers,
            files={'file':('bad.csv',b'wrong\n1\n', 'text/csv')})
        self.wait(response.json(),False)
        self.assertEqual(list((self.root/'projects/demo/exports').rglob('predictions.csv')),[])

    def test_catalog_and_default_name(self):
        overview=self.get('projects/demo/models')
        self.assertEqual(overview['default_name'],'model1')
        keys={e['key'] for e in overview['catalog']}
        self.assertTrue({'sklearn:random_forest','torchvision:resnet18','transformers:bert','statsmodels:arima'}<=keys)

    def test_train_results_prediction_details_export(self):
        did=self.split()
        model,job=self.train(did)
        self.assertEqual(model['record']['status'],'complete')
        self.assertIn('accuracy',model['details']['metrics']['validation'])
        self.assertIn('accuracy',model['record']['test_evaluation']['metrics'])
        self.assertEqual(model['details']['fitting_split'],'train')
        self.assertTrue(any(e['stage']=='training' for e in job['events']))
        self.assertEqual(self.get('projects/demo/models')['default_name'],'model2')
        result=self.wait(self.post('projects/demo/models/model1/predict',{'mode':'row','split':'test','row':1}))
        self.assertIn('actual',result)
        self.assertIn('prediction',result)
        custom=self.wait(self.post('projects/demo/models/model1/predict',{'mode':'custom','values':result['values']}))
        self.assertEqual(custom['prediction'],result['prediction'])
        exported=self.wait(self.post('projects/demo/models/model1/export',{'include_data':True}))
        download=self.client.get(exported['url'])
        self.assertEqual(download.status_code,200)
        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            self.assertIn('model-details.json',archive.namelist())
            self.assertTrue(any(n.startswith('data/') for n in archive.namelist()))
        self.assertEqual(self.client.get('/downloads/demo/../secret').status_code,404)

    def test_no_validation_and_invalid_row(self):
        did=self.split(0)
        model,_=self.train(did)
        self.assertEqual(model['details']['metrics']['validation'],{})
        self.assertIn('accuracy',model['record']['test_evaluation']['metrics'])
        error=self.wait(self.post('projects/demo/models/model1/predict',{'mode':'row','split':'test','row':99999}),False)
        self.assertIn('Row number',error)
        self.assertEqual(self.get('projects/demo/models/model1')['record']['status'],'complete')

    def test_failed_configuration_can_retry(self):
        did=self.split()
        with patch('training.train_models',side_effect=ImportError('Install optional dependency')):
            self.wait(self.post('projects/demo/models/train',{'dataset_id':did,'name':'model1',
                'key':'sklearn:logistic_regression','task':'classification'}),False)
        saved=self.get('projects/demo/models/model1')
        self.assertEqual(saved['record']['status'],'failed')
        self.wait(self.post('projects/demo/models/train',{'name':'model1','action':'retry'}))
        self.assertEqual(self.get('projects/demo/models/model1')['record']['status'],'complete')

    def test_exploration_assignment_and_custom_contract(self):
        did=self.imported()
        source=self.get('projects/demo')['dataset']['sources'][0]
        self.wait(self.post('projects/demo/models/train',{'name':'clusters','key':'sklearn:hierarchical',
            'task':'clustering','source':source,'features':['sepal length (cm)','sepal width (cm)'],
            'params':{'n_clusters':3}}))
        model=self.get('projects/demo/models/clusters')
        self.assertEqual(model['kind'],'exploration_model')
        self.assertFalse(model['custom_supported'])
        result=self.wait(self.post('projects/demo/models/clusters/predict',{'mode':'row','split':'train','row':1}))
        self.assertEqual(result['status'],'stored_assignment')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'pytorch extra')
    def test_live_epoch_events_and_continuation(self):
        did=self.split()
        model,job=self.train(did,'pytorch:mlp',{'epochs':2,'hidden_sizes':[4],'batch_size':32,'patience':None})
        epochs=[e for e in job['events'] if e['stage']=='epoch']
        self.assertEqual(len(epochs),2)
        self.assertIn('train_loss',epochs[-1])
        self.assertTrue(model['learning_curves'][0]['image'].startswith('data:image/png;base64,'))
        self.assertTrue(model['can_resume'])
        self.wait(self.post('projects/demo/models/train',{'action':'resume','name':'model1','additional_epochs':1}))
        result=self.get('projects/demo/models/model1')
        self.assertEqual(result['record']['status'],'complete')
        self.assertTrue(result['record']['history'])

    def test_specialized_input_training_and_prediction(self):
        import pandas as pd
        path=self.root/'interactions.csv'
        pd.DataFrame([{'user':str(u),'item':str((u+i)%12)} for u in range(8) for i in range(8)]).to_csv(path,index=False)
        self.wait(self.post('projects/demo/models/train',{'name':'recommender','key':'scipy:svd',
            'task':'recommendation','params':{'factors':2},
            'input_spec':{'train':str(path),'columns':{'user':'user','item':'item'}}}))
        model=self.get('projects/demo/models/recommender')
        self.assertEqual(model['kind'],'specialized_model')
        self.assertIn('recall_at_10',model['details']['metrics']['validation'])
        result=self.wait(self.post('projects/demo/models/recommender/predict',{'mode':'custom','values':{'user':'0','k':3}}))
        self.assertIn('recommendations',result)
        self.assertEqual(len(self.get('projects/demo/models')['inputs']),1)


if __name__=='__main__': unittest.main()
