import importlib.util
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

HAS_UI=all(importlib.util.find_spec(m) for m in ('fastapi','httpx','multipart'))


@unittest.skipUnless(HAS_UI, 'Install .[ui,test-ui]')
class WebUITests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from etml.web.server import create_app
        self.temp=TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.app=create_app(self.root/'projects', max_bytes=1024*1024)
        self.client=TestClient(self.app).__enter__()
        page=self.client.get('/')
        self.token=re.search(r'name="etml-token" content="([^"]+)',page.text).group(1)
        self.headers={'x-etml-token':self.token}
        self.post('projects',{'name':'demo'})

    def tearDown(self):
        self.client.__exit__(None,None,None)
        self.temp.cleanup()

    def get(self,path):
        r=self.client.get('/api/'+path,headers=self.headers)
        self.assertEqual(r.status_code,200,r.text)
        return r.json()

    def post(self,path,payload):
        r=self.client.post('/api/'+path,json=payload,headers=self.headers)
        self.assertEqual(r.status_code,200,r.text)
        return r.json()

    def wait(self,value,success=True):
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            job=self.get('jobs/'+value['job_id'])
            if job['status'] in {'done','failed'}:
                self.assertEqual(job['status'],'done' if success else 'failed',job)
                return job.get('result',job.get('error'))
            time.sleep(.02)
        self.fail('Job did not finish')

    def imported(self):
        self.wait(self.post('projects/demo/import',{'source':'sklearn','name':'iris','analyze':False}))
        return self.get('projects/demo')['dataset']['id']

    def prepare(self):
        did=self.imported()
        suggested=self.wait(self.post('projects/demo/suggest',{'dataset_id':did,'target':'target','task_type':'classification'}))
        preview=self.wait(self.post('projects/demo/preview',{'dataset_id':did,'recipe':suggested['recipe']}))
        self.wait(self.post('projects/demo/save',{'dataset_id':did,'token':preview['token']}))
        return did

    def test_project_and_session_boundaries(self):
        self.assertEqual(self.client.get('/api/bootstrap').status_code,403)
        self.assertEqual(self.client.post('/api/projects',headers={**self.headers,'origin':'https://evil.example'},json={'name':'bad'}).status_code,403)
        self.assertEqual(self.client.get('/',headers={'host':'evil.example'}).status_code,400)
        self.assertEqual(self.get('bootstrap')['projects'],['demo'])
        self.assertIsNone(self.get('projects/demo')['dataset'])
        self.assertEqual(self.client.post('/api/projects',headers=self.headers,json={'name':'../escape'}).status_code,400)
        self.assertEqual(self.client.get('/static/app.js').status_code,200)

    def test_complete_flow_and_reopen(self):
        did=self.prepare()
        split=self.wait(self.post('projects/demo/split',{'dataset_id':did,'config':{'strategy':'stratified','train':.8,'validation':0,'test':.2,'seed':42}}))
        self.assertEqual(split['prepared_counts']['train'],120)
        self.assertEqual(split['prepared_counts']['test'],30)
        state=self.get('projects/demo')
        self.assertEqual(state['dataset']['state']['split']['run_id'],split['run_id'])
        self.assertEqual(len(state['dataset']['preview']['rows']),10)
        self.assertTrue(any('Processed' in a['label'] for a in state['dataset']['artifacts']))
        from data.projects import ProjectStore
        from workflows.project_preparation import ProjectPreparation
        prep=ProjectPreparation(ProjectStore(self.root/'projects').get('demo'))
        self.assertEqual(prep.state()['split']['run_id'],split['run_id'])
        self.assertTrue((prep.dataset.directory/split['path']/'manifest.json').exists())

    def test_actual_eda_and_report_assets(self):
        did=self.imported()
        result=self.wait(self.post('projects/demo/analyze',{'dataset_id':did}))
        url='/reports/demo/'+did+'/'+result['report']
        report=self.client.get(url)
        self.assertEqual(report.status_code,200)
        self.assertIn('ETML',report.text)
        self.assertIn("frame-ancestors 'self'",report.headers['content-security-policy'])
        sources=re.findall(r'src="([^"]+\.png)"',report.text)
        for src in sources[:2]:
            asset=url.rsplit('/',1)[0]+'/'+src
            self.assertEqual(self.client.get(asset).status_code,200)
        self.assertIn(self.client.get('/reports/demo/'+did+'/../manifest.json').status_code,(400,404))

    def test_upload_and_path_import(self):
        csv=b'x,target\n1,yes\n2,no\n3,yes\n4,no\n'
        result=self.client.post('/api/projects/demo/upload',headers=self.headers,files={'file':('sample.csv',csv,'text/csv')},data={'analyze':'false'})
        self.assertEqual(result.status_code,200,result.text)
        self.wait(result.json())
        data=self.get('projects/demo')['dataset']
        self.assertEqual(data['preview']['rows'][0],[1,'yes'])
        source=self.root/'another.csv';source.write_bytes(csv)
        imported=self.wait(self.post('projects/demo/import',{'source':'path','path':str(source),'analyze':False}))
        self.assertNotEqual(data['id'],imported['dataset_id'])
        self.assertEqual(len(self.get('projects/demo')['datasets']),2)

    def test_invalid_upload_and_size_limit(self):
        r=self.client.post('/api/projects/demo/upload',headers=self.headers,files={'file':('program.exe',b'bad')})
        self.assertEqual(r.status_code,400)
        r=self.client.post('/api/projects/demo/upload',headers=self.headers,files={'file':('huge.csv',b'x'*(1024*1024+1))})
        self.assertEqual(r.status_code,413)
        self.assertIsNone(self.get('projects/demo')['dataset'])

    def test_preview_token_rejected_after_edit(self):
        did=self.imported()
        self.wait(self.post('projects/demo/suggest',{'dataset_id':did,'target':'target','task_type':'classification'}))
        recipe={'format_version':1,'name':'Example','steps':[]}
        first=self.wait(self.post('projects/demo/preview',{'dataset_id':did,'recipe':recipe}))
        self.wait(self.post('projects/demo/preview',{'dataset_id':did,'recipe':recipe}))
        error=self.wait(self.post('projects/demo/save',{'dataset_id':did,'token':first['token']}),False)
        self.assertIn('Preview',error)
        self.assertNotIn('processed',self.get('projects/demo')['dataset']['state'])

    def test_custom_recipe_and_missing_values(self):
        csv=self.root/'missing.csv'
        csv.write_text('age,target\n,0\n20,1\n30,0\n40,1\n')
        did=self.wait(self.post('projects/demo/import',{'source':'path','path':str(csv),'analyze':False}))['dataset_id']
        self.wait(self.post('projects/demo/suggest',{'dataset_id':did,'target':'target','task_type':'classification'}))
        recipe={'format_version':1,'name':'Median','steps':[{'operation':'fill_missing','columns':['age'],'params':{'strategy':'median'},'enabled':True}]}
        result=self.wait(self.post('projects/demo/preview',{'dataset_id':did,'recipe':recipe}))
        self.assertIsNone(result['before']['rows'][0][0])
        self.assertEqual(result['after']['rows'][0][0],30)
        self.assertEqual(result['affected_columns'],['age'])
        self.assertEqual(result['recipe']['steps'][0]['status'],'approved')
        saved=self.wait(self.post('projects/demo/save',{'dataset_id':did,'token':result['token']}))
        self.assertEqual(saved['rows'],4)

    def test_job_failure_releases_project(self):
        error=self.wait(self.post('projects/demo/import',{'source':'path','path':str(self.root/'missing.csv')}),False)
        self.assertTrue(error)
        self.assertIsNone(self.get('projects/demo')['busy'])
        self.imported()

    def test_import_survives_report_failure(self):
        with patch.object(self.app.state.service,'analyze',side_effect=RuntimeError('report failed')):
            result=self.wait(self.post('projects/demo/import',{'source':'sklearn','name':'iris'}))
        self.assertIn('Data saved',result['warning'])
        self.assertIsNotNone(self.get('projects/demo')['dataset'])

    def test_project_rejects_overlapping_mutations(self):
        event=threading.Event()
        service=self.app.state.service
        try:
            job=service.submit('demo','Waiting',lambda:event.wait(5))
            response=self.client.post('/api/projects/demo/import',headers=self.headers,json={'source':'sklearn','name':'iris','analyze':False})
            self.assertEqual(response.status_code,400)
        finally:
            event.set()
        self.wait(job)

    def test_invalid_split_preserves_previous_split(self):
        did=self.prepare()
        config={'train':.8,'validation':0,'test':.2}
        first=self.wait(self.post('projects/demo/split',{'dataset_id':did,'config':config}))
        self.wait(self.post('projects/demo/split',{'dataset_id':did,'config':{'train':.8,'validation':.2,'test':.2}}),False)
        self.assertEqual(self.get('projects/demo')['dataset']['state']['split']['run_id'],first['run_id'])

    def test_failed_preview_preserves_saved_preparation(self):
        did=self.prepare()
        before=self.get('projects/demo')['dataset']['state']
        recipe={'format_version':1,'name':'Bad fit','steps':[{'operation':'fill_missing',
            'columns':['sepal length (cm)'],'params':{'strategy':'median','max_unique':1}}]}
        self.wait(self.post('projects/demo/preview',{'dataset_id':did,'recipe':recipe}),False)
        after=self.get('projects/demo')['dataset']['state']
        self.assertEqual(before['processed'],after['processed'])
        self.assertEqual(before['recipe'],after['recipe'])

    def test_proposed_recipe_survives_reopen(self):
        did=self.imported()
        result=self.wait(self.post('projects/demo/suggest',{'dataset_id':did,'target':'target','task_type':'classification'}))
        self.assertEqual(self.get('projects/demo')['dataset']['recipe'],result['recipe'])

    def test_presplit_project_from_cli(self):
        from data.projects import ProjectStore
        import pandas as pd
        project=ProjectStore(self.root/'projects').get('demo')
        train=self.root/'train.csv'; test=self.root/'test.csv'
        pd.DataFrame({'x':range(40),'target':[0,1]*20}).to_csv(train,index=False)
        pd.DataFrame({'x':range(100,110),'target':[0,1]*5}).to_csv(test,index=False)
        did=project.workspace.import_presplit(train,test=test).dataset_id
        self.wait(self.post('projects/demo/suggest',{'dataset_id':did,'target':'target','task_type':'classification'}))
        preview=self.wait(self.post('projects/demo/preview',{'dataset_id':did,'recipe':{'format_version':1,'name':'No changes','steps':[]}}))
        self.wait(self.post('projects/demo/save',{'dataset_id':did,'token':preview['token']}))
        split=self.wait(self.post('projects/demo/split',{'dataset_id':did,'config':{},'validation_fraction':0}))
        self.assertEqual(split['prepared_counts']['train'],40)
        self.assertEqual(split['prepared_counts']['test'],10)


if __name__=='__main__':
    unittest.main()
