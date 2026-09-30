import unittest
from unittest.mock import patch
import pandas as pd
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import test_web_ui as web

class CoreTests(unittest.TestCase):
    def test_replace_and_restore(self):
        from preprocessing import Recipe, Step
        from preprocessing.executor import fit_batches, FittedRecipe
        frame=pd.DataFrame({'a':[' cat ','dog',None],'b':[1,2,3]})
        recipe=Recipe((Step('find_replace',('a',),{'find':'cat','replace':'fox','mode':'substring'}),)).approve_all()
        fitted=fit_batches(lambda:iter([frame]),list(frame),recipe)
        self.assertEqual(fitted.apply(frame)['a'].iloc[:2].tolist(),[' fox ','dog'])
        self.assertTrue(pd.isna(fitted.apply(frame)['a'].iloc[2]))
        restored=FittedRecipe.from_dict(fitted.to_dict())
        pd.testing.assert_frame_equal(restored.apply(frame),fitted.apply(frame))
        self.assertEqual(frame.loc[0,'a'],' cat ')

    def test_validation_and_recommendations(self):
        from data.intelligence import validate
        from models.catalog import CATALOG
        from models.recommendations import recommend
        with self.assertRaises(ValueError):validate({'summary':'x','columns':[],'possible_targets':[],'dependencies':[]},{'a'})
        self.assertEqual(recommend(CATALOG['sklearn:knn'],3,2)['defaults']['n_neighbors'],1)
        self.assertEqual(recommend(CATALOG['pytorch:mlp'],100,20)['defaults']['hidden_sizes'],[32,16])
        self.assertEqual(CATALOG['pytorch:mlp'].defaults['hidden_sizes'],[64,32])

@unittest.skipUnless(web.HAS_UI,'UI dependencies required')
class ContextWebTests(unittest.TestCase):
    setUp=web.WebUITests.setUp
    tearDown=web.WebUITests.tearDown
    get=web.WebUITests.get
    post=web.WebUITests.post
    wait=web.WebUITests.wait

    def test_context_recipes_report_history(self):
        from data import intelligence
        from eda_tool.artifacts.runs import load_run
        from data.projects import ProjectStore
        source=self.root/'data.csv'
        pd.DataFrame({'a':[1,2,3,4,5,6], 'copy':[1,2,3,4,5,6], 'label':[0,1,0,1,0,1]}).to_csv(source,index=False)
        output={'summary':'Example <script>alert(1)</script>', 'columns':[{'name':c,'description':'Description '+c} for c in ['a','copy','label']], 'possible_targets':['label'], 'dependencies':[{'columns':['a','copy'],'kind':'redundancy','reason':'Possibly duplicated','recommendation':'Review removing copy','drop_candidates':['copy']}]}
        with patch.object(intelligence,'request_summary',return_value=(output,'mock-model')) as mocked:
            result=self.wait(self.post('projects/demo/import',{'source':'path','path':str(source),'analyze':True,'explanation':'A test dataset','ai_analysis':True,'ai_samples':False}))
            self.assertNotIn('sample_values',mocked.call_args.args[0])
        did=result['dataset_id'];dataset=self.get('projects/demo')['dataset']
        self.assertEqual(dataset['intelligence']['analysis']['summary'],output['summary'])
        project=ProjectStore(self.root/'projects').get('demo');ds=project.workspace.get(did)
        report=ds.profiles_dir/result['report'];html=report.read_text(encoding='utf-8')
        self.assertIn('&lt;script&gt;',html);self.assertIn('Possible column dependencies',html)
        load_run(report.parent)
        proposed=self.wait(self.post('projects/demo/suggest',{'dataset_id':did,'target':'label','task_type':'classification'}))['recipe']
        drop=next(s for s in proposed['steps'] if s['operation']=='drop_columns')
        self.assertFalse(drop['enabled']);drop['enabled']=True
        preview=self.wait(self.post('projects/demo/preview',{'dataset_id':did,'recipe':proposed}))
        self.wait(self.post('projects/demo/save',{'dataset_id':did,'token':preview['token']}))
        html=report.read_text(encoding='utf-8');self.assertIn('Preprocessing history',html);self.assertIn('drop_columns on copy',html)
        load_run(report.parent)

    def test_no_opt_in_and_api_failure_preserve_eda(self):
        from data import intelligence
        with patch.object(intelligence,'request_summary',side_effect=ValueError('API unavailable')) as mocked:
            result=self.wait(self.post('projects/demo/import',{'source':'sklearn','name':'iris','analyze':True}))
            mocked.assert_not_called()
            result=self.wait(self.post('projects/demo/intelligence',{'dataset_id':result['dataset_id'],'ai_analysis':True,'explanation':''}))
        self.assertEqual(result['ai_error'],'API unavailable')
        self.assertTrue(self.get('projects/demo')['dataset']['report'])
