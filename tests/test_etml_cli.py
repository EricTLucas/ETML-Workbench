from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from etml.cli import main
from data import DatasetWorkspace
from preprocessing import FittedRecipe
from workflows import WorkflowResult


class ETMLCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.storage = self.root/'datasets'
        self.input = self.root/'customers.csv'
        pd.DataFrame({'age': [10., None, 30., 40.], 'group': ['A','B','A','B']}).to_csv(self.input, index=False)

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main([*args, '--workspace', str(self.storage), '--json'])
        return code, json.loads(out.getvalue()) if out.getvalue() else None, err.getvalue()

    def prepare(self):
        code, _, error = self.cli('datasets','import',str(self.input),'--id','demo')
        self.assertEqual(code,0,error)
        code, proposal, error = self.cli('preprocess','prepare','demo','--role','age=numeric')
        self.assertEqual(code,0,error)
        return proposal

    def approve(self, proposal):
        code, reviewed, _ = self.cli('preprocess','review','demo','--recipe',proposal['recipe'],'--approve-all')
        self.assertEqual(code,0)
        return reviewed

    def test_import_list_show_and_preserve_bytes(self):
        code, payload, _ = self.cli('datasets','import',str(self.input),'--id','demo')
        self.assertEqual(code,0)
        self.assertEqual(Path(payload['raw_files'][0]).read_bytes(),self.input.read_bytes())
        code, listed, _ = self.cli('datasets','list')
        self.assertEqual(listed['datasets'][0]['dataset_id'],'demo')
        code, shown, _ = self.cli('datasets','show','demo','--verify')
        self.assertTrue(shown['verified'])
        self.assertEqual(shown['versions'],[])

    def test_prepare_preview_review_execute_and_export(self):
        proposal=self.prepare()
        self.assertEqual(proposal['steps'][0]['status'],'proposed')
        code, preview, _ = self.cli('preprocess','preview','demo','--recipe',proposal['recipe'])
        self.assertEqual(code,0)
        self.assertEqual(preview['changes']['missing_after']['age'],0)
        self.assertEqual(DatasetWorkspace(self.storage).list_versions('demo'),[])
        reviewed=self.approve(proposal)
        code, result, error=self.cli('preprocess','execute','demo','--recipe',reviewed['recipe'],
                                    '--fit-on-source','--batch-size','2')
        self.assertEqual(code,0,error)
        self.assertEqual(result['version'],'v1')
        self.assertTrue(Path(result['profile_directory']).is_dir())
        output=pd.read_parquet(Path(result['directory'])/'data'/'part-00000.parquet')
        self.assertAlmostEqual(output.age.iloc[1],80/3)
        fit=self.root/'fitted.json'
        code,_,_=self.cli('preprocess','export-fitted','demo','v1','--output',str(fit))
        self.assertEqual(code,0)
        restored=FittedRecipe.load(fit)
        self.assertAlmostEqual(restored.apply(pd.DataFrame({'age':[None],'group':['A']})).age.iloc[0],80/3)
        code, result, error=self.cli('preprocess','execute','demo','--recipe',reviewed['recipe'],
                                    '--fitted',str(fit),'--no-profile')
        self.assertEqual(code,0,error)
        self.assertEqual(result['version'],'v2')

    def test_unapproved_and_missing_fitting_source_block_execution(self):
        proposal=self.prepare()
        code,_,error=self.cli('preprocess','execute','demo','--recipe',proposal['recipe'],'--fit-on-source')
        self.assertEqual(code,2)
        self.assertIn('approval',error)
        reviewed=self.approve(proposal)
        code,_,error=self.cli('preprocess','execute','demo','--recipe',reviewed['recipe'])
        self.assertEqual(code,2)
        self.assertIn('fit_source',error)
        self.assertEqual(DatasetWorkspace(self.storage).list_versions('demo'),[])

    def test_edited_approved_parameters_need_reapproval(self):
        proposal=self.approve(self.prepare())
        exported=self.root/'recipe.json'
        code,_,_=self.cli('preprocess','show','demo','--recipe',proposal['recipe'],'--output',str(exported))
        self.assertEqual(code,0)
        payload=json.loads(exported.read_text())
        payload['steps'][0]['params']['strategy']='median'
        exported.write_text(json.dumps(payload))
        code,changed,_=self.cli('preprocess','review','demo','--recipe',proposal['recipe'],'--from-file',str(exported))
        self.assertEqual(code,0)
        code,_,_=self.cli('preprocess','execute','demo','--recipe',changed['recipe'],'--fit-on-source')
        self.assertEqual(code,2)
        approved=self.approve(changed)
        code,_,_=self.cli('preprocess','execute','demo','--recipe',approved['recipe'],'--fit-on-source','--no-profile')
        self.assertEqual(code,0)

    def test_review_rejection_and_unknown_step(self):
        proposal=self.prepare()
        code,_,_=self.cli('preprocess','review','demo','--recipe',proposal['recipe'],'--approve','unknown')
        self.assertEqual(code,2)
        code,reviewed,_=self.cli('preprocess','review','demo','--recipe',proposal['recipe'],
                                '--reject',proposal['steps'][0]['step_id'])
        self.assertEqual(code,0)
        self.assertEqual(reviewed['steps'][0]['status'],'rejected')
        code,_,_=self.cli('preprocess','execute','demo','--recipe',reviewed['recipe'],'--no-profile')
        self.assertEqual(code,0)

    def test_explicit_training_file_controls_fills(self):
        proposal=self.approve(self.prepare())
        train=self.root/'train.csv'
        pd.DataFrame({'age':[100.,200.],'group':['A','B']}).to_csv(train,index=False)
        code,result,error=self.cli('preprocess','execute','demo','--recipe',proposal['recipe'],
                                   '--fit-data',str(train),'--no-profile')
        self.assertEqual(code,0,error)
        output=pd.read_parquet(Path(result['directory'])/'data'/'part-00000.parquet')
        self.assertEqual(output.age.iloc[1],150.)

    def test_errors_preserve_existing_files_and_source(self):
        proposal=self.prepare()
        code,_,error=self.cli('datasets','import',str(self.input),'--id','demo')
        self.assertEqual(code,2)
        output=self.root/'existing.json'; output.write_text('keep')
        code,_,_=self.cli('preprocess','show','demo','--recipe',proposal['recipe'],'--output',str(output))
        self.assertEqual(code,2)
        self.assertEqual(output.read_text(),'keep')
        code,_,_=self.cli('preprocess','show','demo','--recipe','../outside.json')
        self.assertEqual(code,2)
        self.assertTrue(self.input.exists())

    def test_progress_is_stderr_and_quiet_suppresses_it(self):
        self.cli('datasets','import',str(self.input),'--id','demo')
        code,_,err=self.cli('preprocess','prepare','demo')
        self.assertEqual(code,0)
        self.assertIn('Computing',err)
        code,_,err=self.cli('preprocess','prepare','demo','--quiet')
        self.assertEqual(code,0)
        self.assertEqual(err,'')

    def test_profile_failure_is_distinct_from_execution_failure(self):
        proposal=self.approve(self.prepare())
        from workflows import PreprocessingWorkflow
        original=PreprocessingWorkflow.execute
        def wrapped(self,*args,**kwargs):
            kwargs['profile_after']=False
            result=original(self,*args,**kwargs)
            return WorkflowResult(result.execution,profile_error='Simulated profile failure')
        with patch.object(PreprocessingWorkflow,'execute',wrapped):
            code,result,_=self.cli('preprocess','execute','demo','--recipe',proposal['recipe'],'--fit-on-source')
        self.assertEqual(code,3)
        self.assertEqual(result['status'],'processed_profile_failed')
        self.assertEqual(DatasetWorkspace(self.storage).list_versions('demo'),['v1'])

    def test_eda_delegation_and_catalog(self):
        with patch('eda_tool.cli.main',return_value=0) as delegated:
            self.assertEqual(main(['eda','charts','--json']),0)
            delegated.assert_called_once_with(['eda','charts','--json'])
        code,result,_=self.cli('preprocess','transforms')
        self.assertEqual(code,0)
        self.assertIn('fill_missing',result['transforms'])

    def test_module_entrypoint(self):
        process=subprocess.run([sys.executable,'-m','etml','--version'],capture_output=True,text=True,
                               env=os.environ.copy(),check=False)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertIn('ETML Workbench',process.stdout)

    @unittest.skipUnless(os.name == 'nt', 'Windows sharing-error retry')
    def test_windows_publish_retry_does_not_overwrite(self):
        from data.workspace import _publish_directory
        source, destination = self.root/'staging', self.root/'published'
        source.mkdir()
        error=PermissionError('temporary sharing violation')
        error.winerror=32
        original=Path.rename
        calls=[]
        def rename(path,target):
            calls.append(path)
            if len(calls)==1:
                raise error
            return original(path,target)
        with patch.object(Path,'rename',rename), patch('data.workspace.time.sleep'):
            _publish_directory(source,destination)
        self.assertEqual(len(calls),2)
        source.mkdir()
        with self.assertRaises(FileExistsError):
            _publish_directory(source,destination)
        self.assertTrue(source.exists())


if __name__ == '__main__':
    unittest.main()
