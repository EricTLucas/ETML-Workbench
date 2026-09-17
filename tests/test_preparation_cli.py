from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import pandas as pd
from data import DatasetWorkspace
from etml.cli import main
from preprocessing import Recipe, Step


class PreparationCLITests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.ws=DatasetWorkspace(self.root/'datasets')
        path=self.root/'input.csv'
        pd.DataFrame({'x':range(60),'Survived':[i%2 for i in range(60)]}).to_csv(path,index=False)
        self.ws.import_files(path,dataset_id='titanic')

    def cli(self,*args):
        out,err=io.StringIO(),io.StringIO()
        with redirect_stdout(out),redirect_stderr(err):
            code=main(['tasks',*args,'--workspace',str(self.ws.root),'--json'])
        return code,json.loads(out.getvalue()) if out.getvalue() else None,err.getvalue()

    def create(self):
        code,_,err=self.cli('create','titanic','survival','--target','Survived','--type','classification')
        self.assertEqual(code,0,err)

    def test_create_prepare_inspect(self):
        self.create()
        code,config,_=self.cli('show','titanic','survival')
        self.assertEqual(config['target'],'Survived')
        code,result,err=self.cli('prepare','titanic','survival','--batch-size','7')
        self.assertEqual(code,0,err)
        self.assertIn('fitting',err)
        code,manifest,err=self.cli('inspect-run','titanic','survival',result['run_id'],'--verify')
        self.assertEqual(code,0,err)
        self.assertEqual(manifest['status'],'complete')

    def test_unapproved_recipe_is_rejected(self):
        self.create()
        recipe=self.root/'recipe.json'
        Recipe((Step('fill_missing',('x',),{'value':0}),)).save(recipe)
        code,_,err=self.cli('prepare','titanic','survival','--recipe-file',str(recipe))
        self.assertEqual(code,2)
        self.assertIn('approval',err)

    def test_bad_split_and_quiet(self):
        self.create()
        code,_,_=self.cli('prepare','titanic','survival','--train','1')
        self.assertEqual(code,2)
        code,_,err=self.cli('prepare','titanic','survival','--quiet')
        self.assertEqual(code,0,err)
        self.assertEqual(err,'')


if __name__=='__main__': unittest.main()
