import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from data.projects import ProjectStore
from eda_tool.loader import open_dataset
from eda_tool.row_views import collect_row_views
from eda_tool.workflows import run_eda
from eda_tool.artifacts import load_run
from eda_tool.html_report import HtmlReport
from eda_tool.profiler import ProfileConfig
from preprocessing import Recipe, Step, FittedRecipe
from splitting import SplitConfig
from workflows.project_preparation import ProjectPreparation
from etml.cli import main


class ProjectPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = ProjectStore(self.root/'projects').create('demo')
        self.frame = pd.DataFrame({'x':[np.nan]+list(range(1,60)), 'id':range(60), 'target':np.arange(60)*2.})
        self.source = self.root/'source.csv'
        self.frame.to_csv(self.source,index=False)
        self.dataset = self.project.workspace.import_files(self.source,dataset_id='data')
        self.service = ProjectPreparation(self.project,batch_size=7)

    def configure_recipe(self):
        self.service.configure('target','regression',excluded=['id'])
        self.service.save_recipe(Recipe((Step('fill_missing',('x',),{'strategy':'mean'}).approve(),)))

    def test_preview_finds_changes_in_later_batches(self):
        self.frame['x'] = np.arange(60, dtype=float)
        indexes = [12, 21, 34, 44, 52, 58]
        self.frame.loc[indexes, 'x'] = np.nan
        source = self.root/'later.csv'
        self.frame.to_csv(source,index=False)
        self.project.workspace.import_files(source,dataset_id='later')
        self.service = ProjectPreparation(self.project,dataset_id='later',batch_size=7)
        self.configure_recipe()
        _, before, after = self.service.fit_preview()
        self.assertEqual(list(before.index), indexes[:5])
        self.assertTrue(before.x.isna().all())
        self.assertFalse(after.x.isna().any())
        self.assertTrue(before.attrs['preview']['changed'])

    def test_unchanged_preview_is_explicit(self):
        self.service.configure('target','regression',excluded=['id'])
        self.service.save_recipe(Recipe(()))
        _, before, after = self.service.fit_preview()
        self.assertFalse(before.attrs['preview']['changed'])
        self.assertEqual(before.attrs['preview']['rows_examined'],60)
        pd.testing.assert_frame_equal(before,after)

    def test_exact_windows_cross_batch_boundaries(self):
        for total in (0,1,9,10,11,19,20,101):
            frame = pd.DataFrame({'value':range(total)})
            views = collect_row_views(open_dataset(frame),total,batch_size=7)
            count = min(10,total)
            for key,start in [('first',0),('middle',max(0,(total-count)//2)),('last',max(0,total-count))]:
                self.assertEqual(views[key]['value'].tolist(),list(range(start,start+count)))
                self.assertEqual(list(views[key].index),list(range(start+1,start+count+1)))

    def test_report_windows_persist_escape_and_survive_source_removal(self):
        frame = pd.DataFrame({'value':[f'<script>{i}</script>' for i in range(31)]})
        result = run_eda(frame,export_html=True,output_dir=self.root/'eda',
                         profile_config=ProfileConfig(sample_size=2,batch_size=7))
        self.addCleanup(result.close)
        saved = load_run(self.root/'eda')
        self.assertTrue(saved.manifest['row_previews_saved'])
        self.assertIsNone(saved.manifest['sample'])
        self.assertEqual(saved.profile['row_views'].data['middle'].value.tolist(),frame.value.iloc[10:20].tolist())
        html = HtmlReport(saved.profile,saved.charts).render()
        self.assertIn('First 10 rows',html)
        self.assertIn('Middle 10 rows',html)
        self.assertIn('Last 10 rows',html)
        self.assertNotIn('<script>',html)
        self.assertIn('&lt;script&gt;',html)
        self.assertIn('#rows-middle:checked~.rows-middle',html)
        view_file = self.root/'eda'/'row_views.json'
        view_file.write_text('{}')
        with self.assertRaises(ValueError):
            load_run(self.root/'eda')

    def test_no_html_does_not_persist_row_windows(self):
        result = run_eda(self.frame,output_dir=self.root/'plain')
        self.addCleanup(result.close)
        self.assertFalse(result.manifest['row_previews_saved'])
        self.assertFalse((self.root/'plain'/'row_views.json').exists())

    def test_processed_copy_and_split_refit_on_training_only(self):
        self.configure_recipe()
        fitted,before,after = self.service.fit_preview()
        self.assertEqual(len(before),1)
        self.assertEqual(after.iloc[0].x, self.frame.x.mean())
        processed = self.service.save_processed(fitted)
        path = self.dataset.directory/processed['path']/'data/part-00000.parquet'
        self.assertEqual(pd.read_parquet(path).columns.tolist(),['x','target'])
        result = self.service.split(SplitConfig('random',.6,.2,.2,seed=7))
        train = pd.read_parquet(result.directory/'splits/train/part-00000.parquet')
        actual_fit = FittedRecipe.load(result.directory/'preprocessing/fitted.json')
        self.assertAlmostEqual(actual_fit.states[0]['fills']['x'],train.x.mean())
        self.assertNotEqual(actual_fit.states[0]['fills']['x'],fitted.states[0]['fills']['x'])
        manifest = self.project.workspace.get_task_run('data',result.task_id,result.run_id)
        self.assertEqual(manifest.metadata['source_version'],'raw')
        self.assertFalse(manifest.metadata['processed_source_override'])
        restored = ProjectPreparation(self.project)
        self.assertEqual(restored.state()['split']['run_id'],result.run_id)
        self.assertGreaterEqual(len(restored.artifacts()),3)
        pd.testing.assert_frame_equal(pd.read_csv(self.dataset.raw_files[0]),pd.read_csv(self.source))

    def test_approval_target_and_exclusion_guardrails(self):
        self.service.configure('target','regression',excluded=['id'])
        for step in (Step('fill_missing',('x',),{'strategy':'mean'}),
                     Step('drop_columns',('target',)).approve(),Step('drop_columns',('id',)).approve()):
            with self.assertRaises(ValueError):
                self.service.save_recipe(Recipe((step,)))
        self.assertNotIn('recipe',self.service.state())

    def test_row_dropping_preview_keeps_original_indexes(self):
        self.service.configure('target','regression')
        self.service.save_recipe(Recipe((Step('drop_missing',('x',)).approve(),)))
        _,before,after = self.service.fit_preview()
        self.assertEqual(list(before.index),[0])
        self.assertEqual(list(after.index),[])

    def test_presplit_inspection_excludes_test_and_split_preserves_it(self):
        test = self.root/'test.csv'
        pd.DataFrame({'x':[999.,1000.],'id':[60,61]}).to_csv(test,index=False)
        dataset = self.project.workspace.import_presplit(self.source,test=test,dataset_id='presplit')
        service = ProjectPreparation(self.project,'presplit',batch_size=7)
        service.configure('target','regression',excluded=['id'])
        service.save_recipe(Recipe((Step('fill_missing',('x',),{'strategy':'mean'}).approve(),)))
        fitted,_,_ = service.fit_preview()
        saved = service.save_processed(fitted)
        self.assertEqual(saved['rows'],60)
        self.assertEqual(fitted.states[0]['fills']['x'],self.frame.x.mean())
        result = service.split(SplitConfig('random'),validation_fraction=.2)
        self.assertEqual(result.split_counts['test'],2)
        self.assertEqual(pd.read_parquet(result.directory/'prepared/test/part-00000.parquet').x.tolist(),[999.,1000.])

    def test_failed_split_retains_processed_and_retry_succeeds(self):
        self.configure_recipe()
        fitted,_,_ = self.service.fit_preview()
        self.service.save_processed(fitted)
        with self.assertRaises(ValueError):
            self.service.split(SplitConfig('stratified'))
        self.assertIn('processed',self.service.state())
        self.assertNotIn('split',self.service.state())
        self.service.split(SplitConfig('random'))
        self.assertIn('split',self.service.state())

    def test_new_recipe_preserves_old_artifacts(self):
        self.configure_recipe()
        fitted,_,_ = self.service.fit_preview()
        previous = self.service.save_processed(fitted)
        self.service.split(SplitConfig('random'))
        self.service.save_recipe(Recipe())
        self.assertNotIn('processed',self.service.state())
        self.assertNotIn('split',self.service.state())
        self.assertTrue((self.dataset.directory/previous['path']).exists())
        self.assertEqual(len(self.service.state()['history']),2)

    def test_cli_complete_flow_custom_split_and_resume(self):
        answers = ['1','3','2','2','1','3','1','1','1','5','80,10,5','80,10,10','1','42','2']
        output = io.StringIO()
        with (patch.dict(os.environ,ETML_PROJECTS_DIR=str(self.root/'projects')),
              patch('builtins.input',side_effect=answers), contextlib.redirect_stdout(output)):
            code = main(['demo','--continue'])
        self.assertEqual(code,0,output.getvalue())
        self.assertIn('No changed rows found; first source rows for reference:',output.getvalue())
        self.assertIn('totaling 100',output.getvalue())
        state = self.service.state()
        self.assertEqual(state['split']['config']['train'],.8)
        self.assertEqual(state['processed']['rows'],60)
        with (patch.dict(os.environ,ETML_PROJECTS_DIR=str(self.root/'projects')),
              patch('builtins.input',side_effect=AssertionError('no prompt')),contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(main(['demo','--status']),0)

    def test_cli_custom_recipe_and_edit_imputation(self):
        self.service.configure('target','regression',excluded=['id'])
        custom = self.root/'custom.json'
        Recipe((Step('fill_missing',('x',),{'strategy':'mean'}),)).save(custom)
        # Existing task: load custom, change mean to median, preview, save, defer split.
        answers = ['2',str(custom),'2','2','1','1','2']
        output = io.StringIO()
        with (patch.dict(os.environ,ETML_PROJECTS_DIR=str(self.root/'projects')),
              patch('builtins.input',side_effect=answers),contextlib.redirect_stdout(output)):
            code = main(['demo','--continue'])
        self.assertEqual(code,0,output.getvalue())
        self.assertEqual(self.service.recipe().active_steps[0].params['strategy'],'median')
        self.assertTrue(self.service.recipe().active_steps[0].approved)
        self.assertIn('processed',self.service.state())
        self.assertNotIn('split',self.service.state())

    def test_terminal_automatically_offers_preprocessing_after_import(self):
        output = io.StringIO()
        with (patch.dict(os.environ,ETML_PROJECTS_DIR=str(self.root/'projects')),
              patch('etml.project_cli.sys.stdin.isatty',return_value=True),
              patch('builtins.input',side_effect=['2']),contextlib.redirect_stdout(output)):
            code = main(['new-project','--upload',str(self.source),'--no-eda'])
        self.assertEqual(code,0,output.getvalue())
        self.assertIn('Review preprocessing now',output.getvalue())

    def test_resume_saved_recipe_then_save_processed_without_reimport(self):
        self.configure_recipe()
        with (patch.dict(os.environ,ETML_PROJECTS_DIR=str(self.root/'projects')),
              patch('builtins.input',side_effect=['1','2']),contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(main(['demo','--continue']),0)
        self.assertEqual(len(self.project.datasets()),1)
        self.assertIn('processed',self.service.state())

    def test_generated_suggestions_can_be_changed_in_wizard(self):
        self.service.configure('target','regression',excluded=['id'])
        answers = ['1','2','2','1','1','2']
        output = io.StringIO()
        with (patch.dict(os.environ,ETML_PROJECTS_DIR=str(self.root/'projects')),
              patch('builtins.input',side_effect=answers),contextlib.redirect_stdout(output)):
            self.assertEqual(main(['demo','--continue']),0,output.getvalue())
        self.assertEqual(self.service.recipe().active_steps[0].params,{'strategy':'median'})
        self.assertIn('processed',self.service.state())


if __name__=='__main__':
    unittest.main()
