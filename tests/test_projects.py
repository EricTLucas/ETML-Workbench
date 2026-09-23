import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.projects import ProjectStore
from etml.cli import main


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = self.root / 'projects'
        self.csv = self.root / 'input.csv'
        self.csv.write_text('age,label\n10,a\n20,b\n30,a\n40,b\n50,a\n60,b\n')
        self.env = patch.dict(os.environ, {'ETML_PROJECTS_DIR':str(self.store)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def invoke(self, args, answers=()):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch('builtins.input', side_effect=list(answers)), patch('etml.project_cli.sys.stdin.isatty',return_value=False):
            code = main(args)
        return code, output.getvalue()

    def test_new_project_guided_upload(self):
        with patch('etml.project_cli.webbrowser.open', return_value=True) as browser:
            code, output = self.invoke([], ['1','first','1',str(self.csv)])
        self.assertEqual(code, 0, output)
        project = ProjectStore(self.store).get('first')
        dataset = project.current_dataset()
        self.assertEqual(dataset.raw_files[0].read_bytes(), self.csv.read_bytes())
        self.assertEqual(len(list(dataset.profiles_dir.glob('overview-*/report.html'))), 1)
        browser.assert_called_once()
        self.assertTrue(browser.call_args.args[0].startswith('file:'))
        self.assertIn('Data preview', output)
        self.assertNotIn('"steps"', output)

    def test_no_open_generates_eda_and_existing_does_not_repeat(self):
        with patch('etml.project_cli.webbrowser.open') as browser:
            code, output = self.invoke(['demo','-upload',str(self.csv),'--no-open'])
            self.assertEqual(code, 0, output)
            browser.assert_not_called()
        before = list(ProjectStore(self.store).get('demo').current_dataset().profiles_dir.iterdir())
        code, output = self.invoke([], ['2','1'])
        self.assertEqual(code, 0, output)
        self.assertIn('Current dataset:', output)
        self.assertIn('--continue', output)
        self.assertEqual(before, list(ProjectStore(self.store).get('demo').current_dataset().profiles_dir.iterdir()))

    def test_no_eda_and_retry(self):
        code, output = self.invoke(['demo','--upload',str(self.csv),'--no-eda'])
        self.assertEqual(code, 0, output)
        dataset = ProjectStore(self.store).get('demo').current_dataset()
        self.assertFalse(list(dataset.profiles_dir.iterdir()))
        with patch('etml.project_cli.webbrowser.open', return_value=False):
            code, output = self.invoke(['demo','--eda'])
        self.assertEqual(code, 0, output)
        self.assertIn('Could not open', output)

    def test_project_isolation_and_multiple_imports(self):
        for name in ('first','second','first'):
            code, output = self.invoke([name,'--upload',str(self.csv),'--no-eda'])
            self.assertEqual(code, 0, output)
        store = ProjectStore(self.store)
        self.assertEqual(len(store.get('first').datasets()), 2)
        self.assertEqual(len(store.get('second').datasets()), 1)
        code, output = self.invoke(['projects'])
        self.assertIn('first', output)
        self.assertIn('second', output)

    def test_library_and_guided_sklearn(self):
        code, output = self.invoke(['sklearn'])
        self.assertEqual(code, 0)
        self.assertIn('--sklearn iris', output)
        code, output = self.invoke(['--projects-dir',str(self.store),'--no-eda'],
                                   ['1','flowers','2','1','1'])
        self.assertEqual(code, 0, output)
        dataset = ProjectStore(self.store).get('flowers').current_dataset()
        self.assertEqual(dataset.manifest.provenance['name'], 'iris')

    def test_presplit_preview_and_eda_use_training_only(self):
        test = self.root / 'test.csv'
        test.write_text('age\n999\n')
        code, output = self.invoke(['split','--upload',str(self.csv),'--test',str(test),'--no-open'])
        self.assertEqual(code, 0, output)
        self.assertNotIn('999', output)
        dataset = ProjectStore(self.store).get('split').current_dataset()
        from eda_tool.artifacts import load_run
        profile = load_run(next(dataset.profiles_dir.iterdir())).profile
        self.assertEqual(profile['summary'].data['rows'], 6)

    def test_failed_eda_preserves_dataset(self):
        with patch('eda_tool.workflows.run_eda', side_effect=ValueError('bad schema')):
            code, output = self.invoke(['demo','--upload',str(self.csv)])
        self.assertEqual(code, 3)
        self.assertIn('Your data is saved', output)
        self.assertIn('--eda', output)
        self.assertIsNotNone(ProjectStore(self.store).get('demo').current_dataset())

    def test_invalid_names_and_duplicate_creation(self):
        store = ProjectStore(self.store)
        for name in ('../outside','models','CON','a/b'):
            with self.assertRaises(ValueError):
                store.create(name)
        store.create('safe')
        with self.assertRaises(ValueError):
            store.create('safe')

    def test_cancel_empty_listing_and_legacy_dispatch(self):
        code, output = self.invoke([], ['2'])
        self.assertEqual(code, 0)
        self.assertIn('No projects', output)
        code, output = self.invoke([], ['q'])
        self.assertEqual(code, 130)
        code, output = self.invoke(['datasets','list','--workspace',str(self.root/'legacy'),'--json'])
        self.assertEqual(code, 0)
        self.assertIn('"datasets"', output)

    def test_import_failure_and_eof_are_recoverable(self):
        code, output = self.invoke(['demo','--upload',str(self.root/'absent.csv')])
        self.assertEqual(code, 2)
        self.assertIsNone(ProjectStore(self.store).get('demo').current_dataset())
        with patch('builtins.input', side_effect=EOFError), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 2)

    def test_wizard_flags_and_provider_options(self):
        with patch('etml.project_cli.webbrowser.open') as browser:
            code, output = self.invoke(['--no-open'], ['1','quiet','1',str(self.csv)])
        self.assertEqual(code, 0, output)
        browser.assert_not_called()
        with patch('data.sources.import_huggingface') as provider:
            provider.return_value = ProjectStore(self.store).get('quiet').current_dataset()
            code, output = self.invoke(['hub','--huggingface','owner/data','--config','table',
                '--split','validation','--revision','abc','--columns','age','--no-eda'])
        self.assertEqual(code, 0, output)
        self.assertEqual(provider.call_args.kwargs['config'], 'table')
        self.assertEqual(provider.call_args.kwargs['split'], 'validation')
        self.assertEqual(provider.call_args.kwargs['revision'], 'abc')
        self.assertEqual(provider.call_args.kwargs['columns'], ['age'])


if __name__ == '__main__':
    unittest.main()
