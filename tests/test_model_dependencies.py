import unittest
from unittest.mock import patch, Mock
from models.dependencies import ensure_dependencies, missing_requirements
from importlib.metadata import PackageNotFoundError

class DependenciesTests(unittest.TestCase):
    def test_installed_package_skips_pip(self):
        with patch('models.dependencies.version',return_value='3.0'), patch('models.dependencies.subprocess.run') as run:
            ensure_dependencies('xgboost:boosted_trees')
            run.assert_not_called()

    def test_missing_package_installs_catalog_requirement(self):
        events=[]
        with patch('models.dependencies.missing_requirements',side_effect=[['xgboost>=2.0'],[]]), patch('models.dependencies.subprocess.run',return_value=Mock(returncode=0)) as run:
            ensure_dependencies('xgboost:boosted_trees',events.append)
            self.assertIn('xgboost>=2.0',run.call_args.args[0])
            self.assertEqual(events[0]['stage'],'dependencies')

    def test_failed_install_is_actionable(self):
        with patch('models.dependencies.missing_requirements',return_value=['xgboost>=2.0']), patch('models.dependencies.subprocess.run',return_value=Mock(returncode=1,stderr='offline')):
            with self.assertRaisesRegex(ValueError,'installation failed'): ensure_dependencies('xgboost:boosted_trees')

    def test_missing_and_incompatible_are_distinguished(self):
        with patch('models.dependencies.version',side_effect=PackageNotFoundError):
            self.assertEqual(missing_requirements('xgboost'),['xgboost>=2.0'])
        with patch('models.dependencies.version',return_value='1.0'):
            with self.assertRaisesRegex(ValueError,'incompatible'): missing_requirements('xgboost')
