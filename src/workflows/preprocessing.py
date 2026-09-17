from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import uuid

from data import DatasetWorkspace
from data.manifest import resolve_inside, utc_now, validate_name, write_json
from eda_tool import run_eda
from preprocessing import (
    Recipe, FittedRecipe, ExecutionResult, PreviewResult,
    suggest_recipe, validate_recipe, preview_recipe, execute_recipe,
)
from preprocessing.recipe import fingerprint, json_copy


@dataclass(frozen=True)
class PreprocessingProposal:
    dataset_id: str
    source_version: str
    source_fingerprint: str
    recipe: Recipe
    loader_options: dict
    profile_directory: Path
    recipe_path: Path


@dataclass(frozen=True)
class WorkflowResult:
    execution: ExecutionResult
    profile_directory: Path | None = None
    report_path: Path | None = None
    profile_error: str | None = None


class PreprocessingWorkflow:
    def __init__(self, workspace: DatasetWorkspace | str | Path = 'datasets'):
        self.workspace = workspace if isinstance(workspace, DatasetWorkspace) else DatasetWorkspace(workspace)

    def _source(self, dataset_id, source_version, verify=True):
        dataset = self.workspace.get(dataset_id, verify=verify and source_version == 'raw')
        if source_version == 'raw':
            records = dataset.manifest.files
            paths = dataset.raw_files
        else:
            manifest = self.workspace.get_version(dataset_id, source_version, verify=verify)
            root = resolve_inside(dataset.directory, 'processed/'+source_version)
            records = manifest.files
            paths = tuple(resolve_inside(root, item.path) for item in records)
        identity = fingerprint({'dataset_id': dataset_id, 'source_version': source_version,
                                'files': [{'path': r.path, 'bytes': r.size_bytes, 'sha256': r.sha256}
                                          for r in records]})
        return dataset, list(paths), identity

    def _check_proposal(self, proposal, verify=True):
        if not isinstance(proposal, PreprocessingProposal):
            raise TypeError('Expected a PreprocessingProposal')
        dataset, source, identity = self._source(proposal.dataset_id, proposal.source_version, verify)
        if identity != proposal.source_fingerprint:
            raise ValueError('Source differs from the proposal; prepare a new proposal')
        return dataset, source

    def _save_proposal(self, dataset, source_version, identity, recipe, loader_options, profile_directory):
        recipe = Recipe.from_dict(recipe.to_dict())
        recipe_path = dataset.recipes_dir / ('recipe-'+uuid.uuid4().hex+'.json')
        relative_profile = profile_directory.relative_to(dataset.directory).as_posix()
        payload = {'format_version': 1, 'kind': 'preprocessing_proposal', 'created_at': utc_now(),
                   'dataset_id': dataset.dataset_id, 'source_version': source_version,
                   'source_fingerprint': identity, 'loader_options': json_copy(loader_options),
                   'profile_directory': relative_profile, 'recipe': recipe.to_dict()}
        write_json(recipe_path, payload)
        return PreprocessingProposal(dataset.dataset_id, source_version, identity, recipe,
                                     json_copy(loader_options), profile_directory, recipe_path)

    def import_files(self, paths, *, name=None, dataset_id=None, max_bytes=None, **prepare_options):
        dataset = self.workspace.import_files(paths, name=name, dataset_id=dataset_id, max_bytes=max_bytes)
        return self._prepare_import(dataset.dataset_id, prepare_options)

    def import_upload(self, stream, *, filename, name=None, dataset_id=None, max_bytes=None, **prepare_options):
        dataset = self.workspace.import_upload(stream, filename=filename, name=name,
                                                dataset_id=dataset_id, max_bytes=max_bytes)
        return self._prepare_import(dataset.dataset_id, prepare_options)

    def _prepare_import(self, dataset_id, options):
        try:
            return self.prepare(dataset_id, **options)
        except Exception as exc:
            raise PreparationError(dataset_id, exc) from exc

    def prepare(self, dataset_id, *, source_version='raw', profile_config=None,
                visualizer_config=None, loader_options=None, suggestion_config=None,
                protected_columns=(), generate_summary=False, export_html=False,
                save_sample=False, verify_source=True, progress=None):
        options = json_copy(loader_options or {})
        dataset, source, identity = self._source(dataset_id, source_version, verify_source)
        profile_directory = dataset.profiles_dir / (source_version+'-'+uuid.uuid4().hex)
        result = run_eda(source, profile_config=profile_config, visualizer_config=visualizer_config,
                         loader_options=options, generate_summary=generate_summary,
                         export_html=export_html, save_sample=save_sample,
                         output_dir=profile_directory, title=f'{dataset.manifest.name} - {source_version}',
                         progress=progress)
        try:
            recipe = suggest_recipe(result.profile, config=suggestion_config,
                                     protected_columns=protected_columns)
            validate_recipe(recipe, result.profile['columns'].data, require_approved=False)
            _, _, current = self._source(dataset_id, source_version, verify_source)
            if current != identity:
                raise ValueError('Source changed while profiling; prepare again')
            return self._save_proposal(dataset, source_version, identity, recipe, options, profile_directory)
        finally:
            result.close()

    def load_proposal(self, dataset_id, recipe_filename, *, verify_source=True):
        dataset = self.workspace.get(dataset_id)
        validate_name(recipe_filename)
        path = resolve_inside(dataset.recipes_dir, recipe_filename)
        payload = json.loads(path.read_text(encoding='utf-8'))
        if (type(payload.get('format_version')) is not int or payload['format_version'] != 1
                or payload.get('kind') != 'preprocessing_proposal' or payload.get('dataset_id') != dataset_id):
            raise ValueError('Unsupported or mismatched proposal')
        profile_relative = payload['profile_directory']
        if not profile_relative.startswith('profiles/'):
            raise ValueError('Proposal profile must be inside profiles/')
        proposal = PreprocessingProposal(dataset_id, payload['source_version'], payload['source_fingerprint'],
                                         Recipe.from_dict(payload['recipe']), json_copy(payload['loader_options']),
                                         resolve_inside(dataset.directory, profile_relative), path)
        self._check_proposal(proposal, verify_source)
        return proposal

    def save_review(self, proposal, recipe: Recipe, *, verify_source=True):
        dataset, source = self._check_proposal(proposal, verify_source)
        from eda_tool.loader import open_dataset
        validate_recipe(recipe, open_dataset(source, **proposal.loader_options).schema(), require_approved=False)
        # Preserve the original proposal; callers explicitly approve/reject/edit steps.
        return self._save_proposal(dataset, proposal.source_version, proposal.source_fingerprint,
                                   recipe, proposal.loader_options, proposal.profile_directory)

    def preview(self, proposal, *, recipe=None, fitted=None, max_rows=200, verify_source=True) -> PreviewResult:
        _, source = self._check_proposal(proposal, verify_source)
        return preview_recipe(source, proposal.recipe if recipe is None else recipe, fitted=fitted,
                               max_rows=max_rows, loader_options=proposal.loader_options)

    def execute(self, proposal, *, fitted: FittedRecipe | None = None, fit_source=None,
                fit_loader_options=None, batch_size=50_000, verify_source=True,
                profile_after=True, profile_config=None, visualizer_config=None,
                generate_summary=False, export_html=False, save_sample=False,
                progress=None, profile_progress=None) -> WorkflowResult:
        self._check_proposal(proposal, verify_source)
        # Approval is enforced by execute_recipe; there is no implicit approve_all.
        execution = execute_recipe(self.workspace, proposal.dataset_id, proposal.recipe,
                                    source_version=proposal.source_version, fitted=fitted, fit_source=fit_source,
                                    fit_loader_options=fit_loader_options, batch_size=batch_size,
                                    loader_options=proposal.loader_options, verify_source=verify_source,
                                    progress=progress)
        if not profile_after:
            return WorkflowResult(execution)
        profile_directory = None
        result = None
        try:
            dataset, source, _ = self._source(proposal.dataset_id, execution.version, verify=False)
            profile_directory = dataset.profiles_dir / (execution.version+'-'+uuid.uuid4().hex)
            # Processed Parquet carries its own schema; raw CSV loader overrides do not apply.
            result = run_eda(source, profile_config=profile_config, visualizer_config=visualizer_config,
                             generate_summary=generate_summary, export_html=export_html,
                             save_sample=save_sample, output_dir=profile_directory,
                             title=f'{dataset.manifest.name} - {execution.version}', progress=profile_progress)
            return WorkflowResult(execution, result.output_dir, result.html_path)
        except Exception as exc:
            # The processed version is already committed. Report this separately,
            # rather than suggesting that execution failed and should be repeated.
            return WorkflowResult(execution, profile_error=f'{type(exc).__name__}: {exc}')
        finally:
            if result is not None:
                result.close()

    def load_fitted(self, dataset_id, version, *, verify=True) -> FittedRecipe:
        self.workspace.get_version(dataset_id, version, verify=verify)
        dataset = self.workspace.get(dataset_id)
        path = resolve_inside(dataset.directory, f'processed/{version}/recipe.json')
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or 'fitted' not in payload:
            raise ValueError('This processed version does not contain a fitted preprocessor')
        return FittedRecipe.from_dict(payload['fitted'])


class PreparationError(RuntimeError):
    def __init__(self, dataset_id, cause):
        self.dataset_id = dataset_id
        super().__init__(f'Dataset {dataset_id!r} was imported and its raw files are preserved, '
                         f'but preparation failed: {cause}. Retry prepare(dataset_id) after correcting the issue.')
