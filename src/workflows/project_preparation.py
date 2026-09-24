"""Persistent project preparation. Exploration copies never become training-fit input."""
from pathlib import Path
import json
import uuid
import pandas as pd
from data.manifest import write_json, resolve_inside, sha256_file
from eda_tool.loader import open_dataset
from preprocessing import Recipe
from preprocessing.executor import fit_batches, write_parquet_batches
from preprocessing.transforms.base import batches
from tasks import TaskConfig, validate_task
from splitting import SplitConfig
from .preparation import prepare_task
from .preprocessing import PreprocessingWorkflow


class ProjectPreparation:
    def __init__(self, project, dataset_id=None, *, batch_size=50_000):
        self.project, self.workspace = project, project.workspace
        dataset = self.workspace.get(dataset_id, verify=True) if dataset_id else project.current_dataset()
        if dataset is None:
            raise ValueError('Import data before preprocessing')
        self.dataset = self.workspace.get(dataset.dataset_id, verify=True)
        self.batch_size = batch_size
        self.path = self.dataset.directory/'project-preparation.json'

    def state(self):
        if not self.path.exists():
            return {}
        state = json.loads(self.path.read_text(encoding='utf-8'))
        if state.get('format_version') != 1 or state.get('dataset_id') != self.dataset.dataset_id:
            raise ValueError('Invalid project preparation state')
        if state['source_sha256'] != sha256_file(self.dataset.directory/'manifest.json'):
            raise ValueError('Source metadata changed; create a new preparation')
        return state

    def _save(self, state):
        write_json(self.path, state, overwrite=True)

    def source(self):
        if self.dataset.manifest.split_files:
            return [resolve_inside(self.dataset.directory, self.dataset.manifest.split_files['train'])]
        return list(self.dataset.raw_files)

    def schema(self):
        return open_dataset(self.source()).schema()

    def configure(self, target, task_type, excluded=(), missing_target='error'):
        previous = self.state()
        task = TaskConfig('task-'+uuid.uuid4().hex[:12], target, task_type, tuple(excluded), missing_target)
        validate_task(task, self.schema(), SplitConfig(), Recipe())
        state = {'format_version':1, 'dataset_id':self.dataset.dataset_id,
                 'source_sha256':sha256_file(self.dataset.directory/'manifest.json'),
                 'task':task.to_dict(), 'history':previous.get('history',[])}
        self._save(state)
        return task

    def task(self):
        state = self.state()
        if 'task' not in state:
            raise ValueError('Choose the target and task type first')
        return TaskConfig.from_dict(state['task'])

    def suggest(self):
        task = self.task()
        from eda_tool.profiler import ProfileConfig
        proposal = PreprocessingWorkflow(self.workspace).prepare(self.dataset.dataset_id,
            protected_columns=[task.target,*task.excluded_columns],
            profile_config=ProfileConfig(batch_size=self.batch_size))
        return proposal.recipe

    def save_recipe(self, recipe):
        task = self.task()
        validate_task(task, self.schema(), SplitConfig(), recipe)
        # validate_task also enforces approval for every enabled step.
        relative = 'recipes/project-'+uuid.uuid4().hex+'.json'
        recipe.save(resolve_inside(self.dataset.directory,relative))
        state = self.state()
        state['recipe'] = relative
        state.pop('processed',None)
        state.pop('split',None)
        self._save(state)

    def recipe(self):
        relative = self.state().get('recipe')
        if not relative:
            raise ValueError('Review and save a recipe first')
        return Recipe.load(resolve_inside(self.dataset.directory,relative))

    def fit_preview(self):
        task, recipe = self.task(), self.recipe()
        features = validate_task(task,self.schema(),SplitConfig(),recipe)
        dataset = open_dataset(self.source())
        def factory():
            with batches(lambda:dataset.iter_batches(batch_size=self.batch_size)) as iterator:
                for frame in iterator:
                    yield frame.loc[:,list(features)]
        fitted = fit_batches(factory,features,recipe,batch_size=self.batch_size,
                            fit_label='Exploration only: source/training-upload features before final split')
        # Search in bounded batches; keep at most five changed source rows.
        # Source ordinals must remain global even when a loader resets indexes.
        before_parts, after_parts = [], []
        first_before = first_after = None
        offset = found = 0
        type_columns = {c for step in recipe.active_steps if step.operation == 'convert_type'
                        for c in step.columns}
        type_changes = {}
        with batches(lambda:dataset.iter_batches(batch_size=self.batch_size)) as iterator:
            for frame in iterator:
                before = frame.loc[:,list(features)].copy()
                before.index = range(offset, offset+len(before))
                offset += len(before)
                after = fitted.apply(before)
                if first_before is None:
                    first_before = before.head(5).copy()
                    first_after = after.loc[after.index.intersection(first_before.index)].copy()
                kept = before.index.intersection(after.index)
                changed = pd.Series(~before.index.isin(after.index), index=before.index)
                if list(before.columns) != list(after.columns):
                    changed[:] = True
                elif len(kept):
                    left, right = before.loc[kept], after.loc[kept]
                    equal = left.astype(object).eq(right.astype(object)).fillna(False)
                    equal |= left.isna() & right.isna()
                    changed.loc[kept] = ~equal.all(axis=1)
                    # Explicit type conversions are changes even if values compare equal.
                    for column in type_columns.intersection(before.columns):
                        if before[column].dtype != after[column].dtype:
                            changed.loc[kept] = True
                            type_changes[column] = {'before':str(before[column].dtype),
                                                    'after':str(after[column].dtype)}
                selected = changed[changed].index[:5-found]
                if len(selected):
                    before_parts.append(before.loc[selected].copy())
                    after_parts.append(after.loc[after.index.intersection(selected)].copy())
                    found += len(selected)
                if found == 5:
                    break
        if first_before is None:
            first_before = pd.DataFrame(columns=list(features))
            first_after = fitted.apply(first_before)
        before = pd.concat(before_parts) if found else first_before
        after = pd.concat(after_parts) if found else first_after
        before.attrs['preview'] = {
            'changed': bool(found), 'rows_examined': offset,
            'removed_rows': [int(i)+1 for i in before.index.difference(after.index)],
            'type_changes': type_changes,
            'removed_columns': [c for c in before.columns if c not in after.columns],
            'added_columns': [c for c in after.columns if c not in before.columns],
        }
        return fitted, before, after

    def save_processed(self, fitted):
        task, recipe = self.task(), self.recipe()
        features = validate_task(task,self.schema(),SplitConfig(),recipe)
        if fitted.recipe_fingerprint != recipe.fingerprint or tuple(fitted.input_columns) != tuple(features):
            raise ValueError('Preview fit differs from the reviewed recipe')
        self.workspace.get(self.dataset.dataset_id,verify=True)
        dataset = open_dataset(self.source())
        def factory():
            with batches(lambda:dataset.iter_batches(batch_size=self.batch_size)) as iterator:
                for frame in iterator:
                    output = fitted.apply(frame.loc[:,list(features)])
                    output[task.target] = frame.loc[output.index,task.target]
                    yield output
        payload = {'recipe':recipe.to_dict(),'fitted':fitted.to_dict(),'task':task.to_dict(),
                   'purpose':'exploration_only',
                   'fitting_scope':'source features; train upload only for presplit data',
                   'training_policy':'split raw rows, then refit recipe on train only'}
        with self.workspace.processed_version(self.dataset.dataset_id, recipe=payload) as draft:
            info = write_parquet_batches(factory,draft.data_dir/'part-00000.parquet')
            if not info['rows']:
                raise ValueError('Preprocessing removed every row; edit the recipe')
            draft.schema.update(info['schema'])
            version = draft.version
            self.workspace.get(self.dataset.dataset_id,verify=True)
        state = self.state()
        state['processed'] = {'version':version,'rows':info['rows'],'path':'processed/'+version,
                              'purpose':'exploration_only'}
        state.pop('split',None)
        state['history'].append({'processed':state['processed'],'recipe':state['recipe']})
        self._save(state)
        return state['processed']

    def split(self, config, *, validation_fraction=.2):
        state = self.state()
        if not state.get('processed'):
            raise ValueError('Save the processed copy before splitting')
        self.workspace.get_version(self.dataset.dataset_id,state['processed']['version'],verify=True)
        result = prepare_task(self.workspace,self.dataset.dataset_id,self.task(),
            split_config=config,recipe=self.recipe(),batch_size=self.batch_size,
            validation_fraction=validation_fraction)
        state['split'] = {'run_id':result.run_id,'task_id':result.task_id,
            'path':result.directory.relative_to(self.dataset.directory).as_posix(),
            'split_counts':result.split_counts,'prepared_counts':result.prepared_counts,
            'config':config.to_dict(),'validation_fraction':validation_fraction}
        state['history'].append({'split':state['split'],'recipe':state['recipe']})
        self._save(state)
        return result

    def artifacts(self):
        """Inventory includes earlier versions, not just the wizard's latest pointer."""
        result = [('Raw data',self.dataset.directory/'raw')]
        for version in self.workspace.list_versions(self.dataset.dataset_id):
            result.append(('Processed '+version+' (inspection)',self.dataset.directory/'processed'/version))
        for path in sorted((self.dataset.directory/'tasks').glob('*/runs/run-*/manifest.json')):
            result.append(('Split '+path.parent.name,path.parent))
        return result
