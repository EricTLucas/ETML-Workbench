from dataclasses import dataclass
from pathlib import Path

from data import DatasetWorkspace
from data.manifest import resolve_inside, write_json
from eda_tool.loader import open_dataset
from preprocessing import Recipe, FittedRecipe
from preprocessing.recipe import json_copy
from preprocessing.executor import fit_batches, write_parquet_batches
from preprocessing.transforms.base import batches
from splitting import SplitConfig, split_dataset
from splitting.strategies import NAMES, scalar_key
from tasks import TaskConfig, ROW_ID, validate_task
from .preprocessing import PreprocessingWorkflow


@dataclass(frozen=True)
class PreparationResult:
    dataset_id: str
    task_id: str
    run_id: str
    directory: Path
    split_counts: dict
    prepared_counts: dict
    feature_columns: tuple[str, ...]
    target: str
    fitted: FittedRecipe


def prepare_task(workspace, dataset_id, task: TaskConfig, *, split_config=None, recipe=None,
                 source_version='raw', allow_processed_source=False, loader_options=None,
                 batch_size=50_000, progress=None, validation_fraction=.2):
    workspace = workspace if isinstance(workspace, DatasetWorkspace) else DatasetWorkspace(workspace)
    split_config = split_config or SplitConfig(strategy='stratified' if task.task_type == 'classification' else 'random')
    recipe = Recipe.from_dict((recipe or Recipe()).to_dict())
    loader_options = json_copy(loader_options or {})
    if source_version != 'raw' and not allow_processed_source:
        raise ValueError('Use raw data to avoid pre-split fitting leakage. A processed source requires '
                         'allow_processed_source=True and caller confirmation that cleanup was fixed, not learned.')
    workflow = PreprocessingWorkflow(workspace)
    _, source, identity = workflow._source(dataset_id, source_version, verify=True)
    stored=workspace.get(dataset_id)
    split_sources_input={role:resolve_inside(stored.directory,path) for role,path in stored.manifest.split_files.items()} if source_version=='raw' else {}
    dataset = open_dataset(split_sources_input.get('train',source), **(loader_options or {}))
    features = validate_task(task, dataset.schema(), split_config, recipe)
    workspace.create_task(dataset_id, task)
    with workspace.task_run(dataset_id, task.task_id) as draft:
        _, _, current = workflow._source(dataset_id, source_version, verify=True)
        if current != identity:
            raise ValueError('Source changed during task preparation')
        write_json(draft.directory/'split_config.json', split_config.to_dict())
        if split_sources_input:
            from splitting.presplit import split_presplit
            stats=split_presplit(split_sources_input,draft.directory,task,split_config,
                validation_fraction=validation_fraction,batch_size=batch_size,loader_options=loader_options,progress=progress)
        else:
            stats = split_dataset(source, draft.directory, task, split_config, batch_size=batch_size,
                                  loader_options=loader_options, progress=progress)
        # Verify the second source-reading pass used the original source bytes.
        _, _, current = workflow._source(dataset_id, source_version, verify=True)
        if current != identity:
            raise ValueError('Source changed while splitting')
        split_sources = {name: open_dataset(draft.directory/'splits'/name/'part-00000.parquet') for name in NAMES}

        def train_features():
            with batches(lambda: split_sources['train'].iter_batches(batch_size=batch_size)) as iterator:
                for frame in iterator:
                    yield frame.loc[:, list(features)]

        if progress:
            progress({'stage': 'fitting', 'rows': stats['counts']['train']})
        fitted = fit_batches(train_features, features, recipe, batch_size=batch_size,
                             fit_label=f'{dataset_id}/{task.task_id}/{draft.run_id}/train')
        prep = draft.directory/'preprocessing'
        prep.mkdir()
        recipe.save(prep/'recipe.json')
        fitted.save(prep/'fitted.json')
        prepared_counts, prepared_schemas, prepared_class_counts = {}, {}, {}
        from preprocessing.validation import validate_recipe
        output_features = validate_recipe(recipe, features).output_columns
        for name in NAMES:
            def transformed(name=name):
                emitted = False
                with batches(lambda: split_sources[name].iter_batches(batch_size=batch_size)) as iterator:
                    for frame in iterator:
                        frame = frame.set_index(ROW_ID, drop=False)
                        output = fitted.apply(frame.loc[:, list(features)])
                        if not output.index.is_unique or not output.index.isin(frame.index).all():
                            raise ValueError('Transformation changed row identity')
                        output = output.copy()
                        output[task.target] = frame.loc[output.index, task.target]
                        output[ROW_ID] = output.index.to_numpy(dtype='int64')
                        emitted = True
                        yield output.reset_index(drop=True)
                if not emitted:
                    # Preserve the source schema for disabled (zero-fraction) splits.
                    import pyarrow.parquet as pq
                    frame = pq.read_table(draft.directory/'splits'/name/'part-00000.parquet').to_pandas()
                    output = fitted.apply(frame.loc[:, list(features)])
                    output[task.target] = frame[task.target]
                    output[ROW_ID] = frame[ROW_ID]
                    yield output
            info = write_parquet_batches(transformed, draft.directory/'prepared'/name/'part-00000.parquet')
            if stats['counts'][name] and not info['rows']:
                raise ValueError(f'Preprocessing removed every row from {name}')
            prepared_counts[name] = info['rows']
            prepared_schemas[name] = info['schema']
            if task.task_type == 'classification' and not (name=='test' and not stats.get('test_labeled',True)):
                from collections import Counter
                counts = Counter()
                prepared_source = open_dataset(draft.directory/'prepared'/name/'part-00000.parquet')
                with batches(lambda: prepared_source.iter_batches(columns=[task.target], batch_size=batch_size)) as iterator:
                    for frame in iterator:
                        counts.update(scalar_key(value) for value in frame[task.target])
                prepared_class_counts[name] = dict(counts)
                if name == 'train' and set(counts) != set(stats['class_counts']['train']):
                    raise ValueError('Preprocessing removed an entire target class from training')
            if progress:
                progress({'stage': 'prepared_'+name, 'rows': info['rows']})
        draft.metadata.update(source_version=source_version, source_fingerprint=identity,
                              presplit=bool(split_sources_input),test_labeled=stats.get('test_labeled',True),
                              processed_source_override=allow_processed_source,
                              feature_columns=list(output_features), target=task.target, task_type=task.task_type,
                              split_statistics=stats, prepared_counts=prepared_counts,
                              removed_by_preprocessing={n: stats['counts'][n]-prepared_counts[n] for n in NAMES},
                              prepared_schemas=prepared_schemas,
                              prepared_class_counts=prepared_class_counts, loader_options=loader_options,
                              fitting_scope='train features only; target excluded',
                              split_controls_excluded=[c for c in (split_config.group_column,split_config.time_column) if c],
                              recipe_fingerprint=recipe.fingerprint)
        run_id = draft.run_id
    root = workspace.get(dataset_id).directory/'tasks'/task.task_id/'runs'/run_id
    return PreparationResult(dataset_id, task.task_id, run_id, root, stats['counts'], prepared_counts,
                             tuple(output_features), task.target, fitted)
