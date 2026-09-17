from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from .recipe import Recipe, json_copy
from .transforms import build_transform
from .transforms.base import batches, check_frame
from .validation import validate_recipe


def _source(source, batch_size, loader_options=None):
    from eda_tool.loader import open_dataset
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    dataset = open_dataset(source, **(loader_options or {}))
    return dataset, lambda: dataset.iter_batches(batch_size=batch_size)


def _apply(frame, transforms, names):
    check_frame(frame)
    if set(frame.columns) != set(names):
        raise ValueError('Input columns differ from the fitted recipe; supply the same feature schema')
    result = frame.loc[:, list(names)].copy()
    for transform in transforms:
        result = transform.apply(result)
        check_frame(result)
    return result


@dataclass(frozen=True)
class FittedRecipe:
    recipe: Recipe
    input_columns: tuple[str, ...]
    states: tuple[dict, ...]
    recipe_fingerprint: str
    fit_info: dict

    def _transforms(self):
        if self.recipe.fingerprint != self.recipe_fingerprint:
            raise ValueError('Recipe changed after fitting; refit it')
        validate_recipe(self.recipe, self.input_columns, require_approved=False)
        if len(self.states) != len(self.recipe.active_steps):
            raise ValueError('Fitted state count differs from active steps')
        return [build_transform(s).restore(json_copy(state))
                for s, state in zip(self.recipe.active_steps, self.states)]

    def apply(self, frame):
        return _apply(frame, self._transforms(), self.input_columns)

    def transform_batches(self, source, *, batch_size=50_000, loader_options=None):
        _, factory = _source(source, batch_size, loader_options)
        transforms = self._transforms()
        with batches(factory) as iterator:
            for frame in iterator:
                yield _apply(frame, transforms, self.input_columns)

    def to_dict(self):
        self._transforms()
        return json_copy({'format_version': 1, 'recipe': self.recipe.to_dict(),
                          'input_columns': list(self.input_columns), 'states': list(self.states),
                          'recipe_fingerprint': self.recipe_fingerprint, 'fit_info': self.fit_info})

    @classmethod
    def from_dict(cls, payload):
        value = json_copy(payload)
        if type(value.pop('format_version', None)) is not int or payload['format_version'] != 1:
            raise ValueError('Unsupported fitted recipe format')
        value['recipe'] = Recipe.from_dict(value['recipe'])
        value['input_columns'] = tuple(value['input_columns'])
        value['states'] = tuple(value['states'])
        fitted = cls(**value)
        fitted._transforms()
        return fitted

    def save(self, path):
        payload = self.to_dict()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


def fit_recipe(source, recipe: Recipe, *, batch_size=50_000, loader_options=None,
               require_approved=True, fit_label='explicit fitting source') -> FittedRecipe:
    recipe = Recipe.from_dict(recipe.to_dict())
    dataset, factory = _source(source, batch_size, loader_options)
    names = tuple(dataset.schema())
    validate_recipe(recipe, names, require_approved=require_approved)
    fitted = []
    for step in recipe.active_steps:
        transform = build_transform(step)
        if transform.requires_fit:
            prefix = tuple(fitted)

            def transformed_factory(prefix=prefix):
                with batches(factory) as iterator:
                    for frame in iterator:
                        yield _apply(frame, prefix, names)

            # Each learned stage sees earlier stages already fitted/applied.
            transform.fit(transformed_factory)
        fitted.append(transform)
    return FittedRecipe(recipe, names, tuple(t.state() for t in fitted), recipe.fingerprint,
                        {'label': fit_label, 'batch_size': batch_size,
                         'learned_stages': sum(t.requires_fit for t in fitted),
                         'method': 'one replayable-source pass per learned stage',
                         'split_policy': 'caller-selected source; no automatic splitting'})


@dataclass(frozen=True)
class ExecutionResult:
    dataset_id: str
    version: str
    directory: Path
    input_rows: int
    output_rows: int
    fitted: FittedRecipe


def execute_recipe(workspace, dataset_id, recipe: Recipe, *, source_version='raw',
                   fitted: FittedRecipe | None = None, fit_source=None,
                   batch_size=50_000, loader_options=None, fit_loader_options=None,
                   verify_source=True, progress=None) -> ExecutionResult:
    from data.manifest import resolve_inside, sha256_file
    import pyarrow as pa
    import pyarrow.parquet as pq

    if fitted is not None and fit_source is not None:
        raise ValueError('Supply fitted or fit_source, not both')
    recipe = Recipe.from_dict(recipe.to_dict())
    dataset = workspace.get(dataset_id, verify=verify_source and source_version == 'raw')
    if source_version == 'raw':
        source = list(dataset.raw_files)
        source_manifest_path = dataset.directory/'manifest.json'
    else:
        manifest = workspace.get_version(dataset_id, source_version, verify=verify_source)
        directory = resolve_inside(dataset.directory, 'processed/'+source_version)
        source = [resolve_inside(directory, f.path) for f in manifest.files]
        source_manifest_path = directory/'manifest.json'
    original_digest = sha256_file(source_manifest_path)
    loaded, factory = _source(source, batch_size, loader_options)
    names = tuple(loaded.schema())
    validation = validate_recipe(recipe, names)
    if fitted is None:
        if validation.requires_fit and fit_source is None:
            raise ValueError('Learned transformations require fit_source or an existing FittedRecipe. '
                             'Use training data when preparing a model.')
        fitted = fit_recipe(source if fit_source is None else fit_source, recipe, batch_size=batch_size,
                            loader_options=loader_options if fit_source is None else fit_loader_options)
    if fitted.recipe_fingerprint != recipe.fingerprint or set(fitted.input_columns) != set(names):
        raise ValueError('Fitted recipe does not match the approved recipe and input columns')
    fitted = FittedRecipe.from_dict(fitted.to_dict())
    transforms = fitted._transforms()
    payload = {'recipe': recipe.to_dict(), 'fitted': fitted.to_dict()}
    input_rows = output_rows = 0
    with workspace.processed_version(dataset_id, recipe=payload, source_version=source_version) as draft:
        if sha256_file(source_manifest_path) != original_digest:
            raise ValueError('Source metadata changed during preparation; retry')
        writer = None
        arrow_schema = None
        empty_frame = None
        try:
            with batches(factory) as iterator:
                for batch in iterator:
                    input_rows += len(batch)
                    output = _apply(batch, transforms, fitted.input_columns)
                    output_rows += len(output)
                    if empty_frame is None:
                        empty_frame = output.iloc[:0]
                    if len(output):
                        table = pa.Table.from_pandas(output, preserve_index=False)
                        if writer is None:
                            arrow_schema = table.schema
                            writer = pq.ParquetWriter(draft.data_dir/'part-00000.parquet', arrow_schema)
                            draft.schema.update({c: str(t) for c, t in output.dtypes.items()})
                        elif not table.schema.equals(arrow_schema, check_metadata=False):
                            raise ValueError('Output types changed across batches. Add explicit convert_type '
                                             'steps or stable loader dtypes before exporting.')
                        writer.write_table(table)
                    if progress:
                        progress({'stage': 'processing', 'input_rows': input_rows, 'output_rows': output_rows})
            if writer is None:
                if empty_frame is None:
                    empty_frame = pd.DataFrame(columns=list(validation.output_columns))
                pq.write_table(pa.Table.from_pandas(empty_frame, preserve_index=False),
                               draft.data_dir/'part-00000.parquet')
                draft.schema.update({c: str(t) for c, t in empty_frame.dtypes.items()})
        finally:
            if writer is not None:
                writer.close()
        version = draft.version
    return ExecutionResult(dataset_id, version, dataset.directory/'processed'/version,
                           input_rows, output_rows, fitted)
