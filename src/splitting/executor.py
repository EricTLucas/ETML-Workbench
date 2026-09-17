from contextlib import ExitStack
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pyarrow as pa

from eda_tool.loader import open_dataset
from preprocessing.executor import ParquetBatchWriter
from preprocessing.transforms.base import batches
from tasks.validation import ROW_ID, eligible_rows, validate_task
from preprocessing import Recipe
from .strategies import NAMES, assign, priority, scalar_key


def split_dataset(source, directory, task, config, *, batch_size=50_000, loader_options=None, progress=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'splits').exists() or (directory/'assignments').exists():
        raise FileExistsError('Split outputs already exist')
    dataset = open_dataset(source, **(loader_options or {}))
    validate_task(task, dataset.schema(), config, Recipe())
    factory = lambda: dataset.iter_batches(batch_size=batch_size)
    database = directory/'.assignments.sqlite'
    if database.exists():
        raise FileExistsError(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute('PRAGMA temp_store=FILE')
        connection.execute('PRAGMA cache_size=-8192')
        connection.execute('CREATE TABLE rows(row_id INTEGER PRIMARY KEY, eligible INTEGER, label TEXT, '
                           'priority TEXT, group_key TEXT, time_key INTEGER, split TEXT DEFAULT "excluded")')
        total, schema, labels = 0, None, set()
        with batches(factory) as iterator:
            for frame in iterator:
                keep = eligible_rows(frame, task)
                current = pa.Table.from_pandas(frame, preserve_index=False).schema.remove_metadata()
                schema = current if schema is None else pa.unify_schemas([schema, current], promote_options='permissive')
                timestamps = None
                if config.strategy == 'chronological':
                    timestamps = pd.to_datetime(frame.loc[keep, config.time_column], format=config.time_format,
                                                utc=True, errors='raise')
                    if timestamps.isna().any():
                        raise ValueError('Time split column has missing values')
                    timestamps = iter(timestamps)
                records = []
                for i in range(len(frame)):
                    row_id = total+i
                    label, group, time = None, None, None
                    valid = bool(keep.iloc[i])
                    if valid:
                        if task.task_type == 'classification':
                            label = scalar_key(frame[task.target].iloc[i])
                            labels.add(label)
                            if len(labels) > config.max_classes:
                                raise ValueError('Target exceeds max_classes; check task type or raise the cap')
                        if config.strategy == 'group':
                            value = frame[config.group_column].iloc[i]
                            if pd.isna(value):
                                raise ValueError('Group split column has missing values')
                            group = scalar_key(value)
                        if timestamps is not None:
                            time = int(next(timestamps).value)
                    records.append((row_id, int(valid), label, priority(config.seed, row_id), group, time))
                connection.executemany('INSERT INTO rows(row_id,eligible,label,priority,group_key,time_key) '
                                       'VALUES(?,?,?,?,?,?)', records)
                total += len(frame)
                connection.commit()
                if progress:
                    progress({'stage': 'indexing', 'rows': total})
        if task.task_type == 'classification' and len(labels) < 2:
            raise ValueError('Classification requires at least two observed target classes')
        for index in ('priority', 'label', 'group_key', 'time_key'):
            connection.execute(f'CREATE INDEX idx_{index} ON rows({index})')
        counts = assign(connection, config)
        train_labels = {x[0] for x in connection.execute('SELECT DISTINCT label FROM rows WHERE split="train"')}
        if task.task_type == 'classification' and train_labels != labels:
            raise ValueError('Training split is missing target classes; use stratification or change the split')
        class_counts = {name: dict(connection.execute('SELECT label,count(*) FROM rows WHERE split=? '
                                                     'GROUP BY label ORDER BY label', (name,)))
                        for name in NAMES} if labels else {}
        schema = schema.append(pa.field(ROW_ID, pa.int64()))
        assignment_schema = pa.schema([(ROW_ID, pa.int64()), ('split', pa.string())])
        with ExitStack() as stack:
            writers = {name: stack.enter_context(ParquetBatchWriter(directory/'splits'/name/'part-00000.parquet', schema))
                       for name in NAMES}
            assignment_writer = stack.enter_context(ParquetBatchWriter(
                directory/'assignments'/'part-00000.parquet', assignment_schema))
            offset = 0
            with batches(factory) as iterator:
                for frame in iterator:
                    records = connection.execute('SELECT row_id,split FROM rows WHERE row_id>=? AND row_id<? '
                                                 'ORDER BY row_id', (offset, offset+len(frame))).fetchall()
                    if len(records) != len(frame):
                        raise ValueError('Source row count changed across splitting passes')
                    assignments = pd.DataFrame(records, columns=[ROW_ID, 'split'])
                    assignment_writer.write(assignments)
                    frame = frame.reset_index(drop=True).copy()
                    frame[ROW_ID] = np.arange(offset, offset+len(frame), dtype=np.int64)
                    for name in NAMES:
                        writers[name].write(frame.loc[assignments['split'].eq(name).to_numpy()])
                    offset += len(frame)
                if offset != total:
                    raise ValueError('Source row count changed across splitting passes')
        return {'source_rows': total, 'counts': counts, 'class_counts': class_counts,
                'row_id': 'zero-based ordinal in ordered source files; bound to source checksums',
                'assignment_method': 'sha256 seeded priorities; SQLite disk sorting',
                'proportions': 'exact allocation with per-class rounding for stratification' if config.strategy in {'random', 'stratified'}
                               else 'approximate to preserve groups or tied timestamps'}
    finally:
        connection.close()
        database.unlink(missing_ok=True)
