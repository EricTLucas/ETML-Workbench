from collections import Counter
from dataclasses import replace
from pathlib import Path
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from eda_tool.loader import open_dataset
from preprocessing.executor import write_parquet_batches
from preprocessing.transforms.base import batches
from tasks.validation import ROW_ID,eligible_rows
from .executor import split_dataset
from .strategies import NAMES,scalar_key


def split_presplit(sources,directory,task,config,*,validation_fraction=.2,batch_size=50_000,loader_options=None,progress=None):
    if not 0<validation_fraction<1:
        raise ValueError('validation_fraction must be between zero and one')
    directory=Path(directory)
    train_source=open_dataset(sources['train'],**(loader_options or {}))
    columns=list(train_source.schema())
    controls = replace(config,train=1. if 'validation' in sources else 1-validation_fraction,
                       validation=0. if 'validation' in sources else validation_fraction,test=0.)
    stats=split_dataset(sources['train'],directory,task,controls,batch_size=batch_size,
                        loader_options=loader_options,progress=progress)
    offset=stats['source_rows']
    assignments=[]
    test_labeled=True
    for role in ('validation','test'):
        if role not in sources: continue
        source=open_dataset(sources[role],**(loader_options or {}))
        names=set(source.schema())
        unlabeled=role=='test' and task.target not in names
        if names != set(columns)-({task.target} if unlabeled else set()):
            raise ValueError(f'{role} columns must match training (test may omit the target)')
        if role=='test': test_labeled=not unlabeled
        start=offset
        size=[0,0]
        def frames(source=source,unlabeled=unlabeled,start=start):
            raw,kept=0,0
            with batches(lambda:source.iter_batches(batch_size=batch_size)) as iterator:
                for frame in iterator:
                    frame=frame.reset_index(drop=True).copy()
                    if unlabeled: frame[task.target]=None
                    keep=np.ones(len(frame),dtype=bool) if unlabeled else eligible_rows(frame,task).to_numpy()
                    frame[ROW_ID]=np.arange(start+raw,start+raw+len(frame),dtype=np.int64)
                    raw+=len(frame); kept+=int(keep.sum())
                    yield frame.loc[keep,[*columns,ROW_ID]]
            size[:]=[raw,kept]
        temporary=directory/'splits'/role/'external.parquet'
        info=write_parquet_batches(frames,temporary)
        if not info['rows']: raise ValueError(f'Provided {role} file has no usable rows')
        os.replace(temporary,directory/'splits'/role/'part-00000.parquet')
        assignments.append((role,source,start,unlabeled))
        offset+=size[0]
        stats['counts'][role]=size[1]
        stats['counts']['excluded']+=size[0]-size[1]
    old_assign=directory/'assignments/part-00000.parquet'
    def assignment_frames():
        source=open_dataset(old_assign)
        with batches(lambda:source.iter_batches(batch_size=batch_size)) as iterator:
            yield from iterator
        for role,source,start,unlabeled in assignments:
            offset=start
            with batches(lambda:source.iter_batches(batch_size=batch_size)) as iterator:
                for frame in iterator:
                    keep=np.ones(len(frame),dtype=bool) if unlabeled else eligible_rows(frame,task).to_numpy()
                    yield pd.DataFrame({ROW_ID:np.arange(offset,offset+len(frame),dtype=np.int64),
                                        'split':np.where(keep,role,'excluded')})
                    offset+=len(frame)
    path=directory/'assignments/combined.parquet'
    write_parquet_batches(assignment_frames,path)
    os.replace(path,old_assign)
    if task.task_type=='classification':
        for role in NAMES:
            if role=='test' and not test_labeled:
                stats['class_counts'][role]={}; continue
            counts=Counter()
            source=open_dataset(directory/'splits'/role/'part-00000.parquet')
            with batches(lambda:source.iter_batches(columns=[task.target],batch_size=batch_size)) as iterator:
                for frame in iterator: counts.update(scalar_key(v) for v in frame[task.target])
            stats['class_counts'][role]=dict(counts)
        training_classes=set(stats['class_counts']['train'])
        if any(set(stats['class_counts'][role])-training_classes for role in ('validation','test')):
            raise ValueError('A supplied evaluation split has classes absent from training')
    stats.update(source_rows=offset,presplit=True,test_labeled=test_labeled,
                 assignment_method='Provided split roles preserved; validation carved only from train when absent',
                 validation_fraction=None if 'validation' in sources else validation_fraction,
                 row_id='Ordinal across train, supplied validation, supplied test; original files remain preserved')
    return stats
